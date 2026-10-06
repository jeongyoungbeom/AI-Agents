from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, RunPhase, StageContract
from app.services.sandbox import DockerSandboxCancelled, DockerSandboxError, DockerSandboxTimeout
from app.services.verification import VerificationCancelled, VerificationFailureKind as Kind, VerificationRunner
from tests.pipeline.support import FakeRoleRunner, HostGitSandbox, build_pipeline, create_repository, git, temporary_directory


def parent_results(store, run_id):
    parent = next(item['data']['session_run_id'] for item in store.list_messages(run_id)
                  if item['kind'] == 'pipeline_parent')
    return [item for item in store.list_messages(parent) if item['kind'] == 'pipeline_result']


def build_with_commands(root, source, runner, commands):
    def contract(**kwargs):
        return StageContract(**{**kwargs, 'verification_commands': commands})
    with patch('tests.pipeline.support.StageContract', side_effect=contract):
        return build_pipeline(root, source, runner)


class FailureClassificationTests(unittest.TestCase):
    def classify(self, stdout='', stderr='', code=1):
        return VerificationRunner._failure_kind(stdout, stderr, code)

    def test_handled_stdout_import_does_not_override_terminal_assertion(self):
        self.assertEqual(Kind.TEST_FAILURE, self.classify(
            "ModuleNotFoundError: No module named 'optional'",
            'Traceback (most recent call last):\nAssertionError: feature remains draft'))

    def test_handled_stderr_import_does_not_override_terminal_assertion(self):
        self.assertEqual(Kind.TEST_FAILURE, self.classify(stderr=
            "ModuleNotFoundError: No module named 'optional'\n"
            'AssertionError: feature remains draft'))

    def test_assertion_diagnostic_words_are_code_failure(self):
        for words in ('command not found', 'executable file not found', 'no module named',
                      'cannot find module', 'could not resolve', 'offline mode'):
            with self.subTest(words=words):
                self.assertEqual(Kind.TEST_FAILURE, self.classify(stderr='AssertionError: ' + words))

    def test_general_logs_are_not_proof_of_environment_failure(self):
        for words in ('command not found', 'no module named', 'module not found', 'could not resolve'):
            with self.subTest(words=words):
                self.assertEqual(Kind.TEST_FAILURE, self.classify(stdout='previous log: ' + words))
                self.assertEqual(Kind.TEST_FAILURE, self.classify(stderr='previous log: ' + words))

    def test_actual_missing_dependency_keeps_environment_classification(self):
        for diagnostic in ("ModuleNotFoundError: No module named 'required'",
                           'ImportError: No module named required',
                           "Error: Cannot find module 'required'",
                           '> Could not resolve required:dependency:1.0',
                           '> No cached version of required available for offline mode.'):
            with self.subTest(diagnostic=diagnostic):
                self.assertEqual(Kind.DEPENDENCY_UNAVAILABLE, self.classify(stderr=diagnostic))
        self.assertEqual(Kind.TEST_FAILURE, self.classify(stderr=
            "ImportError: cannot import name 'value' from partially initialized module 'fixture'"))

    def test_pytest_import_collection_error_on_stdout(self):
        self.assertEqual(Kind.DEPENDENCY_UNAVAILABLE, self.classify(
            stdout="E   ModuleNotFoundError: No module named 'required'"))

    def test_actual_missing_tool(self):
        self.assertEqual(Kind.TOOL_UNAVAILABLE, self.classify(code=127))
        self.assertEqual(Kind.TOOL_UNAVAILABLE, self.classify(stderr='exec: "python": executable file not found in $PATH', code=126))

    def test_success_with_dependency_warning_remains_passed(self):
        self.assertEqual(Kind.PASSED, self.classify("ModuleNotFoundError: No module named 'optional'", code=0))


class PipelineResolutionTests(unittest.TestCase):
    def test_optional_import_warning_with_assertion_calls_finisher_once_and_completes(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            (source / 'verify.py').write_text(
                "from pathlib import Path\ntry:\n    import optional_s09_fixture\n"
                "except ModuleNotFoundError as exc:\n    print(type(exc).__name__ + ':', exc)\n"
                "assert Path('feature.txt').read_text().strip() == 'fixed', 'no module named; command not found'\n",
                encoding='utf-8')
            git(source, 'add', 'verify.py')
            git(source, 'commit', '-m', 'handled optional import with required assertion')
            runner = FakeRoleRunner(review_responses=[[]])
            store, worker, run_id = build_pipeline(root, source, runner)
            observed = []
            original = worker.coordinator.verifier.run_all
            def verify(*args, **kwargs):
                results = original(*args, **kwargs)
                observed.append(results)
                return results
            with patch.object(worker.coordinator.verifier, 'run_all', side_effect=verify):
                self.assertTrue(worker.run_once())
            self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])
            self.assertEqual(1, [role for role, _ in runner.calls].count(RoleId.IMPROVEMENT))
            self.assertFalse(worker.coordinator.budget.can_retry(run_id, 'stage-001', 'verification_failure'))
            result = parent_results(store, run_id)[-1]
            self.assertEqual('passed', result['data']['execution_evidence']['stage-001']['commands'][0]['failure_kind'])
            self.assertEqual(git(source, 'rev-parse', 'HEAD'), result['data']['validated_sha'])
            self.assertEqual(2, len(observed))
            self.assertEqual(Kind.TEST_FAILURE, observed[0][0].failure_kind)
            self.assertEqual(1, observed[0][0].return_code)
            self.assertIn('AssertionError', observed[0][0].stderr)
            self.assertEqual(0, observed[1][0].return_code)

    def test_required_dependency_failure_pauses_without_code_retry(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            (source / 'verify.py').write_text('import required_missing_s09_fixture\n', encoding='utf-8')
            git(source, 'add', 'verify.py')
            git(source, 'commit', '-m', 'required dependency unavailable')
            runner = FakeRoleRunner(development_outputs=['fixed'], review_responses=[[]])
            store, worker, run_id = build_pipeline(root, source, runner)
            worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual('NEEDS_ATTENTION', store.pipeline_job(run_id)['status'])
            self.assertNotIn(RoleId.IMPROVEMENT, [role for role, _ in runner.calls])
            self.assertTrue(worker.coordinator.budget.can_retry(run_id, 'stage-001', 'verification_failure'))
            self.assertEqual(Kind.DEPENDENCY_UNAVAILABLE, parent_results(store, run_id)[-1]['data']['execution_evidence']['stage-001']['commands'][0]['failure_kind'])

    def _startup_failure_and_recovery(self, failed_index):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            for name in ('first.py', 'last.py'):
                (source / name).write_text('assert True\n', encoding='utf-8')
            git(source, 'add', '-A')
            git(source, 'commit', '-m', 'multiple verification commands')
            commands = ('python first.py', 'python verify.py', 'python last.py')
            runner = FakeRoleRunner(development_outputs=['fixed'], review_responses=[[]])
            store, worker, run_id = build_with_commands(root, source, runner, commands)
            base = git(source, 'rev-parse', 'HEAD')
            original = worker.coordinator.verifier.sandbox.popen
            attempted = []
            def startup(repository, argv, **kwargs):
                if kwargs.get('component') == 'verification':
                    attempted.append(tuple(argv))
                    if len(attempted) == failed_index + 1:
                        raise DockerSandboxError('fixture: verification container unavailable')
                return original(repository, argv, **kwargs)
            with patch.object(worker.coordinator.verifier.sandbox, 'popen', side_effect=startup):
                worker.run_once()
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            self.assertEqual('NEEDS_ATTENTION', store.pipeline_job(run_id)['status'])
            self.assertEqual(base, git(source, 'rev-parse', 'HEAD'))
            before = store.pipeline_workspace(run_id)
            candidate = before['candidate_sha']
            worktree = Path(before['worktree']['worktree_path'])
            self.assertEqual(candidate, git(worktree, 'rev-parse', 'HEAD'))
            self.assertEqual('fixed\n', (worktree / 'feature.txt').read_text())
            self.assertEqual(0, store.reserved_token_total(run_id))
            self.assertEqual(240, store.usage_total(run_id))
            self.assertTrue(worker.coordinator.budget.can_retry(run_id, 'stage-001', 'verification_failure'))
            result = parent_results(store, run_id)[-1]
            evidence = result['data']['execution_evidence']['stage-001']
            self.assertEqual(candidate, evidence['candidate_sha'])
            self.assertEqual(['feature.txt'], evidence['changed_files'])
            self.assertEqual(list(commands[failed_index:]), evidence['unperformed_commands'])
            self.assertEqual(failed_index + 1, len(evidence['commands']))
            for command in evidence['commands'][:failed_index]:
                self.assertTrue(command['started'])
                self.assertEqual(0, command['return_code'])
            failure = evidence['commands'][-1]
            self.assertFalse(failure['started'])
            self.assertIsNone(failure['return_code'])
            self.assertEqual(Kind.ENVIRONMENT_UNAVAILABLE, failure['failure_kind'])
            self.assertIn('시작 실패: ' + commands[failed_index], result['content'])
            self.assertIn('미수행:', result['content'])
            artifact = json.loads((root / 'artifacts' / run_id / 'stages/stage-001/verification.json').read_text(encoding='utf-8'))
            self.assertFalse(artifact['code_repair_attempted'])
            self.assertFalse(artifact['commands'][-1]['started'])
            self.assertNotIn(RoleId.IMPROVEMENT, [role for role, _ in runner.calls])
            # The gateway's approved resume path restarts on the retained workspace.
            worker.machine.transition(store.load_run(run_id), RunPhase.DEVELOPING, message='환경 복구 후 재개')
            self.assertEqual('QUEUED', worker.resume(run_id))
            observed_worktrees = []
            def restored(repository, argv, **kwargs):
                if kwargs.get('component') == 'verification':
                    observed_worktrees.append(Path(repository))
                return original(repository, argv, **kwargs)
            with patch.object(worker.coordinator.verifier.sandbox, 'popen', side_effect=restored):
                worker.run_once()
            self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase)
            self.assertEqual([worktree] * len(commands), observed_worktrees)
            self.assertEqual(candidate, git(source, 'rev-parse', 'HEAD'))
            self.assertEqual(candidate, store.pipeline_workspace(run_id)['validated_sha'])
            self.assertEqual(480, store.usage_total(run_id))
            self.assertEqual(0, store.reserved_token_total(run_id))
            completed = parent_results(store, run_id)[-1]
            self.assertEqual([], completed['data']['execution_evidence']['stage-001']['unperformed_commands'])
            self.assertTrue(all(item['passed'] for item in completed['data']['execution_evidence']['stage-001']['commands']))

    def test_first_verification_startup_failure_preserves_candidate_and_resumes(self):
        self._startup_failure_and_recovery(0)

    def test_later_verification_startup_failure_preserves_completed_commands_and_resumes(self):
        self._startup_failure_and_recovery(1)

    def test_general_startup_programming_error_remains_failed(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = FakeRoleRunner(development_outputs=['fixed'], review_responses=[[]])
            store, worker, run_id = build_pipeline(root, source, runner)
            with patch.object(worker.coordinator.verifier, 'run_all', side_effect=ValueError('fixture implementation bug')):
                worker.run_once()
            self.assertEqual('FAILED', store.pipeline_job(run_id)['status'])
            self.assertEqual(RunPhase.FAILED, store.load_run(run_id).phase)

    def test_verification_cancellation_is_not_environment_failure(self):
        with temporary_directory() as directory:
            source = create_repository(Path(directory))
            from app.services.git import GitRepository
            sandbox = HostGitSandbox()
            with patch.object(sandbox, 'popen') as start, self.assertRaises(VerificationCancelled):
                VerificationRunner(sandbox=sandbox).run_all(GitRepository(source, sandbox), ('python verify.py',), cancelled=lambda: True)
            start.assert_not_called()

    def test_verification_timeout_preserves_timeout_kind(self):
        with temporary_directory() as directory:
            source = create_repository(Path(directory))
            (source / 'sleep.py').write_text('import time\ntime.sleep(30)\n', encoding='utf-8')
            git(source, 'add', 'sleep.py')
            git(source, 'commit', '-m', 'timeout fixture')
            from app.services.git import GitRepository
            sandbox = HostGitSandbox()
            result = VerificationRunner(sandbox=sandbox, timeout_seconds=.05, poll_seconds=.02).run_all(
                GitRepository(source, sandbox), ('python sleep.py',))
            self.assertEqual(Kind.TIMEOUT, result[0].failure_kind)
            self.assertEqual(124, result[0].return_code)
            self.assertTrue(result[0].started)


class StartupResultTests(unittest.TestCase):
    class Repository:
        path = Path('.')
        def snapshot(self):
            return object()
        def assert_position(self, snapshot, *, allow_worktree_changes):
            assert not allow_worktree_changes

    def test_cancelled_startup_is_not_swallowed_as_environment_failure(self):
        sandbox = HostGitSandbox()
        with patch.object(sandbox, 'popen', side_effect=DockerSandboxCancelled('cancelled')), self.assertRaises(VerificationCancelled):
            VerificationRunner(sandbox=sandbox).run_all(self.Repository(), ('python verify.py',))

    def test_timeout_during_startup_has_no_exit_and_preserves_timeout_kind(self):
        sandbox = HostGitSandbox()
        with patch.object(sandbox, 'popen', side_effect=DockerSandboxTimeout('startup timeout')):
            results = VerificationRunner(sandbox=sandbox).run_all(self.Repository(), ('python verify.py',))
        self.assertEqual(Kind.TIMEOUT, results[0].failure_kind)
        self.assertFalse(results[0].started)
        self.assertIsNone(results[0].return_code)

    def test_unstarted_result_has_no_exit_and_stops_later_commands(self):
        sandbox = HostGitSandbox()
        with patch.object(sandbox, 'popen', side_effect=DockerSandboxError('unavailable')) as start:
            results = VerificationRunner(sandbox=sandbox).run_all(self.Repository(), ('python verify.py', 'python later.py'))
        self.assertEqual(1, start.call_count)
        self.assertFalse(results[0].to_dict()['started'])
        self.assertFalse(results[0].passed)
        self.assertIsNone(results[0].to_dict()['return_code'])
        self.assertEqual(Kind.ENVIRONMENT_UNAVAILABLE, results[0].failure_kind)
        from app.pipeline.coordinator import PipelineCoordinator
        failure = PipelineCoordinator._verification_failure_text(results)
        self.assertIn('시작 실패(종료 코드 없음)', failure)
        self.assertNotIn('종료 코드: None', failure)


if __name__ == '__main__':
    unittest.main()
