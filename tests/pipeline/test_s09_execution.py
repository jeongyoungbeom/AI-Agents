from __future__ import annotations

import json
import base64
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.contracts import RoleId, TokenUsage
from app.services.git import GitRepository, GitRepositoryError
from app.services.hermes import HermesExecutionError, HermesResult, HermesRunner, HermesSettings
from app.services.hermes.workspace import ModelWorkspace, ModelWorkspaceError
from app.services.sandbox import DEFAULT_SECURE_DOCKER_IMAGE
from app.services.verification import VerificationFailureKind, VerificationResult, VerificationRunner
from app.storage import ArtifactStore
from tests.pipeline.support import FakeRoleRunner, HostGitSandbox, build_pipeline, create_repository, git, temporary_directory
from tests.pipeline.test_hermes_contract import FakeProcess
from app.services.budget import conservative_prompt_tokens


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repository = self.root / 'source'
        self.repository.mkdir()
        (self.repository / '.git').write_text('gitdir: D:/private/original/.git/worktrees/w\n')
        (self.repository / 'feature.txt').write_text('draft\n')
        (self.repository / 'outside.txt').write_text('preserve\n')

    def workspace(self, **kwargs):
        return ModelWorkspace(self.repository, self.root / 'models', image=DEFAULT_SECURE_DOCKER_IMAGE, **kwargs)

    def test_review_snapshot_contains_exact_patch_and_no_git_metadata(self):
        review = {'base_sha': 'a' * 40, 'candidate_sha': 'b' * 40,
                  'patch': 'diff --git a/feature.txt b/feature.txt\n+한글\n'}
        workspace = self.workspace(review=review)
        self.assertFalse((workspace.path / '.git').exists())
        self.assertEqual(review['patch'], (workspace.path / '.ai-agents-review/diff.patch').read_text(encoding='utf-8'))
        policy = json.loads(workspace.policy.read_text())
        self.assertEqual([], policy['scope'])
        self.assertEqual(DEFAULT_SECURE_DOCKER_IMAGE, policy['image'])

    def test_scoped_files_and_new_directory_are_preserved(self):
        workspace = self.workspace(scope=('feature.txt', 'new/'))
        (workspace.path / 'feature.txt').write_text('fixed\n')
        (workspace.path / 'new/test.py').write_text('assert True\n')
        self.assertEqual(('feature.txt', 'new/test.py'), workspace.preserve_changes())
        self.assertEqual('fixed\n', (self.repository / 'feature.txt').read_text())
        self.assertEqual('preserve\n', (self.repository / 'outside.txt').read_text())

    def test_untouched_missing_file_placeholder_does_not_create_source_file(self):
        workspace = self.workspace(scope=('missing.txt',))
        self.assertEqual((), workspace.preserve_changes())
        self.assertFalse((self.repository / 'missing.txt').exists())

    def test_directory_deletion_is_preserved(self):
        (self.repository / 'src').mkdir()
        (self.repository / 'src/obsolete.py').write_text('obsolete\n')
        workspace = self.workspace(scope=('src/',))
        (workspace.path / 'src/obsolete.py').unlink()
        self.assertEqual(('src/obsolete.py',), workspace.preserve_changes())
        self.assertFalse((self.repository / 'src/obsolete.py').exists())

    def test_out_of_scope_snapshot_edit_is_rejected_before_any_copy(self):
        workspace = self.workspace(scope=('feature.txt',))
        (workspace.path / 'feature.txt').write_text('fixed\n')
        (workspace.path / 'outside.txt').write_text('bad\n')
        with self.assertRaises(ModelWorkspaceError):
            workspace.preserve_changes()
        self.assertEqual('draft\n', (self.repository / 'feature.txt').read_text())

    def test_user_change_during_invocation_is_preserved(self):
        workspace = self.workspace(scope=('feature.txt',))
        (workspace.path / 'feature.txt').write_text('model\n')
        (self.repository / 'feature.txt').write_text('user\n')
        with self.assertRaises(ModelWorkspaceError):
            workspace.preserve_changes()
        self.assertEqual('user\n', (self.repository / 'feature.txt').read_text())
        self.assertEqual('model\n', (workspace.path / 'feature.txt').read_text())

    def test_reserved_git_and_review_scopes_are_rejected(self):
        for scope in ('.git/', '.git/config', '.ai-agents-review/diff.patch'):
            with self.subTest(scope=scope), self.assertRaises(ModelWorkspaceError):
                self.workspace(scope=(scope,))

    def test_unknown_runtime_exit_preserves_scoped_model_edits(self):
        settings = replace(HermesSettings.load(Path(__file__).resolve().parents[2]), docker_required=True)
        runner = HermesRunner(settings, ArtifactStore(self.root / 'artifacts'))
        def interrupted(*args, **kwargs):
            (Path(args[3]) / 'feature.txt').write_text('partial\n')
            raise HermesExecutionError('timeout', category='timeout')
        with patch.object(runner, '_run', side_effect=interrupted), self.assertRaises(HermesExecutionError) as raised:
            runner.run('RUN-S09', 'stage-001', RoleId.DEVELOPMENT, self.repository, 'fixture',
                       allow_writes=True, write_scope=('feature.txt',))
        self.assertFalse(raised.exception.usage_reported)
        self.assertEqual('partial\n', (self.repository / 'feature.txt').read_text())

    def test_integrity_failure_keeps_successful_model_usage_and_snapshot(self):
        settings = replace(HermesSettings.load(Path(__file__).resolve().parents[2]), docker_required=True)
        runner = HermesRunner(settings, ArtifactStore(self.root / 'artifacts'))
        def raced(*args, **kwargs):
            (Path(args[3]) / 'feature.txt').write_text('model\n')
            (self.repository / 'feature.txt').write_text('user\n')
            return HermesResult('{}', TokenUsage(100, 20), .01)
        with patch.object(runner, '_run', side_effect=raced), self.assertRaises(HermesExecutionError) as raised:
            runner.run('RUN-S09', 'stage-001', RoleId.DEVELOPMENT, self.repository, 'fixture',
                       allow_writes=True, write_scope=('feature.txt',))
        self.assertEqual(120, raised.exception.usage.total_tokens)
        self.assertEqual('workspace_integrity', raised.exception.category)
        self.assertEqual('user\n', (self.repository / 'feature.txt').read_text())

    def test_runner_policy_restricts_cli_toolsets_even_with_broad_override(self):
        settings = replace(HermesSettings.load(Path(__file__).resolve().parents[2]), docker_required=True)
        runner = HermesRunner(settings, ArtifactStore(self.root / 'artifacts'))
        commands = []
        def start(command, **kwargs):
            commands.append((command, kwargs['env']))
            return FakeProcess(command, {'status': 'succeeded', 'text': '{}'},
                usage={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120})
        with patch.object(HermesSettings, 'validate_installation'), patch('app.services.hermes.runner.subprocess.Popen', side_effect=start):
            runner.run('RUN-S09', 'stage-001', RoleId.REVIEW, self.repository, 'fixture',
                       allow_writes=False, toolsets='coding,browser,code_execution')
        command, env = commands[0]
        self.assertEqual('terminal,file', command[command.index('--toolsets') + 1])
        self.assertIn('AI_AGENTS_WORKSPACE_POLICY', env)
        self.assertEqual(8, runner.pipeline_invocation_turns(RoleId.REVIEW))
        self.assertEqual(960, runner.pipeline_invocation_estimate(RoleId.REVIEW, 100, 20))


class ExecutionTests(unittest.TestCase):
    def test_pipeline_reserves_every_bounded_turn_and_passes_same_limit_to_runner(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            class BoundedRunner(FakeRoleRunner):
                def pipeline_invocation_turns(self, role_id):
                    return 8
                def pipeline_invocation_estimate(self, role_id, input_tokens, output_tokens):
                    return 8 * (input_tokens + output_tokens)
                def run(inner, *args, max_turns=None, **kwargs):
                    self.assertEqual(8, max_turns)
                    expected = 8 * (conservative_prompt_tokens(args[4]) + worker.coordinator.budget.policy.provider_input_overhead_tokens + kwargs['max_output_tokens'])
                    self.assertEqual(expected, store.reserved_token_total(args[0], stage_id=args[1]))
                    return super(BoundedRunner, inner).run(*args, **kwargs)
            runner = BoundedRunner(development_outputs=['fixed'], review_responses=[[]])
            store, worker, run_id = build_pipeline(root, source, runner)
            worker.run_once()
            self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])

    def test_review_patch_preserves_trailing_spaces_non_utf8_and_binary_bytes(self):
        with temporary_directory() as directory:
            source = create_repository(Path(directory))
            repository = GitRepository(source, HostGitSandbox())
            for content in (b'trailing space \n', b'non-utf8 \xff\n', b'\x00binary\xff\n'):
                base = git(source, 'rev-parse', 'HEAD')
                (source / 'feature.txt').write_bytes(content)
                git(source, 'add', 'feature.txt')
                git(source, 'commit', '-m', 'raw candidate')
                candidate = git(source, 'rev-parse', 'HEAD')
                bundle = repository.review_bundle(base, candidate)
                raw = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary', '--no-ext-diff', '--no-textconv', f'{base}..{candidate}'])
                self.assertEqual(raw, base64.b64decode(bundle['patch_base64']))

    def test_review_bundle_uses_exact_committed_range_and_rejects_dirty_candidate(self):
        with temporary_directory() as directory:
            source = create_repository(Path(directory))
            repository = GitRepository(source, HostGitSandbox())
            base = git(source, 'rev-parse', 'HEAD')
            (source / 'feature.txt').write_text('draft\n')
            git(source, 'add', 'feature.txt')
            git(source, 'commit', '-m', 'candidate')
            candidate = git(source, 'rev-parse', 'HEAD')
            bundle = repository.review_bundle(base, candidate)
            raw = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary', '--no-ext-diff', '--no-textconv', f'{base}..{candidate}'])
            self.assertEqual(raw, base64.b64decode(bundle['patch_base64']))
            (source / 'feature.txt').write_text('unreviewed\n')
            with self.assertRaises(GitRepositoryError):
                repository.review_bundle(base, candidate)

    def test_environment_failure_never_calls_finisher_or_consumes_code_retry(self):
        for kind in (VerificationFailureKind.TOOL_UNAVAILABLE, VerificationFailureKind.DEPENDENCY_UNAVAILABLE,
                     VerificationFailureKind.TIMEOUT):
            with self.subTest(kind=kind), temporary_directory() as directory:
                root = Path(directory)
                source = create_repository(root)
                runner = FakeRoleRunner(development_outputs=['fixed'], review_responses=[[]])
                store, worker, run_id = build_pipeline(root, source, runner)
                with patch.object(worker.coordinator.verifier, 'run_all', return_value=(
                    VerificationResult('python verify.py', 1, '', 'fixture environment error', .01, kind),)):
                    worker.run_once()
                self.assertEqual('NEEDS_ATTENTION', store.pipeline_job(run_id)['status'])
                self.assertNotIn(RoleId.IMPROVEMENT, [role for role, _ in runner.calls])
                self.assertEqual(kind, store.pipeline_workspace(run_id)['execution_evidence']['stage-001']['commands'][0]['failure_kind'])
                self.assertEqual(git(source, 'rev-parse', 'HEAD'), store.pipeline_workspace(run_id)['worktree']['source_snapshot']['head'])

    def test_assertion_failure_still_repairs_and_parent_explains_actual_execution(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = FakeRoleRunner(review_responses=[[]])
            store, worker, run_id = build_pipeline(root, source, runner)
            parent_id = next(item['data']['session_run_id'] for item in store.list_messages(run_id)
                             if item['kind'] == 'pipeline_parent')
            worker.run_once()
            self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])
            self.assertEqual(1, [role for role, _ in runner.calls].count(RoleId.IMPROVEMENT))
            message = next(item for item in store.list_messages(parent_id) if item['kind'] == 'pipeline_result')
            evidence = message['data']['execution_evidence']['stage-001']
            self.assertEqual(['feature.txt'], evidence['changed_files'])
            self.assertEqual('python verify.py', evidence['commands'][0]['command'])
            self.assertTrue(evidence['commands'][0]['passed'])
            self.assertEqual([], evidence['unperformed_commands'])

    def test_assertion_message_with_dependency_word_is_code_failure(self):
        self.assertEqual(VerificationFailureKind.TEST_FAILURE, VerificationRunner._failure_kind('', 'AssertionError: dependency state incorrect', 1))
        self.assertEqual(VerificationFailureKind.DEPENDENCY_UNAVAILABLE, VerificationRunner._failure_kind('', "ModuleNotFoundError: No module named 'missing_dependency'", 1))
        self.assertEqual(VerificationFailureKind.TOOL_UNAVAILABLE, VerificationRunner._failure_kind('', '', 127))


if __name__ == '__main__':
    unittest.main()
