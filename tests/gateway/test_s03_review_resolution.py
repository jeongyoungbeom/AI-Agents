import json
import sqlite3
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from app.contracts import RunPhase, RunState
from app.orchestrator import RunStateMachine
from app.services.context import ContextService
from app.services.repository import RepositoryReadContext
from app.storage import StateStore, StoreError
from tests.gateway.support import (
    AI_ROOT, TEST_REPOSITORY_HEAD, TEST_REPOSITORY_IDENTITY, build_application,
    temporary_directory,
)
from tests.gateway.test_s03_context import Reader, Team, Tools, incoming


class S03ReviewResolutionTests(unittest.TestCase):
    @staticmethod
    def cold_restore(database, run_id):
        script = '''
import json, sys
from pathlib import Path
from app.storage import StateStore
from app.services.context import ContextService
store = StateStore(Path(sys.argv[1]))
project = store.load_project_selection('telegram', '200')
bundle = ContextService(store).build(
    sys.argv[2], repository_identity=project.repository_identity,
    repository_head_sha=project.head_sha,
)
print(json.dumps(bundle.to_dict(), ensure_ascii=False))
'''
        result = subprocess.run(
            [sys.executable, '-c', script, str(database), run_id],
            cwd=AI_ROOT,
            capture_output=True, text=True, encoding='utf-8', timeout=15,
        )
        if result.returncode:
            raise AssertionError(result.stderr)
        return json.loads(result.stdout)

    def test_stale_followup_keeps_source_head_across_more_turns_and_restart(self):
        with temporary_directory() as directory:
            root = Path(directory)
            team, reader = Team(), Reader()
            store, application = build_application(
                root, object(), team_backend=team, repository_reader=reader,
                repository_tools=Tools(),
            )
            application.handle(incoming(1, 'D:\\projects\\sample'))
            application.handle(incoming(2, '이 프로젝트 사용 승인해'))
            application.handle(incoming(3, '센티널아 프로젝트 설명해'))
            store.refresh_project_head(
                'telegram', '200', '100', TEST_REPOSITORY_IDENTITY, 'c' * 40,
            )
            for number in (4, 5):
                application.handle(incoming(number, '첫 번째를 설명해'))
            reopened, restarted = build_application(
                root, object(), team_backend=team, repository_reader=reader,
                repository_tools=Tools(),
            )
            restarted.handle(incoming(6, '그럼 테스트할 때야?'))
            session = reopened.load_conversation_session('telegram', '200')
            answers = [item for item in reopened.list_messages(session.session_run_id)
                       if item['kind'] == 'assistant_message'
                       and '첫 번째 문제는 테스트 누락' in item['content']]
            self.assertEqual(4, len(answers))
            self.assertEqual(1, reader.inspections)
            for answer in answers:
                self.assertEqual(TEST_REPOSITORY_HEAD, answer['data']['head_sha'])
                self.assertTrue(answer['data']['untrusted_repository_data'])
            for context in team.contexts[1:]:
                history = [item for item in context.recent_messages
                           if item['kind'] == 'assistant_message'
                           and '첫 번째 문제는 테스트 누락' in item['content']]
                self.assertTrue(history)
                self.assertTrue(all(item['data']['stale_repository_snapshot']
                                    for item in history))
            cold = self.cold_restore(root / 'state.db', session.session_run_id)
            cold_answers = [item for item in cold['recent_messages']
                            if item['kind'] == 'assistant_message'
                            and '첫 번째 문제는 테스트 누락' in item['content']]
            self.assertEqual(4, len(cold_answers))
            self.assertTrue(all(item['data']['head_sha'] == TEST_REPOSITORY_HEAD
                                and item['data']['stale_repository_snapshot']
                                for item in cold_answers))

    def test_mixed_snapshot_answer_retains_both_heads(self):
        class AdvancingReader(Reader):
            head = TEST_REPOSITORY_HEAD

            def inspect(self, *_args, **_kwargs):
                self.inspections += 1
                return RepositoryReadContext(
                    TEST_REPOSITORY_IDENTITY, self.head, 'main', ('src/a.py',),
                    (('src/a.py', 'def first():\n    pass\n'),),
                )

        with temporary_directory() as directory:
            team, reader = Team(), AdvancingReader()
            store, application = build_application(
                Path(directory), object(), team_backend=team, repository_reader=reader,
                repository_tools=Tools(),
            )
            application.handle(incoming(1, 'D:\\projects\\sample'))
            application.handle(incoming(2, '이 프로젝트 사용 승인해'))
            application.handle(incoming(3, '센티널아 프로젝트 설명해'))
            reader.head = 'c' * 40
            application.handle(incoming(4, '프로젝트 설명을 새로 해'))
            application.handle(incoming(5, '앞 설명과 비교해'))
            mixed = [item for item in team.contexts[-1].recent_messages
                     if item['kind'] == 'assistant_message'
                     and item['data']['external_message_id'] == 'message-4'][0]
            self.assertEqual([TEST_REPOSITORY_HEAD, reader.head],
                             mixed['data']['repository_source_heads'])
            self.assertEqual('', mixed['data']['head_sha'])
            self.assertTrue(mixed['data']['stale_repository_snapshot'])
            self.assertEqual(2, reader.inspections)

    def _pipeline(self, root):
        team = Team()
        store, application = build_application(root, object(), team_backend=team)
        application.handle(incoming(1, 'D:\\projects\\sample'))
        application.handle(incoming(2, '이 프로젝트 사용 승인해'))
        session = store.load_conversation_session('telegram', '200')
        run_id = 'RUN-S03-REVIEW-PIPELINE'
        store.create_run(RunState(
            run_id, repository=str(root), repository_identity=TEST_REPOSITORY_IDENTITY,
            repository_head_sha=TEST_REPOSITORY_HEAD,
        ))
        store.set_conversation_task('telegram', '200', run_id)
        store.enqueue_pipeline_job(run_id, 'telegram', '200')
        store.claim_next_pipeline_job('worker')
        state = store.load_run(run_id)
        store.save_run(replace(state, phase=RunPhase.COMPLETED))
        return store, application, session, run_id, team

    def test_pipeline_result_survives_followup_finalize_order_and_restart(self):
        for followup_first in (True, False):
            with self.subTest(followup_first=followup_first), temporary_directory() as directory:
                root = Path(directory)
                store, application, session, run_id, team = self._pipeline(root)
                if followup_first:
                    application.handle(incoming(3, '무엇을 완료했어?'))
                    self.assertEqual('', store.load_conversation_session(
                        'telegram', '200').active_task_id)
                store.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
                if not followup_first:
                    application.handle(incoming(3, '무엇을 완료했어?'))
                with self.assertRaises(StoreError):
                    store.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
                reopened, restarted = build_application(root, object(), team_backend=team)
                restarted.handle(incoming(4, '그 작업 결과를 설명해'))
                results = [item for item in reopened.list_messages(session.session_run_id)
                           if item['kind'] == 'pipeline_result']
                self.assertEqual(1, len(results))
                self.assertEqual('COMPLETED', store.pipeline_job(run_id)['status'])
                self.assertTrue(any(item['kind'] == 'pipeline_result'
                                    for item in team.contexts[-1].recent_messages))
                cold = self.cold_restore(root / 'state.db', session.session_run_id)
                self.assertEqual(1, sum(item['kind'] == 'pipeline_result'
                                       for item in cold['recent_messages']))

    def test_pipeline_parent_is_fixed_when_active_task_or_session_changes(self):
        for rebind_session in (False, True):
            with self.subTest(rebind_session=rebind_session), temporary_directory() as directory:
                root = Path(directory)
                store, application, session, run_id, _team = self._pipeline(root)
                application.handle(incoming(3, '무엇을 완료했어?'))
                store.create_run(RunState('RUN-S03-NEXT'))
                store.set_conversation_task('telegram', '200', 'RUN-S03-NEXT')
                if rebind_session:
                    with sqlite3.connect(store.path) as connection:
                        connection.execute(
                            'UPDATE conversation_sessions SET session_run_id = ? '
                            'WHERE channel = ? AND conversation_id = ?',
                            ('RUN-S03-NEXT', 'telegram', '200'),
                        )
                store.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
                reopened = StateStore(root / 'state.db')
                results = [item for item in reopened.list_messages(session.session_run_id)
                           if item['kind'] == 'pipeline_result']
                self.assertEqual(1, len(results))
                self.assertFalse(any(item['kind'] == 'pipeline_result'
                                     for item in reopened.list_messages('RUN-S03-NEXT')))

    def test_pipeline_result_insert_failure_rolls_back_job_then_retry_links_once(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application, session, run_id, _team = self._pipeline(root)
            application.handle(incoming(3, '무엇을 완료했어?'))
            with sqlite3.connect(store.path) as connection:
                connection.execute('''CREATE TRIGGER fail_parent_result BEFORE INSERT ON messages
                    WHEN NEW.kind = 'pipeline_result' BEGIN
                    SELECT RAISE(ABORT, 'injected parent insert failure'); END''')
            with self.assertRaises(sqlite3.IntegrityError):
                store.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
            self.assertEqual('RUNNING', store.pipeline_job(run_id)['status'])
            self.assertFalse(any(item['kind'] == 'pipeline_result'
                                 for item in store.list_messages(session.session_run_id)))
            with sqlite3.connect(store.path) as connection:
                connection.execute('DROP TRIGGER fail_parent_result')
            store.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
            with self.assertRaises(StoreError):
                store.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
            self.assertEqual(1, sum(item['kind'] == 'pipeline_result'
                                    for item in store.list_messages(session.session_run_id)))

    def test_concurrent_pipeline_finalizers_only_record_one_parent_result(self):
        with temporary_directory() as directory:
            root = Path(directory)
            store, application, session, run_id, _team = self._pipeline(root)
            application.handle(incoming(3, '무엇을 완료했어?'))
            second = StateStore(root / 'state.db')
            barrier = threading.Barrier(2)

            def finish(target):
                barrier.wait(timeout=5)
                try:
                    target.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
                    return True
                except StoreError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(finish, target) for target in (store, second)]
                self.assertEqual([False, True], sorted(f.result() for f in futures))
            self.assertEqual(1, sum(item['kind'] == 'pipeline_result'
                                    for item in store.list_messages(session.session_run_id)))

    def test_legacy_queued_pipeline_without_parent_marker_links_after_detach(self):
        with temporary_directory() as directory:
            store, application, session, run_id, _team = self._pipeline(Path(directory))
            with sqlite3.connect(store.path) as connection:
                connection.execute("DELETE FROM messages WHERE kind = 'pipeline_parent'")
            application.handle(incoming(3, '무엇을 완료했어?'))
            store.finish_pipeline_job(run_id, 'worker', 'COMPLETED')
            self.assertEqual(1, sum(item['kind'] == 'pipeline_result'
                                    for item in store.list_messages(session.session_run_id)))

    def test_enqueue_parent_marker_failure_rolls_back_and_duplicate_enqueue_preserves_link(self):
        with temporary_directory() as directory:
            store, application = build_application(Path(directory), object(), team_backend=Team())
            application.handle(incoming(1, '안녕'))
            session = store.load_conversation_session('telegram', '200')
            run_id = 'RUN-S03-ENQUEUE-ATOMIC'
            store.create_run(RunState(run_id))
            store.set_conversation_task('telegram', '200', run_id)
            with sqlite3.connect(store.path) as connection:
                connection.execute('''CREATE TRIGGER fail_parent_marker BEFORE INSERT ON messages
                    WHEN NEW.kind = 'pipeline_parent' BEGIN
                    SELECT RAISE(ABORT, 'injected parent marker failure'); END''')
            with self.assertRaises(sqlite3.IntegrityError):
                store.enqueue_pipeline_job(run_id, 'telegram', '200')
            self.assertIsNone(store.pipeline_job(run_id))
            with sqlite3.connect(store.path) as connection:
                connection.execute('DROP TRIGGER fail_parent_marker')
            store.enqueue_pipeline_job(run_id, 'telegram', '200')
            store.clear_conversation_task('telegram', '200')
            store.enqueue_pipeline_job(run_id, 'telegram', '200')
            parents = [item for item in store.list_messages(run_id)
                       if item['kind'] == 'pipeline_parent']
            self.assertEqual(1, len(parents))
            self.assertEqual(session.session_run_id, parents[0]['data']['session_run_id'])

    def test_default_context_cap_keeps_both_source_locations_and_originals(self):
        for message_length in (1900, 3000):
            with self.subTest(message_length=message_length), temporary_directory() as directory:
                root = Path(directory)
                store = StateStore(root / 'state.db')
                RunStateMachine(store).create_run('RUN-S03-BUDGET')
                context = ContextService(store)
                for path in ('src/a.py', 'src/b.py'):
                    context.add_repository_evidence(
                        'RUN-S03-BUDGET', key=path, content='E' * 1600,
                        data={'repository_identity': TEST_REPOSITORY_IDENTITY,
                              'head_sha': TEST_REPOSITORY_HEAD, 'path': path,
                              'start_line': 1, 'end_line': 80, 'source_ref': path},
                    )
                for number in range(12):
                    context.add_message(
                        'RUN-S03-BUDGET', 'user', f'최근-{number} ' + 'M' * message_length,
                        data={'repository_identity': TEST_REPOSITORY_IDENTITY},
                    )
                bundle = context.build(
                    'RUN-S03-BUDGET', repository_identity=TEST_REPOSITORY_IDENTITY,
                    repository_head_sha=TEST_REPOSITORY_HEAD,
                )
                self.assertLessEqual(bundle.characters, 24000)
                self.assertTrue(bundle.truncated)
                self.assertEqual({'src/a.py', 'src/b.py'},
                                 {item['data']['path'] for item in bundle.evidence})
                self.assertIn('최근-11', bundle.recent_messages[-1]['content'])
                for item in bundle.evidence:
                    self.assertEqual(TEST_REPOSITORY_HEAD, item['data']['head_sha'])
                    self.assertEqual((1, 80), (item['data']['start_line'], item['data']['end_line']))
                    self.assertEqual(item['data']['path'], item['data']['source_ref'])
                    self.assertTrue(item['data']['untrusted_repository_data'])
                stored = [item for item in store.list_messages('RUN-S03-BUDGET')
                          if item['kind'] == 'repository_evidence']
                self.assertEqual([1600, 1600], [len(item['content']) for item in stored])
                restored = ContextService(StateStore(root / 'state.db')).build(
                    'RUN-S03-BUDGET', repository_identity=TEST_REPOSITORY_IDENTITY,
                    repository_head_sha=TEST_REPOSITORY_HEAD,
                )
                self.assertEqual(bundle.evidence, restored.evidence)
                other_project = context.build('RUN-S03-BUDGET', repository_identity='d' * 64)
                self.assertEqual((), other_project.evidence)


if __name__ == '__main__':
    unittest.main()
