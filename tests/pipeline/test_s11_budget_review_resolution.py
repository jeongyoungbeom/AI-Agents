from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import traceback
import unittest
from dataclasses import replace
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.contracts import RoleId, RunPhase, StageContract, TokenUsage
from app.services.hermes import HermesResult
from tests.pipeline import test_s11_budget_resume as budget_support
from tests.pipeline.support import build_pipeline, create_repository, git, temporary_directory
from tests.pipeline.test_s08_resume import reopen_worker
from tests.pipeline.test_s10_steering import SteeringBackend, gateway, incoming


class CapturingOverrunRunner(budget_support.OverrunRunner):
    def __init__(self, *, question=False):
        super().__init__()
        self.question = question
        self.prompts = []
        self.paths = []

    def run(self, *args, **kwargs):
        self.prompts.append(args[4])
        self.paths.append(str(args[3]))
        if self.question and not self.calls:
            self.calls.append((args[2], kwargs['allow_writes']))
            return HermesResult(json.dumps({'summary': '방향 확인 필요',
                'needs_user_input': ['이 구현 방향으로 계속할까요?']}, ensure_ascii=False),
                TokenUsage(total_tokens=100000), 0.01)
        return super().run(*args, **kwargs)


class AnswerThenCompletionOverrunRunner(CapturingOverrunRunner):
    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        tokens = 100000 if len(self.calls) == 2 else 120
        return HermesResult(result.text, TokenUsage(total_tokens=tokens), result.elapsed_seconds)


def lock_probe(control, operation, output):
    helper = budget_support.S11BudgetResumeTests()
    # 영구 운영 자원 없이 합성 child에서만 교착/timeout을 관측한다.
    with temporary_directory() as directory:
        root = Path(directory)
        store, worker, run_id, source, runner, manager, actor, event = helper.paused(root)
        app = helper.application(root, worker, manager)
        if control == 'reset':
            manager.acknowledge_overrun(run_id, event['event_id'], 100000, **actor)
        manager.policy = replace(manager.policy, retries={'technical_error': 1})
        if operation in {'settle', 'release', 'recover'}:
            store.create_token_reservation('peer', run_id, 'chat', 'development', 'conversation', 10)
            from app.services.budget import BudgetReservation
            reservation = BudgetReservation('peer', 10)
            if operation == 'recover':
                store.create_model_call('peer-call', 'peer-call', run_id, 'chat',
                    'development', 'conversation', 'peer', 'peer-request')
        gate, entering_store = threading.Event(), threading.Event()
        original_lock = store._lock

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == 'budget-peer':
                    entering_store.set()
                original_lock.acquire()
                return self

            def __exit__(self, *args):
                original_lock.release()

        store._lock = ObservedLock()
        method = 'reset_budget' if control == 'reset' else 'acknowledge_overrun'
        original_control = getattr(manager, method)

        def observed_control(*args, **kwargs):
            gate.set()  # 실제 Gateway의 바깥 store transaction 안이다.
            if not entering_store.wait(5):
                raise RuntimeError('peer did not reach store lock')
            return original_control(*args, **kwargs)

        setattr(manager, method, observed_control)
        outcomes = {}

        def control_thread():
            try:
                text = '/budget_reset' if control == 'reset' else f"/budget_ack {event['event_id']} 100000"
                reply = app.handle(helper.message(1, text))
                outcomes['control'] = reply[0].metadata['request_result']
            except Exception:
                outcomes['control_error'] = traceback.format_exc()

        def peer_thread():
            try:
                if operation == 'reserve':
                    outcomes['reservation'] = manager.reserve(run_id, 'chat', 'development', 'conversation', 1).reservation_id
                elif operation == 'retry':
                    outcomes['retry_count'] = manager.record_retry(run_id, 'chat', 'technical_error', 'synthetic')
                elif operation == 'release':
                    manager.release_reservation(reservation)
                elif operation == 'settle':
                    manager.record_usage(run_id, 'chat', 'development', 'conversation', TokenUsage(total_tokens=7), reservation=reservation)
                elif operation == 'recover':
                    manager.recover_interrupted_calls(run_id, 'chat', 'peer-request')
                elif operation == 'transaction':
                    with manager.transaction():
                        manager.record_usage(run_id, 'chat', 'development', 'conversation', TokenUsage(total_tokens=7))
                outcomes['peer'] = 'returned'
            except Exception as exc:
                from app.services.budget import BudgetExceeded
                if isinstance(exc, BudgetExceeded):
                    outcomes['peer'] = 'budget-denied'
                else:
                    outcomes['peer_error'] = traceback.format_exc()

        a = threading.Thread(target=control_thread, name='gateway-control', daemon=True)
        b = threading.Thread(target=peer_thread, name='budget-peer', daemon=True)
        a.start()
        ready = gate.wait(5)
        if ready:
            b.start()
            a.join(3)
            b.join(3)
        frames = sys._current_frames()
        record = dict(control=control, operation=operation, ready=ready,
            alive=[t.name for t in (a,b) if t.is_alive()], outcomes=outcomes,
            stacks={t.name: ''.join(traceback.format_stack(frames[t.ident])) for t in (a,b) if t.ident in frames})
        passed = ready and not record['alive'] and not any(k.endswith('_error') for k in outcomes)
        if passed:
            record.update(usage=store.usage_total(run_id), reserved=store.reserved_token_total(run_id),
                control_events=store.budget_control_events(run_id),
                inbound=store.inbound_receipt('telegram', 'chat-1:1'))
            expected = 100007 if operation in {'settle','transaction'} else 100010 if operation == 'recover' else 100000
            passed = record['usage'] == expected
            if operation in {'settle','release','recover'}:
                passed = passed and record['reserved'] == 0 and outcomes['control']['outcome'] == 'failed'
            if operation == 'recover':
                passed = passed and store.has_budget_anomaly(run_id)
            passed = passed and record['inbound']['status'] == 'COMPLETED'
        record['passed'] = passed
        Path(output).write_text(json.dumps(record, ensure_ascii=False, indent=2), 'utf-8')
        if not passed:
            # 죽은 합성 thread의 잠금/DB를 tempfile 정리에서 기다리지 않는다.
            os._exit(1)


class S11BudgetReviewResolutionTests(unittest.TestCase):
    def lock_cases(self, operations):
        output_root = Path(os.environ.get('S11_RESOLUTION_EVIDENCE', Path(__file__).resolve().parents[2] / 'test-tmp'))
        output_root.mkdir(parents=True, exist_ok=True)
        for control in ('reset', 'ack'):
            for operation in operations:
                with self.subTest(control=control, operation=operation):
                    output = output_root / f'lock-{control}-{operation}.json'
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--lock-probe', control, operation, str(output)], capture_output=True, text=True, timeout=35)
                    self.assertEqual(0, result.returncode, result.stdout + result.stderr +
                        (output.read_text('utf-8') if output.exists() else 'no observation'))
                    self.assertTrue(json.loads(output.read_text('utf-8'))['passed'])

    def test_gateway_controls_overlap_reservation_and_retry_without_deadlock(self):
        self.lock_cases(('reserve', 'retry'))

    def test_gateway_controls_overlap_settlement_and_release_without_deadlock(self):
        self.lock_cases(('settle', 'release'))

    def test_gateway_controls_overlap_outer_budget_transaction_and_recovery(self):
        self.lock_cases(('transaction', 'recover'))

    def answer_case(self, restart):
        helper = budget_support.S11BudgetResumeTests()
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = CapturingOverrunRunner(question=True)
            store, worker, run_id = build_pipeline(root, source, runner)
            self.assertTrue(worker.run_once())
            manager = worker.coordinator.budget
            app = helper.application(root, worker, manager)
            first = store.open_execution_question_for_run(run_id)
            checkpoint = store.pipeline_workspace(run_id)
            original = store.model_call(checkpoint['invocation']['call_id'])
            original_source = git(source, 'rev-parse', 'HEAD')
            plan_hash = store.load_run(run_id).approved_plan_hash
            event = next(e for e in store.budget_control_events(run_id) if e['event_type'] == 'BUDGET_RESERVATION_EXCEEDED')
            self.assertEqual('success', app.handle(helper.message(1, '/budget_reset'))[0].metadata['request_result']['outcome'])
            self.assertEqual('success', app.handle(helper.message(2, f"/budget_ack {event['event_id']} 100000"))[0].metadata['request_result']['outcome'])
            self.assertIsNone(worker.coordinator._acknowledged_development(store.load_run(run_id),
                StageContract.from_dict(store.load_plan_revision(run_id,
                    store.load_run(run_id).plan_revision)['plan']['stages'][0]), checkpoint))
            retained_events = store.budget_control_events(run_id)
            if restart:
                store, worker = reopen_worker(root, runner)
                app = helper.application(root, worker, worker.coordinator.budget)
            reply = app.handle(helper.message(3, '네, 이 방향으로 계속해 주세요.'))
            self.assertIn('QUEUED', reply[0].text)
            self.assertTrue(worker.run_once())
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase, store.pipeline_job(run_id)['last_error'])
            self.assertIsNone(store.open_execution_question_for_run(run_id))
            self.assertEqual([first['question_id']], [q['question_id'] for q in store.answered_execution_questions(run_id, 'stage-001')])
            self.assertEqual([(RoleId.DEVELOPMENT, True), (RoleId.DEVELOPMENT, True), (RoleId.REVIEW, False)], runner.calls)
            self.assertIn('네, 이 방향으로 계속해 주세요.', runner.prompts[1])
            self.assertEqual(1, len(set(runner.paths)))
            self.assertEqual(checkpoint['worktree'], store.pipeline_workspace(run_id)['worktree'])
            self.assertEqual(plan_hash, store.load_run(run_id).approved_plan_hash)
            self.assertEqual(original, store.model_call(original['call_id']))
            self.assertEqual(retained_events, store.budget_control_events(run_id))
            self.assertEqual(100240, store.usage_total(run_id))
            self.assertEqual(240, store.budget_usage_total(run_id))
            self.assertNotEqual(original_source, git(source, 'rev-parse', 'HEAD'))
            self.assertEqual(store.pipeline_workspace(run_id)['validated_sha'], git(source, 'rev-parse', 'HEAD'))
            self.assertFalse(any(e['event_type'] == 'DEVELOPMENT_RESULT_REUSED' for e in store.list_events(run_id)))

    def test_answered_overrun_question_executes_builder_with_answer(self):
        self.answer_case(False)

    def test_answered_overrun_question_after_restart_preserves_worktree_and_usage(self):
        self.answer_case(True)

    def test_new_supplement_invalidates_acknowledged_completed_builder_input(self):
        helper = budget_support.S11BudgetResumeTests()
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = CapturingOverrunRunner()
            store, worker, run_id = build_pipeline(root, source, runner)
            worker.run_once()
            manager = worker.coordinator.budget
            app = helper.application(root, worker, manager)
            original = store.model_call(store.pipeline_workspace(run_id)['invocation']['call_id'])
            event = next(e for e in store.budget_control_events(run_id) if e['event_type'] == 'BUDGET_RESERVATION_EXCEEDED')
            app.handle(helper.message(1, '/budget_reset'))
            app.handle(helper.message(2, f"/budget_ack {event['event_id']} 100000"))
            steering, _ = gateway(root, store, worker, SteeringBackend('supplement', ['feature.txt']))
            steering.handle(incoming(3, '빈 입력도 처리해 줘'))
            app.handle(helper.message(4, '/resume'))
            self.assertTrue(worker.run_once())
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase, store.pipeline_job(run_id)['last_error'])
            self.assertEqual(3, len(runner.calls))
            self.assertIn('빈 입력도 처리해 줘', runner.prompts[1])
            self.assertEqual(original, store.model_call(original['call_id']))
            self.assertEqual(100240, store.usage_total(run_id))
            self.assertFalse(any(e['event_type'] == 'DEVELOPMENT_RESULT_REUSED' for e in store.list_events(run_id)))

    def test_completed_builder_with_unchanged_prior_answer_is_reused_without_new_call(self):
        helper = budget_support.S11BudgetResumeTests()
        with temporary_directory() as directory:
            root = Path(directory)
            source = create_repository(root)
            runner = AnswerThenCompletionOverrunRunner(question=True)
            store, worker, run_id = build_pipeline(root, source, runner)
            worker.run_once()
            app = helper.application(root, worker, worker.coordinator.budget)
            self.assertEqual(120, store.usage_total(run_id))
            app.handle(helper.message(1, '네, 이 방향으로 계속해 주세요.'))
            self.assertTrue(worker.run_once())
            self.assertEqual(RunPhase.PAUSED, store.load_run(run_id).phase)
            checkpoint = store.pipeline_workspace(run_id)
            original = store.model_call(checkpoint['invocation']['call_id'])
            answers = store.answered_execution_questions(run_id, 'stage-001')
            self.assertEqual([q['question_id'] for q in answers],
                json.loads(original['result_json'])['answered_question_ids'])
            event = next(e for e in store.budget_control_events(run_id) if e['event_type'] == 'BUDGET_RESERVATION_EXCEEDED')
            app.handle(helper.message(2, '/budget_reset'))
            app.handle(helper.message(3, f"/budget_ack {event['event_id']} 100000"))
            app.handle(helper.message(4, '/resume'))
            self.assertTrue(worker.run_once())
            self.assertEqual(RunPhase.COMPLETED, store.load_run(run_id).phase, store.pipeline_job(run_id)['last_error'])
            self.assertEqual([(RoleId.DEVELOPMENT, True), (RoleId.DEVELOPMENT, True), (RoleId.REVIEW, False)], runner.calls)
            self.assertEqual(original, store.model_call(original['call_id']))
            self.assertEqual(checkpoint['candidate_sha'], git(source, 'rev-parse', 'HEAD'))
            self.assertEqual(100240, store.usage_total(run_id))
            self.assertEqual(120, store.budget_usage_total(run_id))
            self.assertEqual(1, sum(e['event_type'] == 'DEVELOPMENT_RESULT_REUSED' for e in store.list_events(run_id)))


if __name__ == '__main__':
    if '--lock-probe' in sys.argv:
        lock_probe(*sys.argv[2:])
    else:
        unittest.main()
