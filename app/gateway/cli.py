from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import FoundationConfig
from app.gateway.bootstrap import build_telegram_runtime
from app.gateway.runtime_guard import GatewayInstanceLock
from app.services.hermes import HermesSettings
from app.services.retention import RetentionManager, RetentionPolicy
from app.services.security import CredentialFileProtector
from app.storage import StateStore


AI_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="공통 대화 게이트웨이")
    parser.add_argument(
        "command",
        choices=("run", "check", "secure", "backup", "purge", "restore"),
        nargs="?",
        default="run",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Telegram 네트워크 요청 없이 설정만 확인"
    )
    parser.add_argument("--backup", help="data/backups 아래에서 복구할 SQLite 백업 파일")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "secure":
            hermes = HermesSettings.load(AI_ROOT)
            protected = CredentialFileProtector(AI_ROOT, hermes.home).protect()
            print("인증 파일 ACL 적용: " + ", ".join(str(path) for path in protected))
            return 0
        if arguments.command in {"backup", "purge", "restore"}:
            foundation = FoundationConfig.load(AI_ROOT)
            retention = RetentionManager(
                AI_ROOT,
                StateStore(foundation.database),
                RetentionPolicy.load(AI_ROOT / "config" / "limits.json"),
            )
            if arguments.command == "backup":
                print(retention.backup_database())
                return 0
            if arguments.command == "purge":
                with GatewayInstanceLock(AI_ROOT / "data" / "gateway.lock"):
                    backup = retention.backup_database()
                    result = retention.purge()
                print(json.dumps({"backup": str(backup), **result.to_dict()}, ensure_ascii=False))
                return 0
            if not arguments.backup:
                raise ValueError("restore 명령에는 --backup이 필요합니다.")
            with GatewayInstanceLock(AI_ROOT / "data" / "gateway.lock"):
                print(retention.restore_database(Path(arguments.backup)))
            return 0
        runtime = build_telegram_runtime(AI_ROOT)
        if arguments.command == "check":
            runtime.hermes_settings.validate_installation()
            print(runtime.adapter.check(online=not arguments.offline))
            print("Hermes 및 ChatGPT OAuth 상태: 준비됨")
            return 0
        if arguments.offline:
            raise ValueError("run 명령에는 --offline을 사용할 수 없습니다.")
        with GatewayInstanceLock(AI_ROOT / "data" / "gateway.lock"):
            runtime.sandbox.cleanup_stale()
            stop_request = AI_ROOT / "data" / "gateway.stop"
            stop_request.unlink(missing_ok=True)
            print(runtime.adapter.check(online=True), flush=True)
            runtime.client.delete_webhook()
            print("대화 게이트웨이를 시작했습니다. 종료하려면 Ctrl+C를 누르세요.", flush=True)
            runtime.outbound_worker.start()
            runtime.activity_worker.start()
            runtime.conversation_worker.start()
            runtime.repository_analysis_worker.start()
            runtime.pipeline_worker.start()
            try:
                runtime.runner.run_forever(stop_requested=stop_request.exists)
            finally:
                # A conversation worker may currently be blocked inside a long
                # Hermes call. Stop the process tree first; waiting for the
                # worker before doing so can make `stop` hang for the full
                # model timeout.
                runtime.hermes_runner.request_stop()
                runtime.pipeline_worker.stop()
                runtime.conversation_worker.stop()
                runtime.repository_analysis_worker.stop()
                runtime.pipeline_worker.join(timeout=45)
                runtime.conversation_worker.join(timeout=45)
                runtime.repository_analysis_worker.join(timeout=45)
                runtime.outbound_worker.stop()
                runtime.outbound_worker.join(timeout=15)
                runtime.activity_worker.stop()
                runtime.activity_worker.join(timeout=15)
                still_running = []
                if runtime.pipeline_worker.is_alive():
                    still_running.append("pipeline")
                if runtime.conversation_worker.is_alive():
                    still_running.append("conversation")
                if runtime.repository_analysis_worker.is_alive():
                    still_running.append("repository-analysis")
                if runtime.outbound_worker.is_alive():
                    still_running.append("outbound")
                if runtime.activity_worker.is_alive():
                    still_running.append("activity")
                if still_running:
                    raise RuntimeError(
                        "안전 종료 시간 안에 워커가 멈추지 않았습니다: "
                        + ", ".join(still_running)
                    )
                if runtime.hermes_runner.active_process_count():
                    raise RuntimeError("종료 후에도 Hermes 프로세스가 남아 있습니다.")
                stop_request.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("대화 게이트웨이를 종료했습니다.")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
