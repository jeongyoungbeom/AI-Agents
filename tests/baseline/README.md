# S00 기준 재현 테스트

`test_s00_findings.py`와 `test_s00_boundaries.py`의 9개 테스트는 **현재 결함의 재현**이다. 통과는 나쁜 동작이 그대로 관찰됐다는 뜻이다. 해당 단계에서 수정할 때 기대값을 정상 동작으로 뒤집은 회귀 테스트를 만들고, 대응하는 S00 재현 테스트는 제거하거나 갱신한다.

`fixture.py`는 이후 단계가 재사용할 작은 committed Git 프로젝트를 만든다. `test_s00_fixture.py`는 그 프로젝트의 파일과 commit을 검사한다.

저장소 루트에서 mock/로컬 fixture만 실행:

```powershell
$python = 'D:\AI-Agents\runtime\hermes-agent\venv\Scripts\python.exe'
& $python -m unittest discover -s tests/baseline -p 'test_s00_*.py' -v
```

테스트 도우미는 `test-tmp/`, `artifacts/`에 임시 자료를 만들며 운영 `data/agent-team.db`를 사용하지 않는다. 실제 모델·Docker·Telegram 호출의 성공을 이 명령으로 주장하지 않는다. 현재 운영 gateway가 실행 중이므로 실제 호출 검증은 해당 단계의 승인 범위와 실행 절차를 먼저 확인한다.

기존 모듈별 mock 검증은 `scripts/gateway-check.ps1`, `scripts/pipeline-check.ps1` 등에 있다. 실제 채널 경로는 `scripts/chat-gateway.ps1`과 `app/gateway/cli.py`이며 시작·중지·Telegram 송신은 이 기준 테스트에 포함하지 않는다.
