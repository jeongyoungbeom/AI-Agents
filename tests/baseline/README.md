# S00 기준 재현 테스트

`test_s00_findings.py`와 `test_s00_boundaries.py`는 S00 당시 결함 재현 자료다. 수정 전 재현 테스트의 통과는 결함 확인이었다. S01에서는 경고가 섞인 문자열을 parser가 거절하는 계약을 유지하고, `tests/pipeline/test_hermes_contract.py`에서 Hermes stdout 경고와 모델 결과가 분리되는 정상 동작을 검증한다. 다른 결함은 해당 단계에서 수정한 뒤 대응 테스트를 회귀 기준으로 갱신한다.

`fixture.py`는 이후 단계가 재사용할 작은 committed Git 프로젝트를 만든다. `test_s00_fixture.py`는 그 프로젝트의 파일과 commit을 검사한다.

S02에서는 REV-011의 오류 안내와 장기 분석 실패 카드 기대값을 실패 표시 회귀로 전환했다. `tests/gateway/test_request_outcomes.py`가 결과 상태·시도/성공 수·취소·발신·migration을 검증한다. S03 이후의 미완료 결함 재현은 계속 결함 확인을 뜻한다.

저장소 루트에서 mock/로컬 fixture만 실행:

```powershell
$python = 'D:\AI-Agents\runtime\hermes-agent\venv\Scripts\python.exe'
& $python -m unittest discover -s tests/baseline -p 'test_s00_*.py' -v
```

테스트 도우미는 `test-tmp/`, `artifacts/`에 임시 자료를 만들며 운영 `data/agent-team.db`를 사용하지 않는다. 실제 모델·Docker·Telegram 호출의 성공을 이 명령으로 주장하지 않는다.

기존 모듈별 mock 검증은 `scripts/gateway-check.ps1`, `scripts/pipeline-check.ps1` 등에 있다. 실제 채널 경로는 `scripts/chat-gateway.ps1`과 `app/gateway/cli.py`이며 시작·중지·Telegram 송신은 이 기준 테스트에 포함하지 않는다.

S04에서는 REV-004·REV-005의 이름 기반 조건부 상담과 수정 금지·목적 질문 판정을 정상 동작 회귀로 전환했다. `tests/gateway/test_s04_intent.py`가 7개 고정 문장, 현재 원문 기반 상담 대상 제한, 문맥에 따른 모델 의도와 실행 승인 분리를 검증한다. 아직 남은 다른 결함 재현은 정상 동작 테스트로 해석하지 않는다.

S06에서는 REV-006의 12KB 큰 파일 제외 재현을 연속 행 조회 계획 유지 회귀로 전환했다. 49KB 이상 파일의 끝부분, 40배치 이상 계획, 누적 근거와 checkpoint 검증은 `tests/gateway/test_s06_repository.py`에 있다. 실제 Docker·모델·Telegram 수용 검증은 이 로컬 회귀와 별도로 수행한다.

S08에서는 REV-012의 완료 단계 파일 소실 재현을 동일 작업 공간·candidate 보존 회귀로 전환했다. `tests/pipeline/test_s08_resume.py`는 실제 Git/SQLite와 실제 하위 프로세스 종료를 사용해 질문·미커밋 변경·commit 전후 종료·최종 반영 후 종료·중복 재개·lease 경계를 검증한다. 모델과 Docker Git 경계는 대역이며 실제 provider/Docker/Telegram 성공을 뜻하지 않는다.
