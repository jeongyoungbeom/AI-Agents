# Hermes 실행 어댑터

`config/agents.json`이 provider·모델·reasoning·프로필·timeout·필수 격리의 정본이다. 현재 provider는 `openai-codex`, 대화·계획·세 역할은 `gpt-6-sol`/`xhigh`다. `config/roles.json`은 표시 이름과 지시 파일만 관리한다.

대화는 `--no-tools`·1턴, 계획과 pipeline invocation은 최대 8턴을 사용한다. 프로필의 120/160턴 값이 pipeline 호출을 무제한 실행하지 않는다. 앱 전체 timeout은 현재 3,600초다. S09 실제 fixture의 240초 설정은 해당 검증 범위의 기록이며 운영 기본값이 아니다.

앱은 `--result-file`의 status/text/error/failure_reason/session_id와 별도 reported usage를 검증한다. stdout 경고를 모델 JSON으로 파싱하지 않는다. 인증·provider·timeout·형식·startup 오류를 분류하고 role log 참조를 남긴다. 입증된 startup만 예약을 해제하며 usage 없는 종료는 비용과 파일을 보존해 자동 재호출을 차단한다. `openai-codex`에는 현재 출력 상한을 전달하지 않으므로 토큰 예약을 provider 비용 hard cap으로 설명하지 않는다.

도구 실행은 private snapshot과 trusted workspace policy를 사용한다. 읽기 역할에는 정확한 base/candidate SHA·byte diff를 전달하고 snapshot은 readonly다. 쓰기는 승인 scope의 bind만 허용한다. pinned Docker image·network=none·호스트 환경/프로필 asset mount 금지·terminal,file 도구 경계를 유지한다. exact-file scope는 rename/delete를 지원하지 않으며 directory scope가 필요하다. 원본 반영은 gateway가 검증 candidate를 적용한다.

Windows의 별도 `runtime/hermes-agent` Git 저장소와 프로젝트용 로컬 수정이 필요하다. 구조화된 result-file, no-tools 대화, reported usage, watchdog, workspace policy와 CLI tool registry 계약을 지원해야 한다. 이 수정이 없는 upstream Hermes로 그대로 교체할 수 없다. 현재 런타임과 로컬 수정본의 설치·재현 패키지는 공개 저장소에 포함하지 않으므로, 소스 복제만으로 운영 환경이 준비되지는 않는다. 실행 환경과 검증 상태는 [루트 README](../../../README.md)를 따른다. 인증 파일 본문은 기록에 포함하지 않는다.
