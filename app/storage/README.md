# 상태 저장 구조

5-A부터 현재 상태는 SQLite에서 다음 영역으로 분리한다.

```text
conversation_sessions       채널·사용자·현재 역할·대화 모드·대화 로그·선택적 활성 작업
conversation_projects       대화에서 현재 선택한 프로젝트
task_definitions            작업 목표와 작업에 고정된 프로젝트
task_plan_state             현재 계획 버전과 해시
task_execution_approvals    현재 계획에 대한 실행 승인
scoped_repository_approvals 채널·대화·사용자 범위의 프로젝트 사용 승인
runs                        실행 단계·체크포인트·토큰·오류 상태
```

`RunState`는 기존 파이프라인과의 안정적인 호환을 위한 합성 읽기 모델이다. 저장할 때는 각 영역으로 나누고, 읽을 때 다시 조합한다. `runs.state_json`에는 실행 상태만 저장하며 체크포인트는 당시 전체 상태를 보존한다.

자유대화 세션은 로그 보존용 `session_run_id`만 가지며, 개발 의도가 확인될 때 별도의 `active_task_id`를 붙인다. 따라서 인사나 일반 질문만으로 개발 작업이 만들어지지 않는다.

기존 `conversation_bindings`와 통합 `runs.state_json` 데이터는 버전 마이그레이션으로 최초 한 번만 자동 이관한다. 이전 테이블은 복구 호환성을 위해 삭제하지 않지만 새 쓰기나 재이관에는 사용하지 않는다.

현재 프로젝트는 활성 작업과 별도이므로 `새 작업`을 시작해도 유지된다. 작업 목표, 계획과 실행 승인은 새 작업으로 복사되지 않는다.
