# 단계 파이프라인

승인된 계획의 각 단계를 아래 분기로 처리한다.

```text
빌더 개발 → 센티널 독립 리뷰
  ├─ 문제 없음 → 피니셔 생략 → 결정론적 검증
  ├─ 설계 문제 → 빌더 재작업(최대 1회) → 센티널 재리뷰
  └─ 구현 문제 → 피니셔 코드 수정 → 결정론적 검증
       └─ critical/high → 검증된 최종 SHA를 센티널이 마감 재리뷰(1회)
```

- 모든 단계에서 위 분기를 반복한다.
- 재작업이나 마감 재리뷰 후 문제가 남으면 자동 반복하지 않고 멈춘다.
- 진행 메시지는 공통 발신함에 저장되어 Telegram으로 전달된다.
- 토큰과 재시도 횟수는 `config/limits.json`을 따른다.
- assertion·구현 검증 실패는 명확한 근거로 피니셔 보완을 한 번만 추가 실행한다. 도구/의존성/컨테이너 startup·timeout 환경 실패는 보완 횟수를 쓰지 않고 환경 확인으로 중단한다.
- 같은 저장소의 동시 쓰기는 SQLite lease/heartbeat와 owned root의 OS process lock으로 막는다.
- owned worktree·stage base/current/validated SHA·질문·중간 지시를 영속 연결한다. 재시작은 같은 공간과 candidate를 대조한다. 미확인 모델 결과/비용은 예약·파일을 보존하고 `NEEDS_ATTENTION`에서 자동 재호출을 차단한다.
- 실행 중 지시는 다음 안전 지점에서 적용한다. 테스트 우선과 기존 완료 조건이 충돌하면 구체적인 질문으로 확인하며 새 scope 승인을 자동으로 만들지 않는다.
- 개발·보완·리뷰·검증은 run별 임시 Git worktree에서 진행한다. 마지막 단계의 모든
  리뷰와 검증이 통과한 candidate commit만 원본 snapshot 재검증 뒤 적용하며, 원본이
  dirty이거나 HEAD/branch/status가 바뀌면 `worktree-recovery.json`을 남기고 멈춘다.

확인된 예약 초과는 현재 중단된 개발 run 소유자가 `/budget_ack 이벤트ID 실제보고토큰`으로 명시적으로 확인할 수 있다. `/budget_reset`은 원 usage/result/event를 보존하는 새 예산 기준점이며 초과 확인과 별개다. 둘 다 자동 실행을 시작하지 않고 `/resume`으로 같은 candidate를 이어간다. 확인한 완료 Builder 호출과 실제 보고량·승인 plan/stage/candidate가 일치하면 Builder 실행/정산을 반복하지 않고 보존 결과로 리뷰를 이어간다. 새로운 redirect/test_first는 이 재사용 대상이 아니며 unknown·활성 호출/예약은 계속 차단한다. 기존 권한·source/worktree·scope·검증 경계를 통과해야 반영한다.
