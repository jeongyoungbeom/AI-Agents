# 앱 구조

Telegram 대화, 에이전트 실행, 승인과 개발 파이프라인을 모듈별로 분리한 Python 애플리케이션입니다. 사용법과 현재 실행 제약은 [루트 README](../README.md)를 참고하세요.

```text
app/
├─ gateway/          공통 대화 처리와 채널별 어댑터
├─ orchestrator/     승인과 단계 상태 전환
├─ agents/           역할 지침, 프롬프트와 응답 처리
├─ contracts/        계획, 실행 결과와 역할 간 인계 형식
├─ services/
│  ├─ attachments/   Telegram 첨부파일 검사와 보관
│  ├─ budget/        토큰 예약·정산과 재시도 제한
│  ├─ context/       대화·결정·기억의 범위 관리
│  ├─ logging/       비밀정보 제거와 실행 기록
│  ├─ hermes/        모델 실행과 결과 계약 검증
│  ├─ git/           저장소 상태와 임시 worktree 관리
│  ├─ repository/    승인된 커밋의 검색·조회·장기 분석
│  ├─ sandbox/       Docker 실행 격리
│  ├─ security/      로컬 인증 파일 보호
│  ├─ retention/     기록 보관과 정리
│  ├─ toolchains/    검증 도구와 이미지 선택
│  └─ verification/  검증 명령의 정책 검사와 실행
├─ pipeline/         작업 큐, 역할 순서, 중지와 재개
└─ storage/          SQLite 상태와 산출물 저장
```

## 처리 흐름

1. 채널 어댑터가 외부 메시지를 공통 수신 형식으로 변환합니다.
2. 게이트웨이가 사용자 권한, 역할, 대화 상태와 프로젝트 승인을 확인합니다.
3. 대화·분석·개발 요청을 각각 영속 큐에 넣어 워커가 처리합니다.
4. 개발은 승인된 계획에 따라 빌더 → 센티널 → 필요한 보완 → 검증 순서로 진행합니다.
5. 결과와 사용량을 저장하고 공통 발신함을 통해 채널에 전달합니다.

대화와 작업 상태는 SQLite에 보존합니다. 개발 작업은 임시 worktree에서 실행하고, 검증한 candidate만 원본 상태를 확인한 뒤 반영합니다. 모델 결과나 사용량을 확인할 수 없으면 자동 재실행을 막고 확인 필요 상태로 남깁니다.

## 주요 진입점

| 구성요소 | 진입점 | 설명 |
| --- | --- | --- |
| 실행 명령 | `gateway/cli.py` | 실행, 점검, 백업과 복구 |
| 구성 조립 | `gateway/bootstrap.py` | 설정과 서비스 연결 |
| 대화 | `gateway/core/conversation.py` | 대화·프로젝트·승인 모드 |
| 개발 | `pipeline/coordinator.py` | 단계 실행과 역할 분기 |
| 저장 | `storage/sqlite_store.py` | 상태, 큐와 체크포인트 |

현재 실제 어댑터는 Telegram이며 다른 채널은 `ChatAdapter`를 구현해 연결할 수 있습니다. 자세한 계약은 [게이트웨이](gateway/README.md), [파이프라인](pipeline/README.md), [Hermes 어댑터](services/hermes/README.md), [저장소](storage/README.md) 문서를 참고하세요.
