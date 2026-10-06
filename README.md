# AI-Agents

**Telegram에서 대화하며 로컬 Git 프로젝트의 개발·리뷰·보완을 진행하는 AI 에이전트 시스템입니다.**

아이디어를 이야기하고, 프로젝트를 함께 살펴보고, 승인한 계획에 따라 코드를 수정합니다. 개발·리뷰·보완은 서로 다른 역할이 맡고, 작업 상태와 대화는 SQLite에 저장해 중지와 재개를 지원합니다.

> **개발 중인 프로젝트입니다.** 핵심 기능 구현과 회귀 테스트를 진행했으며, 실제 Telegram 환경의 전체 수용 검증은 아직 완료되지 않았습니다. 현재 실행 구성은 Windows의 `D:\AI-Agents`와 별도로 수정한 Hermes 런타임을 기준으로 합니다. 이 저장소만 복제해서 바로 실행할 수 있는 배포판은 아닙니다.

## 세 에이전트의 역할

| 에이전트 | 역할 | 하는 일 |
| --- | --- | --- |
| 빌더 | 설계·개발 | 요구사항을 구체화하고 코드를 구현합니다. |
| 센티널 | 독립 리뷰 | 설계와 구현을 검토하고 문제의 종류와 심각도를 판단합니다. |
| 피니셔 | 보완 | 리뷰에서 발견한 구현 문제를 수정합니다. |

자유 대화에서는 원하는 역할을 부르거나 `얘들아`, `셋 다`로 함께 이야기할 수 있습니다. 개발 작업은 리뷰 결과에 따라 빌더 재작업 또는 피니셔 보완으로 이어지며, 문제가 없으면 보완 단계를 생략합니다.

## 주요 기능

- **자연어 대화:** 별도의 작업 명령 형식 없이 질문과 개발 요청을 보냅니다.
- **프로젝트 조회:** 승인한 Git 저장소의 커밋된 코드에서 파일 검색, 상세 읽기, 장기 분석을 수행합니다.
- **계획과 실행 승인:** 프로젝트 사용과 코드 변경을 각각 명시적으로 승인합니다.
- **개발 파이프라인:** 단계별 개발 → 독립 리뷰 → 필요한 보완 → 검증을 실행합니다.
- **중지와 재개:** 대화, 질문, 중간 지시, 작업 공간과 결과를 저장해 다음 안전 지점에서 이어갑니다.
- **사용량 관리:** 역할별 토큰 사용량, 예산 경고와 재시도 기록을 확인합니다.
- **첨부파일 처리:** 허용된 UTF-8 텍스트 문서와 사진을 대화의 참고자료로 사용합니다.

현재 제공하는 실제 채널은 Telegram입니다. 다른 채널은 공통 `ChatAdapter` 인터페이스로 확장할 수 있습니다.

## 사용 흐름

```text
자유 대화 → 프로젝트 선택·승인 → 계획 검토·승인
         → 빌더 개발 → 센티널 리뷰 → 필요한 재작업·보완
         → 검증 → 검증된 결과를 로컬 저장소에 반영
```

Telegram에서 다음과 같이 시작할 수 있습니다.

```text
빌더야, 로그인 기능을 추가하고 싶어
C:\projects\sample
이 프로젝트 사용 승인해

계획에서 세션 만료 테스트도 포함해줘
개발 시작해
```

프로젝트를 읽는 승인 문장은 `이 프로젝트 사용 승인해`, 계획을 실행하는 승인 문장은 `개발 시작해`입니다. 계획이 바뀌면 다시 실행 승인을 받습니다.

| 메시지 | 기능 |
| --- | --- |
| `센티널아, 이 코드를 검토해줘` | 원하는 역할에게 요청 |
| `얘들아, 이 설계에 대해 의견 줘` | 세 역할의 의견 요청 |
| `상태` / `사용량` | 진행 상태와 토큰 사용량 확인 |
| `중지` / `재개` | 실행 중 작업의 안전한 일시 중지와 재개 |
| `새 작업` | 현재 프로젝트에서 별도의 작업 시작 |
| `도움말` | 지원하는 대화 기능 확인 |

## 실행 준비

### 필요한 환경

- Windows와 PowerShell, Git
- Docker Desktop의 Linux engine과 설정에 지정된 이미지
- 프로젝트용으로 수정한 Hermes 런타임과 해당 Python 가상환경
- Hermes의 `openai-codex` 인증 및 Telegram 봇 토큰

**Hermes는 별도로 준비해야 합니다.** 현재 어댑터는 구조화된 결과 파일, 실제 사용량 보고, 도구 없는 대화, 작업 공간 정책 등의 로컬 수정에 의존합니다. 일반 upstream Hermes로 그대로 대체할 수 없으며, 런타임·가상환경·인증 파일은 이 저장소에 포함하지 않습니다. 세부 계약은 [Hermes 어댑터 설명](app/services/hermes/README.md)을 참고하세요.

### 1. 소스와 런타임 준비

```powershell
git clone https://github.com/jeongyoungbeom/AI-Agents.git D:\AI-Agents
Set-Location D:\AI-Agents
```

스크립트의 기본 경로는 `D:\AI-Agents`입니다. 다른 위치를 사용하려면 `scripts/`의 경로 설정도 맞춰야 합니다. 다음 실행 파일이 있는지 확인하세요.

```text
runtime/hermes-agent/venv/Scripts/python.exe
runtime/hermes-agent/venv/Scripts/pythonw.exe
hermes-home/bin/hermes.exe
```

### 2. Telegram 설정

예시 파일을 복사합니다. 이미 설정한 파일은 덮어쓰지 않습니다.

```powershell
if (-not (Test-Path .\config\secrets.env)) {
    Copy-Item .\config\secrets.env.example .\config\secrets.env
}
```

`config/secrets.env`에 다음 값을 입력합니다.

```dotenv
TELEGRAM_BOT_TOKEN=<BotFather에서 발급받은 토큰>
TELEGRAM_ALLOWED_USERS=<허용할 숫자 사용자 ID>
TELEGRAM_ALLOWED_CHATS=
```

사용자 ID가 여러 개면 쉼표로 구분합니다. 기본 구성은 개인 채팅 전용이므로 `TELEGRAM_ALLOWED_CHATS`는 비워 둡니다. 예시의 `GIT_ACCESS_TOKEN`은 현재 로컬 개발 흐름에 필요하지 않습니다.

### 3. 인증과 실행

준비된 Hermes 실행 파일로 인증한 뒤 구성과 연결을 확인합니다.

```powershell
.\scripts\finish-auth.ps1
.\scripts\chat-gateway.ps1 check -Offline
.\scripts\chat-gateway.ps1 check
.\scripts\chat-gateway.ps1 run
```

`check -Offline`은 Telegram 네트워크 연결을 생략하지만 로컬 DB 초기화·마이그레이션과 로그 기록을 수행합니다. Docker 이미지와 런타임이 준비되지 않으면 점검과 도구 실행은 중단됩니다.

## 설정 파일

| 파일 | 설정 내용 |
| --- | --- |
| [agents.json](config/agents.json) | provider, 모델, 추론 강도, 역할 프로필, 실행 격리 |
| [roles.json](config/roles.json) | 에이전트 표시 이름과 역할별 지침 |
| [app.json](config/app.json) | 저장 경로와 승인 규칙 |
| [channels.json](config/channels.json) | Telegram, 워커, 첨부파일과 분석 범위 |
| [limits.json](config/limits.json) | 토큰 예산, 재시도와 보관 정책 |
| [toolchains.json](config/toolchains.json) | 검증에 사용할 Docker 이미지와 실행 도구 |

## 운영과 복구

```powershell
.\scripts\chat-gateway.ps1 start    # 백그라운드 실행
.\scripts\chat-gateway.ps1 status   # 실행 상태
.\scripts\chat-gateway.ps1 logs     # 최근 로그
.\scripts\chat-gateway.ps1 stop     # 워커와 하위 프로세스를 정리한 뒤 종료
.\scripts\chat-gateway.ps1 restart  # 안전 종료 후 다시 시작
.\scripts\chat-gateway.ps1 backup   # SQLite 백업
```

대화와 작업 상태는 `data/agent-team.db`, 작업별 기록은 `artifacts/`, 전역 로그는 `logs/`에 저장합니다. 백업·정리·복구는 [저장소 설명](app/storage/README.md)과 [보관 정책](config/limits.json)을 참고하세요.

토큰 예산은 호출 전 예약하고 보고된 사용량으로 정산합니다. 예약량은 provider 비용의 강제 상한이 아닙니다. 사용량을 확인할 수 없는 중단이나 초과가 발생하면 비용과 작업 결과를 보존하고 확인 필요 상태로 멈춥니다. 작업 소유자는 `/budget_reset`, `/budget_ack 이벤트ID 실제보고토큰`, `/resume`으로 확인된 작업을 이어갈 수 있습니다. 이 명령은 계정 사용 한도나 credit을 변경하지 않습니다. 상세 재개 조건은 [파이프라인 설명](app/pipeline/README.md)에 있습니다.

## 검증

준비된 로컬 환경에서 다음 검증 명령을 사용할 수 있습니다.

```powershell
.\scripts\foundation-check.ps1  # 설정·상태·예산 등 기반 검증
.\scripts\gateway-check.ps1     # 기반 + 대화 게이트웨이 검증
.\scripts\pipeline-check.ps1    # 기반 + 게이트웨이 + 개발 파이프라인 검증
.\scripts\f7-validation.ps1     # 로컬 fixture 기반 통합 검증
```

이 검증들은 실제 모델과 Telegram을 호출하지 않는 대역·로컬 fixture를 사용합니다. 실제 Docker 경계 검증은 준비된 이미지가 있을 때 `f7-validation.ps1 -Docker`로 실행합니다. 실제 Telegram·provider 수용 검증은 별도로 진행하며, 자동 테스트 통과만으로 완료를 판단하지 않습니다.

## 프로젝트 구조

```text
app/
├─ gateway/       대화 처리, Telegram 어댑터와 워커
├─ agents/        빌더·센티널·피니셔의 실행 지침과 응답 처리
├─ pipeline/      개발·리뷰·보완·검증 파이프라인
├─ orchestrator/  승인과 작업 상태 전환
├─ contracts/     계획과 역할 간 인계 형식
├─ services/      문맥, 예산, Git, Hermes, 격리와 검증
└─ storage/       SQLite 상태와 산출물 저장
config/           공개 설정과 비밀 설정 예시
scripts/          실행·운영·검증 명령
tests/            회귀 테스트와 로컬 fixture
coordinator/      이전 명령형 실행기
```

전체 구성은 [앱 구조](app/README.md), 대화 확장은 [게이트웨이](app/gateway/README.md), 코드 조회는 [저장소 조회 서비스](app/services/repository/README.md)에 설명되어 있습니다. [이전 실행기](coordinator/README.md)는 `-AllowLegacy`를 명시해야 실행됩니다.

## 코드 변경 경계

자유 대화는 승인된 저장소의 커밋된 코드를 제한적으로 읽습니다. 개발·보완은 승인한 범위의 임시 Git worktree에서 수행하고, 모든 단계의 리뷰와 검증을 통과한 결과만 원본 상태를 다시 확인한 뒤 반영합니다. Docker 도구 실행에는 네트워크와 호스트 자격증명을 전달하지 않습니다. 앱의 개발 흐름은 로컬 커밋과 반영까지 지원하며 GitHub push·PR·배포는 자동 실행하지 않습니다.

개인 개발용 에이전트 지침·스킬·세션 계획·리뷰 기록과 실행 데이터는 로컬에 보관합니다. 공개 저장소에는 프로그램 소스, 실행에 필요한 역할 지침, 설정 예시와 테스트를 포함합니다.
