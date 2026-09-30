# 프로젝트별 toolchain

`config/toolchains.json`은 Gateway가 소유하는 실행 profile 목록이다. profile은 digest로 고정된 image, 지원 stack, 필요한 실행 파일의 최소 버전, 그리고 cache 정책만 정의한다. 저장소는 Docker image, network, mount, argv를 직접 지정할 수 없다.

저장소 최상위의 `.ai-agents/toolchain.json`은 아래처럼 정의된 profile 이름 하나만 선택할 수 있다.

```json
{"schema_version": 1, "profile": "gradle-jdk21-controlled-cache"}
```

승인 전 preflight는 고정 HEAD tree에서 Gradle/Node/Python manifest와 wrapper를 읽고, 모든 검증 명령의 executable·version·Linux wrapper·dependency cache를 read-only 컨테이너에서 검사한다. `gradlew.bat`만 있는 프로젝트, 여러 stack이 감지됐는데 override가 없는 프로젝트, cache 없는 외부 dependency 프로젝트는 역할 호출과 큐 등록 전에 멈춘다.

기본 `gradle-jdk21` profile에는 dependency cache가 없다. Spring/Kotlin처럼 외부 dependency가 필요한 저장소는 운영자가 준비한 read-only named volume 또는 cache를 포함한 별도 digest-pinned image profile을 루트 catalog에 추가한 뒤, repository override로 그 profile을 선택해야 한다. 기존 `gradle-jdk21-controlled-cache`는 `ai-agents-gradle-cache-v1` volume의 `/home/gradle/.gradle/ai-agents-ready` marker를 확인하는 예시다. 실행 중에는 network가 계속 차단되며 cache와 profile은 저장소 코드가 변경할 수 없다.
