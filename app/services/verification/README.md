# 검증 서비스

테스트·린트·타입 검사 명령을 셸 없이 인자 배열로 실행한다.
셸 연결, 리다이렉션, 삭제, Push, Merge 명령은 허용하지 않는다. 시간 제한과 사용자 중지를 지원하며, 검증이 추적 파일이나 Git 위치를 바꾸면 즉시 멈춘다.

실행 승인이 저장되기 전 `toolchains` 서비스가 선택한 digest-pinned profile을 확인한다. 검증 실행은 이 profile과 읽기 전용 cache mount를 그대로 재사용한다. 실패 결과에는 `tool_unavailable`, `dependency_unavailable`, `test_failure`, `timeout` 중 하나를 남겨 도구·cache 문제를 테스트 실패와 구분한다.
