# Git 서비스

프로젝트 위치는 고정하지 않고 작업마다 승인된 절대경로를 전달받는다. F-4부터
Builder와 Finisher가 받는 경로는 원본 checkout이 아니라 run별 gateway-owned 임시
Git worktree다.

- 시작 전 원본 Git 최상위 경로, 브랜치, HEAD, tracked/untracked 상태, 커밋 작성자 설정 기록
- OS temp 아래의 run별 detached worktree 생성; 고정 `workspaces` 폴더는 사용하지 않음
- host lifecycle은 `git worktree add --no-checkout`으로 metadata만 만들며, repository filter가
  동작할 수 있는 실제 checkout은 network-off Docker Git 경계에서만 수행
- linked worktree의 host Git metadata는 gateway가 만든 컨테이너 전용 pointer로만 고정 Git
  서비스에 연결하며, repository-derived argv나 모델 입력으로 mount를 확장하지 않음
- 역할 실행 전후 브랜치와 HEAD 변경 감지
- 저장소 밖을 가리키는 symlink를 거부하고, 리뷰 단계의 작업 트리 변경도 감지
- 개발·보완 결과는 승인된 단계 `scope`의 저장소 상대 파일만 파일 단위로 stage하여 오케스트레이터가 로컬 커밋 (`git add -A` 사용 안 함)
- 모든 단계의 리뷰·검증이 끝난 뒤에도 원본 snapshot(HEAD, branch, status)이 동일하고
  candidate range가 승인 scope 안에 있을 때만 고정 Git argv로 cherry-pick 적용
- `push`, 원격 작업, 모델 주도 branch 전환은 없음

기존 사용자 변경, 원본 HEAD 변경, scope 위반, 취소 또는 읽기 전용 역할의 변경이 있으면
원본은 자동 복구·반영하지 않고 멈춘다. 변경이 남은 임시 worktree는
`worktree-recovery.json`과 binary patch를 artifacts에 보관하고 사용자 확인 전 삭제하지 않는다.
