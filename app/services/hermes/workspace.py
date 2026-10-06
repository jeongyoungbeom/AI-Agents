from __future__ import annotations

import hashlib
import base64
import json
import os
import shutil
import stat
import uuid
from pathlib import Path

from app.contracts import normalize_stage_scope


class ModelWorkspaceError(RuntimeError):
    pass


class ModelWorkspace:
    """Model tools see a disposable file snapshot, never the source Git directory.

    Docker mounts this snapshot read-only and overlays only approved write paths.
    Changes are copied back at the invocation boundary, including uncertain exits,
    so the coordinator's existing checkpoint/recovery trail keeps model edits.
    """

    def __init__(self, repository: Path, destination: Path, *, scope=(), review=None,
                 image: str, cache_mounts=()):
        self.repository = repository.resolve(strict=True)
        self.root = destination / uuid.uuid4().hex
        self.path = self.root / 'workspace'
        self.path.mkdir(parents=True)
        self.before = self._files(self.repository)
        for relative in self.before:
            source = self.repository / relative
            target = self.path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self.scope = tuple(normalize_stage_scope(value) for value in scope)
        self.placeholders = {}
        for value in self.scope:
            parts = Path(value.rstrip('/')).parts
            if any(part.casefold() in {'.git', '.ai-agents-review'} for part in parts):
                raise ModelWorkspaceError('Git metadata·리뷰 자료에 쓰기 권한을 줄 수 없습니다.')
            target = self.path / value.rstrip('/')
            if value.endswith('/'):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.touch()
                    self.placeholders[value] = target.stat().st_mtime_ns
                if not target.is_file():
                    raise ModelWorkspaceError('파일 scope가 일반 파일을 가리키지 않습니다.')
        if review is not None:
            directory = self.path / '.ai-agents-review'
            if directory.exists():
                raise ModelWorkspaceError('리뷰 snapshot의 예약 경로가 저장소 파일과 충돌합니다.')
            directory.mkdir()
            (directory / 'diff.patch').write_bytes(base64.b64decode(review['patch_base64'], validate=True)
                if 'patch_base64' in review else review['patch'].encode('utf-8'))
            (directory / 'manifest.json').write_text(
                json.dumps({key: value for key, value in review.items() if key not in {'patch', 'patch_base64'}},
                           ensure_ascii=False, indent=2), encoding='utf-8')
        self.policy = self.root / 'policy.json'
        self.policy.write_text(json.dumps({
            'version': 1, 'workspace': str(self.path.resolve()), 'scope': list(self.scope),
            'image': image,
            'cache_mounts': [{'name': mount.name, 'target': mount.target} for mount in cache_mounts],
        }), encoding='utf-8')

    @staticmethod
    def _files(root: Path) -> dict[str, str]:
        result = {}
        for parent, directories, files in os.walk(root, followlinks=False):
            parent_path = Path(parent)
            directories[:] = [name for name in directories if name.casefold() != '.git']
            for name in [*directories, *files]:
                path = parent_path / name
                attributes = getattr(path.lstat(), 'st_file_attributes', 0)
                if path.is_symlink() or attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
                    raise ModelWorkspaceError('모델 snapshot의 symlink/junction은 허용하지 않습니다.')
            for name in files:
                if name.casefold() == '.git':
                    continue
                path = parent_path / name
                if not path.is_file():
                    raise ModelWorkspaceError('모델 snapshot에는 일반 파일만 허용합니다.')
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def preserve_changes(self) -> tuple[str, ...]:
        if not self.scope:
            return ()
        after = self._files(self.path)
        for value, timestamp in self.placeholders.items():
            path = self.path / value
            if path.is_file() and path.stat().st_size == 0 and path.stat().st_mtime_ns == timestamp:
                after.pop(value, None)
        changed = tuple(sorted(path for path in set(self.before) | set(after)
                               if self.before.get(path) != after.get(path)))
        if any(not any(path == scope or (scope.endswith('/') and path.startswith(scope))
                       for scope in self.scope) for path in changed):
            raise ModelWorkspaceError(f'모델이 scope 밖 snapshot을 변경했습니다. 보존 위치: {self.root}')
        if self._files(self.repository) != self.before:
            raise ModelWorkspaceError(f'호출 중 작업 공간이 변경되어 모델 파일을 자동 복사하지 않습니다. 보존 위치: {self.root}')
        # Validate every path before the first write. The run's OS execution lock
        # remains held by the coordinator across this copy and its Git checkpoint.
        for relative in changed:
            destination = self.repository / relative
            if relative not in after:
                destination.unlink()
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.path / relative, destination)
        return changed
