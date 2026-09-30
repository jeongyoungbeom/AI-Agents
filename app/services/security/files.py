from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path


class CredentialFileProtector:
    """Limits known local credential files to the current user and SYSTEM."""

    def __init__(self, root: Path, hermes_home: Path):
        self.root = root.resolve()
        self.hermes_home = hermes_home.resolve()

    def protect(self) -> tuple[Path, ...]:
        protected: list[Path] = []
        for path in self.paths():
            if not path.is_file():
                continue
            self._protect_file(path)
            protected.append(path)
        return tuple(protected)

    def paths(self) -> tuple[Path, ...]:
        return (
            self.root / "config" / "secrets.env",
            self.hermes_home / "auth.json",
            self.hermes_home / "auth.lock",
            self.hermes_home / "state.json",
        )

    @staticmethod
    def _protect_file(path: Path) -> None:
        if os.name != "nt":
            path.chmod(0o600)
            return
        encoded = base64.b64encode(
            (
                "$path = " + json.dumps(str(path)) + "; "
                "$current = [Security.Principal.WindowsIdentity]::GetCurrent().User; "
                "$system = [Security.Principal.SecurityIdentifier]::new('S-1-5-18'); "
                "$acl = [System.IO.File]::GetAccessControl($path); "
                "$acl.SetAccessRuleProtection($true, $false); "
                "foreach ($rule in @($acl.Access)) { $acl.RemoveAccessRuleSpecific($rule) }; "
                "$inheritance = [Security.AccessControl.InheritanceFlags]::None; "
                "$propagation = [Security.AccessControl.PropagationFlags]::None; "
                "foreach ($identity in @($current, $system)) { "
                "$rule = [Security.AccessControl.FileSystemAccessRule]::new($identity, "
                "[Security.AccessControl.FileSystemRights]::FullControl, $inheritance, $propagation, "
                "[Security.AccessControl.AccessControlType]::Allow); $acl.AddAccessRule($rule) }; "
                "[System.IO.File]::SetAccessControl($path, $acl)"
            ).encode("utf-16le")
        ).decode("ascii")
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            detail = result.stderr.strip().replace("\r", " ").replace("\n", " ")
            raise RuntimeError(
                "credential file ACL update failed" + (f": {detail[:300]}" if detail else "")
            )
