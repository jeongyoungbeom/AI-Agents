from __future__ import annotations

import base64
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.security import CredentialFileProtector
from tests.foundation.support import temporary_directory


class CredentialFileProtectorTests(unittest.TestCase):
    def test_windows_acl_protection_keeps_only_current_user_and_system(self):
        with temporary_directory() as directory:
            root = Path(directory)
            secrets = root / "config" / "secrets.env"
            auth = root / "hermes-home" / "auth.json"
            for path in (secrets, auth):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("secret=value", encoding="utf-8")
            protector = CredentialFileProtector(root, root / "hermes-home")
            with patch("app.services.security.files.os.name", "nt"), patch(
                "app.services.security.files.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                protected = protector.protect()

            self.assertEqual((secrets, auth), protected)
            script = base64.b64decode(run.call_args.args[0][-1]).decode("utf-16le")
            self.assertIn("SetAccessRuleProtection($true, $false)", script)
            self.assertIn("S-1-5-18", script)

    @unittest.skipUnless(os.name == "nt", "Windows ACL 전용 테스트")
    def test_windows_acl_is_applied_to_a_temporary_secret_file(self):
        with temporary_directory() as directory:
            root = Path(directory)
            secrets = root / "config" / "secrets.env"
            secrets.parent.mkdir(parents=True)
            secrets.write_text("token=temporary", encoding="utf-8")

            CredentialFileProtector(root, root / "hermes-home").protect()

            script = (
                "$acl = [System.IO.File]::GetAccessControl(" + repr(str(secrets)) + "); "
                "$acl.Access | ForEach-Object { "
                "$_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value }"
            )
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            identities = {line.strip() for line in result.stdout.splitlines() if line.strip()}
            self.assertIn("S-1-5-18", identities)
            self.assertEqual(2, len(identities))
