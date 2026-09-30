from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
PWSH = shutil.which("pwsh")


class F7ValidationScriptTests(unittest.TestCase):
    @unittest.skipUnless(PWSH, "PowerShell 7 (pwsh)가 필요합니다.")
    def test_optional_image_miss_keeps_successful_validation_exit_code_zero(self):
        script = AI_ROOT / "scripts" / "f7-validation.ps1"
        quoted_script = str(script).replace("'", "''")
        command = f"""
$ErrorActionPreference = 'Stop'
. '{quoted_script}' -Docker -DockerOnly
function Invoke-F7Tests {{ param($Name, $Tests) }}
function docker {{
    if ($args[0] -eq 'version') {{ $global:LASTEXITCODE = 0; '28.3.3'; return }}
    if ($args[0] -eq 'image' -and $args[1] -eq 'inspect') {{
        $global:LASTEXITCODE = if ($args[2] -like 'nikolaik/*') {{ 0 }} else {{ 1 }}
        return
    }}
}}
Invoke-F7Validation
if ($LASTEXITCODE -ne 0) {{ throw "Successful required checks left exit code $LASTEXITCODE" }}
"""
        result = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
            text=True, encoding="utf-8", errors="replace", capture_output=True,
            check=False, timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    @unittest.skipUnless(PWSH, "PowerShell 7 (pwsh)가 필요합니다.")
    def test_missing_optional_gradle_image_is_a_warning_branch(self):
        """Native-command errors must not turn an optional image miss into a throw."""
        script = AI_ROOT / "scripts" / "f7-validation.ps1"
        quoted_script = str(script).replace("'", "''")
        command = f"""
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
function docker {{
    $script:nativeErrorPreferenceAtInspect = $PSNativeCommandUseErrorActionPreference
    if ($PSNativeCommandUseErrorActionPreference) {{
        throw 'The optional image inspection did not suppress native-command errors.'
    }}
    $global:LASTEXITCODE = 1
}}
. '{quoted_script}'
if (Test-F7OptionalDockerImage -Image 'gradle@sha256:not-local') {{
    throw 'The fake Docker image must be unavailable.'
}}
if ($script:nativeErrorPreferenceAtInspect) {{
    throw 'The optional image inspection inherited native-command errors.'
}}
if (-not $PSNativeCommandUseErrorActionPreference) {{
    throw 'The caller native-command preference was not restored.'
}}
"""
        result = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=10,
        )

        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
