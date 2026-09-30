from __future__ import annotations

import tempfile
from pathlib import Path


AI_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = AI_ROOT / "artifacts"


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="foundation-test-", dir=ARTIFACTS_ROOT)
