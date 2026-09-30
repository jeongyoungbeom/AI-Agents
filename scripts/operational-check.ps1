param(
    [switch]$Online
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

$agentRoot = "D:\AI-Agents"
$pythonPath = Join-Path $agentRoot "runtime\hermes-agent\venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable not found: $pythonPath"
}

Push-Location -LiteralPath $agentRoot
try {
    $checkArguments = @("-m", "app.gateway.cli", "check")
    if (-not $Online) {
        $checkArguments += "--offline"
    }
    & $pythonPath @checkArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Gateway check failed with exit code $LASTEXITCODE."
    }

    & $pythonPath -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Operational regression tests failed with exit code $LASTEXITCODE."
    }

    Write-Host "5-D 운영 점검을 통과했습니다."
}
finally {
    Pop-Location
}
