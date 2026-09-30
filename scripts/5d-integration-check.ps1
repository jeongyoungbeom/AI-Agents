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
        throw "Gateway readiness check failed with exit code $LASTEXITCODE."
    }

    & $pythonPath -m unittest tests.integration.test_5d_operational_flow -v
    if ($LASTEXITCODE -ne 0) {
        throw "5-D integration flow failed with exit code $LASTEXITCODE."
    }

    Write-Host "5-D 통합 운영 흐름 점검을 통과했습니다. Telegram 메시지, GitHub push, PR은 수행하지 않았습니다."
}
finally {
    Pop-Location
}
