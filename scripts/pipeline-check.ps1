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
    & $pythonPath -m unittest discover -s "tests\foundation" -v
    if ($LASTEXITCODE -ne 0) { throw "Foundation verification failed." }
    & $pythonPath -m unittest discover -s "tests\gateway" -v
    if ($LASTEXITCODE -ne 0) { throw "Gateway verification failed." }
    & $pythonPath -m unittest discover -s "tests\pipeline" -v
    if ($LASTEXITCODE -ne 0) { throw "Pipeline verification failed." }
}
finally {
    Pop-Location
}
