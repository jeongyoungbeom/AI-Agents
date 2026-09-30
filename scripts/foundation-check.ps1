$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"

$agentRoot = "D:\AI-Agents"
$pythonPath = Join-Path $agentRoot "runtime\hermes-agent\venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable not found: $pythonPath"
}

$previousBytecodeSetting = $env:PYTHONDONTWRITEBYTECODE
$locationWasPushed = $false

try {
    $env:PYTHONDONTWRITEBYTECODE = "1"
    Push-Location -LiteralPath $agentRoot
    $locationWasPushed = $true

    & $pythonPath -m unittest discover -s "tests\foundation" -v
    if ($LASTEXITCODE -ne 0) {
        throw "Foundation verification failed with exit code $LASTEXITCODE."
    }
}
finally {
    if ($locationWasPushed) {
        Pop-Location
    }
    if ($null -eq $previousBytecodeSetting) {
        Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONDONTWRITEBYTECODE = $previousBytecodeSetting
    }
}
