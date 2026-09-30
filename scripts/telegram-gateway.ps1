param(
    [ValidateSet('run', 'check')]
    [string]$Command = 'run',
    [switch]$Offline,
    [switch]$AllowLegacy
)

if (-not $AllowLegacy) {
    Write-Error "이 스크립트는 이전 명령형 게이트웨이입니다. 새 시스템은 scripts\chat-gateway.ps1을 사용하세요. 이전 실행이 꼭 필요하면 -AllowLegacy를 명시하세요."
    exit 2
}

$env:HERMES_HOME = 'D:\AI-Agents\hermes-home'
$python = 'D:\AI-Agents\runtime\hermes-agent\venv\Scripts\python.exe'
$gateway = 'D:\AI-Agents\coordinator\telegram_gateway.py'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Hermes Python not found: $python"
}

$arguments = @($gateway, $Command)
if ($Offline) {
    $arguments += '--offline'
}

& $python @arguments
exit $LASTEXITCODE
