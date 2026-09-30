param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('preflight', 'validate', 'run', 'status', 'notify-test')]
    [string]$Command,

    [Parameter(Position = 1)]
    [string]$Target,

    [switch]$AllowMissingAuth,
    [switch]$AllowLegacy
)

if (-not $AllowLegacy) {
    Write-Error "이 스크립트는 이전 명령형 파이프라인입니다. 현재 개발 기준이 아니며, 실행하려면 -AllowLegacy를 명시하세요."
    exit 2
}

$python = 'D:\AI-Agents\runtime\hermes-agent\venv\Scripts\python.exe'
$coordinator = 'D:\AI-Agents\coordinator\agent_pipeline.py'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Hermes Python was not found: $python"
}

$arguments = @($coordinator, $Command)
if ($Target) {
    $arguments += $Target
}
if ($AllowMissingAuth) {
    $arguments += '--allow-missing-auth'
}

& $python @arguments
exit $LASTEXITCODE
