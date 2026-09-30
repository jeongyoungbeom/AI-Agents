param(
    [ValidateSet("run", "check", "start", "stop", "restart", "status", "logs", "secure", "backup", "purge", "restore")]
    [string]$Command = "run",
    [switch]$Offline,
    [string]$RunId = "",
    [string]$Backup = "",
    [ValidateRange(1, 5000)]
    [int]$Tail = 100
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$agentRoot = "D:\AI-Agents"
$pythonPath = Join-Path $agentRoot "runtime\hermes-agent\venv\Scripts\python.exe"
$pythonWindowlessPath = Join-Path $agentRoot "runtime\hermes-agent\venv\Scripts\pythonw.exe"
$stopRequestPath = Join-Path $agentRoot "data\gateway.stop"
$gatewayTaskName = "AI-Agents-Telegram-Gateway"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $pythonWindowlessPath -PathType Leaf)) {
    throw "Windowless Python executable not found: $pythonWindowlessPath"
}

function Get-GatewayProcess {
    try {
        # GatewayInstanceLock deliberately keeps the lock file open without a
        # shared read handle on Windows. Inspect the exact Python command line
        # instead of trying to read that locked PID hint.
        $processes = @(
            Get-CimInstance Win32_Process -ErrorAction Stop |
                Where-Object { $_.CommandLine -match 'app\.gateway\.cli\s+run' }
        )
        $processIds = @($processes | ForEach-Object { [int]$_.ProcessId })
        # The Hermes virtual-environment launcher starts the real interpreter
        # as its child with the same command line. Treat that pair as one
        # gateway instance and keep the outer process for start/stop status.
        $roots = @(
            $processes | Where-Object {
                $processIds -notcontains [int]$_.ParentProcessId
            }
        )
        if ($roots.Count -gt 1) {
            throw "게이트웨이 프로세스가 둘 이상 감지됐습니다. logs\gateway.log를 확인해 주세요."
        }
        return $roots | Select-Object -First 1
    }
    catch {
        return $null
    }
}

function Start-Gateway {
    $existing = Get-GatewayProcess
    if ($null -ne $existing) {
        Write-Output "게이트웨이가 이미 실행 중입니다. PID: $($existing.ProcessId)"
        return
    }
    Remove-Item -LiteralPath $stopRequestPath -Force -ErrorAction SilentlyContinue
    $action = New-ScheduledTaskAction `
        -Execute $pythonWindowlessPath `
        -Argument "-m app.gateway.cli run" `
        -WorkingDirectory $agentRoot
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit ([TimeSpan]::Zero)
    Register-ScheduledTask `
        -TaskName $gatewayTaskName `
        -Action $action `
        -Settings $settings `
        -Description "AI-Agents Telegram gateway (manual start; no startup trigger)" `
        -Force | Out-Null
    Start-ScheduledTask -TaskName $gatewayTaskName
    $deadline = (Get-Date).AddSeconds(10)
    do {
        Start-Sleep -Milliseconds 250
        $existing = Get-GatewayProcess
        $taskInfo = Get-ScheduledTaskInfo -TaskName $gatewayTaskName -ErrorAction SilentlyContinue
    } while ($null -eq $existing -and (Get-Date) -lt $deadline)
    if ($null -eq $existing) {
        $lastResult = if ($null -ne $taskInfo) { $taskInfo.LastTaskResult } else { "unknown" }
        throw "게이트웨이 시작을 확인하지 못했습니다. 예약 작업 결과: $lastResult. logs\gateway.log를 확인해 주세요."
    }
    Write-Output "게이트웨이를 백그라운드로 시작했습니다. PID: $($existing.ProcessId)"
}

function Stop-Gateway {
    $existing = Get-GatewayProcess
    if ($null -eq $existing) {
        Write-Output "실행 중인 게이트웨이가 없습니다."
        return
    }
    New-Item -ItemType File -Path $stopRequestPath -Force | Out-Null
    $deadline = (Get-Date).AddSeconds(45)
    do {
        Start-Sleep -Milliseconds 500
        $existing = Get-GatewayProcess
    } while ($null -ne $existing -and (Get-Date) -lt $deadline)
    if ($null -ne $existing) {
        throw "안전 종료 시간이 초과됐습니다. 현재 작업을 확인한 뒤 다시 stop을 실행해 주세요."
    }
    Write-Output "게이트웨이를 안전하게 종료했습니다."
}

switch ($Command) {
    "start" { Start-Gateway; exit 0 }
    "stop" { Stop-Gateway; exit 0 }
    "restart" { Stop-Gateway; Start-Gateway; exit 0 }
    "status" {
        $existing = Get-GatewayProcess
        if ($null -eq $existing) {
            Write-Output "게이트웨이: 중지됨"
            exit 1
        }
        Write-Output "게이트웨이: 실행 중 (PID: $($existing.ProcessId), 시작: $($existing.CreationDate))"
        exit 0
    }
    "logs" {
        $logPath = if ($RunId) { Join-Path $agentRoot "artifacts\$RunId\timeline.log" } else { Join-Path $agentRoot "logs\gateway.log" }
        if (-not (Test-Path -LiteralPath $logPath -PathType Leaf)) {
            throw "로그 파일을 찾을 수 없습니다: $logPath"
        }
        Get-Content -LiteralPath $logPath -Tail $Tail
        exit 0
    }
}

$arguments = @("-m", "app.gateway.cli", $Command)
if ($Offline) {
    $arguments += "--offline"
}
if ($Backup) {
    $arguments += @("--backup", $Backup)
}

Push-Location -LiteralPath $agentRoot
try {
    & $pythonPath @arguments
    exit $LASTEXITCODE
}
finally { Pop-Location }
