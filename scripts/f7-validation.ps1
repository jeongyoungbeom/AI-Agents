param(
    [switch]$Docker,
    [switch]$DockerOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

$agentRoot = "D:\AI-Agents"
$pythonPath = Join-Path $agentRoot "runtime\hermes-agent\venv\Scripts\python.exe"
$secureImage = "nikolaik/python-nodejs@sha256:8f958bdc1b4a422bfafd97cab4f69836401f616ae985d4b57a53d254f5bcb038"
$gradleImage = "gradle@sha256:83798adeb903471219ad918aadda1addb6067e2b9bb3e5332e9f3eb1a382bf43"
$previousDockerFixtureSetting = $env:AI_AGENTS_F7_DOCKER

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable not found: $pythonPath"
}
if ($DockerOnly -and -not $Docker) {
    throw "-DockerOnly requires -Docker."
}

function Invoke-F7Tests {
    param(
        [string]$Name,
        [string[]]$Tests
    )

    & $pythonPath -B -m unittest -v $Tests
    if ($LASTEXITCODE -ne 0) {
        throw "F-7 $Name validation failed with exit code $LASTEXITCODE."
    }
}

function Test-F7OptionalDockerImage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Image
    )

    # This optional check must remain a branch even when callers enable
    # PSNativeCommandUseErrorActionPreference globally.
    $previousNativeCommandErrorPreference = $PSNativeCommandUseErrorActionPreference
    try {
        $PSNativeCommandUseErrorActionPreference = $false
        docker image inspect $Image 1>$null 2>$null
        return $LASTEXITCODE -eq 0
    }
    finally {
        $PSNativeCommandUseErrorActionPreference = $previousNativeCommandErrorPreference
    }
}

function Invoke-F7Validation {
Push-Location -LiteralPath $agentRoot
try {
    if (-not $DockerOnly) {
        Invoke-F7Tests -Name "결정적" -Tests @(
            "tests.integration.test_f7_deterministic_load",
            "tests.integration.test_5d_operational_flow",
            "tests.foundation.test_budget",
            "tests.foundation.test_retention",
            "tests.gateway.test_attachments",
            "tests.gateway.test_review_hardening.ReviewHardeningTests.test_verification_policy_rejects_shell_chaining",
            "tests.gateway.test_conversation_foundation.ConversationQueueTests",
            "tests.gateway.test_team_conversation.TeamConversationRoutingTests.test_repository_instruction_like_call_is_blocked_before_scheduling",
            "tests.gateway.test_team_conversation.TeamConversationRoutingTests.test_repository_call_blocklist_covers_english_instruction_phrases",
            "tests.gateway.test_repository_batch_protocol",
            "tests.pipeline.test_pipeline_flow",
            "tests.pipeline.test_queue_and_permissions",
            "tests.pipeline.test_docker_isolation.DockerIsolationTests.test_created_container_cleanup_is_exact_and_idempotent",
            "tests.pipeline.test_docker_isolation.DockerIsolationTests.test_run_cleans_owned_container_on_success_failure_timeout_and_cancel",
            "tests.pipeline.test_docker_isolation.DockerIsolationTests.test_parallel_cleanup_cannot_remove_another_operations_container",
            "tests.pipeline.test_docker_isolation.DockerIsolationTests.test_startup_cleanup_skips_live_and_unrelated_containers"
        )
    }

    if ($Docker) {
        docker version --format '{{.Server.Version}}'
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Desktop is not ready."
        }

        docker image inspect $secureImage *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Required F-7 Python/Node image is not local: $secureImage"
        }

        $env:AI_AGENTS_F7_DOCKER = "1"
        Invoke-F7Tests -Name "Docker" -Tests @(
            "tests.gateway.test_repository_reader",
            "tests.pipeline.test_docker_isolation.DockerIsolationTests.test_git_commit_and_verification_run_through_the_container_boundary",
            "tests.pipeline.test_docker_isolation.DockerIsolationTests.test_linked_worktree_commit_and_verification_keep_source_unchanged_until_apply",
            "tests.integration.test_f7_docker_toolchains.F7DockerToolchainTests.test_python_and_node_fixture_runs_through_the_pinned_profile"
        )

        if (Test-F7OptionalDockerImage -Image $gradleImage) {
            Invoke-F7Tests -Name "Kotlin/Gradle Docker fixture" -Tests @(
                "tests.integration.test_f7_docker_toolchains.F7DockerToolchainTests.test_kotlin_gradle_fixture_runs_without_network_or_dependencies"
            )
        }
        else {
            Write-Warning "Kotlin/Gradle fixture는 image가 없어 실행하지 않았습니다: $gradleImage"
            Write-Host "이미지 다운로드는 자동으로 수행하지 않습니다. 사용자 승인 후 준비한 뒤 -Docker를 다시 실행하세요."
        }
    }

    # 선택 이미지가 없으면 docker inspect의 종료 코드 1이 남는다.
    # 필수 검증이 끝났으므로 호출자에게도 성공을 반환한다.
    $global:LASTEXITCODE = 0
    if ($DockerOnly) {
        Write-Host "F-7 Docker 검증을 통과했습니다. 실제 Telegram/OAuth 모델 호출은 실행하지 않았습니다."
    }
    else {
        Write-Host "F-7 결정적 검증을 통과했습니다. 실제 Telegram/OAuth 모델 호출은 실행하지 않았습니다."
    }
}
finally {
    if ($null -eq $previousDockerFixtureSetting) {
        Remove-Item Env:AI_AGENTS_F7_DOCKER -ErrorAction SilentlyContinue
    }
    else {
        $env:AI_AGENTS_F7_DOCKER = $previousDockerFixtureSetting
    }
    Pop-Location
}
}

if ($MyInvocation.InvocationName -ne ".") {
    Invoke-F7Validation
}
