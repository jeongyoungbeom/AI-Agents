$env:HERMES_HOME = 'D:\AI-Agents\hermes-home'
$hermes = 'D:\AI-Agents\hermes-home\bin\hermes.exe'

Write-Host 'This opens the ChatGPT device-code OAuth flow. No password is stored by this script.'
& $hermes auth add openai-codex
exit $LASTEXITCODE
