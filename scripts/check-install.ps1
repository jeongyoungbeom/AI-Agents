$pipeline = 'D:\AI-Agents\scripts\agent-pipeline.ps1'
& $pipeline preflight -AllowMissingAuth -AllowLegacy
exit $LASTEXITCODE
