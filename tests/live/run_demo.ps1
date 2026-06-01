# ──────────────────────────────────────────────────────────────────────────
# run_demo.ps1  —  Launch the Sotis live demo (safe to show on screen)
#
# Loads the API key from demo.secret.ps1 (gitignored, never recorded), sets the
# non-sensitive env vars, and runs the LangGraph eval. The only thing visible
# on camera is `.\run_demo.ps1` — no secrets.
#
# Usage:
#   1. Put your key in tests/live/demo.secret.ps1
#   2. From repo root:  .\tests\live\run_demo.ps1
# ──────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$repo = "F:\Sotis"
$ws   = "$repo\tests\live_test_workspace"

# 1. Load the secret key (gitignored, off-camera)
$secret = "$repo\tests\live\demo.secret.ps1"
if (-not (Test-Path $secret)) {
    Write-Host "[demo] Missing $secret — copy your API key in there first." -ForegroundColor Red
    exit 1
}
. $secret
if (-not $env:OPENAI_API_KEY -or $env:OPENAI_API_KEY -eq "PUT_YOUR_OPENROUTER_KEY_HERE") {
    Write-Host "[demo] Set your real key in demo.secret.ps1 first." -ForegroundColor Red
    exit 1
}

# 2. Non-sensitive config (safe on camera)
$env:OPENAI_API_BASE  = "https://openrouter.ai/api/v1"
$env:OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
$env:WORKSPACE_DIR    = $ws
$env:SOTIS_LOG_DIR    = "$ws\logs"

Write-Host "[demo] Launching Sotis agent run (model: $env:OPENROUTER_MODEL)..." -ForegroundColor Cyan
Set-Location $ws
& "$repo\.venv\Scripts\python.exe" "$repo\tests\live\eval_langgraph.py"
