# ──────────────────────────────────────────────────────────────────────────
# run_demo.ps1  —  Launch the Sotis live demo (safe to show on screen)
#
# Loads API keys from demo.secret.ps1 (gitignored, never recorded), maps the
# chosen provider's key + endpoint + model, and runs the LangGraph eval.
# The only thing visible on camera is `.\run_demo.ps1 groq` — no secrets.
#
# Usage (from repo root):
#   .\tests\live\run_demo.ps1 groq          (default)
#   .\tests\live\run_demo.ps1 openrouter
#   .\tests\live\run_demo.ps1 google
# ──────────────────────────────────────────────────────────────────────────

param(
    [ValidateSet("groq", "openrouter", "google")]
    [string]$Provider = "groq"
)

$ErrorActionPreference = "Stop"
$repo = "F:\Sotis"
$ws   = "$repo\tests\live_test_workspace"

# 1. Load the secret keys (gitignored, off-camera)
$secret = "$repo\tests\live\demo.secret.ps1"
if (-not (Test-Path $secret)) {
    Write-Host "[demo] Missing $secret - put your keys there first." -ForegroundColor Red
    exit 1
}
. $secret

# 2. Map the chosen provider -> key, endpoint, model
switch ($Provider) {
    "groq" {
        $key  = $GROQ_KEY
        $env:OPENAI_API_BASE  = "https://api.groq.com/openai/v1"
        $env:GROQ_MODEL       = "llama-3.3-70b-versatile"
        Remove-Item Env:OPENROUTER_MODEL -ErrorAction SilentlyContinue
    }
    "openrouter" {
        $key  = $OPENROUTER_KEY
        $env:OPENAI_API_BASE  = "https://openrouter.ai/api/v1"
        $env:OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
        Remove-Item Env:GROQ_MODEL -ErrorAction SilentlyContinue
    }
    "google" {
        $key  = $GOOGLE_KEY
        $env:OPENAI_API_BASE  = "https://generativelanguage.googleapis.com/v1beta/openai/"
        Remove-Item Env:GROQ_MODEL, Env:OPENROUTER_MODEL -ErrorAction SilentlyContinue
    }
}

if (-not $key -or $key -like "PUT_YOUR_*") {
    Write-Host "[demo] No valid key for '$Provider' in demo.secret.ps1." -ForegroundColor Red
    exit 1
}
$env:OPENAI_API_KEY = $key

# 3. Non-sensitive config (safe on camera)
$env:WORKSPACE_DIR = $ws
$env:SOTIS_LOG_DIR = "$ws\logs"

Write-Host "[demo] Provider: $Provider" -ForegroundColor Cyan
Set-Location $ws
& "$repo\.venv\Scripts\python.exe" "$repo\tests\live\eval_langgraph.py"
