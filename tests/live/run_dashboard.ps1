# ──────────────────────────────────────────────────────────────────────────
# run_dashboard.ps1  —  Launch the Sotis dashboard pointed at the demo logs
# (safe to show on screen — no secrets). Run this in the second window first.
# ──────────────────────────────────────────────────────────────────────────

$repo = "F:\Sotis"
$env:SOTIS_LOG_DIR = "$repo\tests\live_test_workspace\logs"
Set-Location $repo
& "$repo\.venv\Scripts\python.exe" -m streamlit run "$repo\sotis\obs\app.py"
