# Launcher for EraseAll.
#
# Port 8000 is a popular default (Django, Docker dashboards, other dev servers),
# so we never assume it's free: we scan upward for a free one, and if the thing
# already on a port turns out to be THIS app we just open it instead of starting
# a second copy. The browser is opened only once the server actually answers -
# opening it up front lands the tab on "connection refused" while uvicorn is
# still importing torch, which looks exactly like a broken app.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSCommandPath
Set-Location $root

$FIRST_PORT = 8000
$LAST_PORT  = 8020

function Write-Step($msg)  { Write-Host "   $msg" }
function Write-Bad($msg)   { Write-Host "   $msg" -ForegroundColor Red }
function Write-Warn($msg)  { Write-Host "   $msg" -ForegroundColor Yellow }
function Write-Good($msg)  { Write-Host "   $msg" -ForegroundColor Green }

Write-Host ""
Write-Host "   EraseAll" -ForegroundColor Cyan
Write-Host "   ------------------" -ForegroundColor Cyan

# --- prerequisites ----------------------------------------------------------
$py = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Bad "Setup needed: the 'venv' folder is missing."
    Write-Step "Open PowerShell in this folder and run the install steps in README.md:"
    Write-Step "  py -m venv venv"
    Write-Step "  venv\Scripts\pip install -r requirements.txt"
    Write-Step "  venv\Scripts\python download_models.py"
    exit 1
}

$missing = @()
foreach ($m in @('big-lama.pt', 'mobile_sam.pt')) {
    if (-not (Test-Path (Join-Path $root "models\$m"))) { $missing += $m }
}
if ($missing.Count -gt 0) {
    Write-Bad "Missing AI model(s): $($missing -join ', ')"
    Write-Step "Run this once (needs internet, ~240 MB):"
    Write-Step "  venv\Scripts\python download_models.py"
    exit 1
}
if (-not (Test-Path (Join-Path $root 'models\yolo11n-seg.pt'))) {
    Write-Warn "People mode is unavailable (models\yolo11n-seg.pt missing)."
    Write-Warn "Everything else works. Run download_models.py to enable it."
}

# --- pick a port ------------------------------------------------------------
function Test-PortFree([int]$Port) {
    try {
        $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
        $l.Start(); $l.Stop()
        return $true
    } catch { return $false }
}

# Is the thing on this port our own editor rather than someone else's server?
function Test-OurApp([int]$Port) {
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:$Port/api/health" -TimeoutSec 2 -UseBasicParsing
        return ($r.Content -match '"sam"')
    } catch { return $false }
}

$port = 0
for ($p = $FIRST_PORT; $p -le $LAST_PORT; $p++) {
    if (Test-PortFree $p) { $port = $p; break }
    if (Test-OurApp $p) {
        Write-Good "Already running on port $p - opening it in your browser."
        Start-Process "http://127.0.0.1:$p"
        Write-Step "(To stop it, close the other EraseAll window.)"
        exit 0
    }
    Write-Warn "Port $p is in use by another program - trying $($p + 1)..."
}

if ($port -eq 0) {
    Write-Bad "No free port between $FIRST_PORT and $LAST_PORT."
    Write-Step "Something is using them all. Restart your PC, or free a port and retry."
    exit 1
}

$url = "http://127.0.0.1:$port"
Write-Step "Starting on $url"
Write-Step "Your browser opens by itself when it's ready (about 5-10 seconds)."
Write-Host ""
Write-Step "Keep this window open while you work. Press Ctrl+C to stop."
Write-Host ""

# --- open the browser once the server actually answers ----------------------
$poll = '$d=(Get-Date).AddSeconds(240); while((Get-Date) -lt $d){ try{ $null=Invoke-WebRequest ''URLHERE/api/health'' -TimeoutSec 2 -UseBasicParsing; Start-Process ''URLHERE''; break } catch { Start-Sleep -Milliseconds 300 } }'
$poll = $poll.Replace('URLHERE', $url)
Start-Process powershell -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden', '-Command', $poll -WindowStyle Hidden

# --- run the server (blocks until Ctrl+C) -----------------------------------
& $py -m uvicorn server.main:app --host 127.0.0.1 --port $port
$code = $LASTEXITCODE

Write-Host ""
if ($code -ne 0 -and $code -ne $null) {
    Write-Bad "The server stopped unexpectedly (exit code $code)."
    Write-Step "Scroll up for the error. Common fixes:"
    Write-Step "  - reinstall packages:  venv\Scripts\pip install -r requirements.txt"
    Write-Step "  - re-download models:  venv\Scripts\python download_models.py"
} else {
    Write-Step "Server stopped."
}
