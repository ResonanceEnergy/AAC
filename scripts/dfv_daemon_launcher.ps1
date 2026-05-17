param()
# DFV daemon launcher — invoked by Windows Task \NCC\DFV-Daemon on logon.
# Keeps logs in C:\dev\AAC_fresh\logs\dfv_daemon.{out,err}.log and survives reboots.

$ErrorActionPreference = 'Stop'
Set-Location -Path 'C:\dev\AAC_fresh'

$logDir = 'C:\dev\AAC_fresh\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$python = 'C:\dev\AAC_fresh\.venv\Scripts\python.exe'
$launch = 'C:\dev\AAC_fresh\launch.py'

# Guard: don't spawn a second daemon if one is already alive.
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
Where-Object { $_.CommandLine -match 'launch\.py\s+dfv' }
if ($existing) {
    Write-Output ("dfv daemon already running (PID {0}); exiting." -f ($existing.ProcessId -join ','))
    exit 0
}

# Start the daemon detached so the scheduler task completes immediately,
# leaving the Python process to run for the session.
$out = Join-Path $logDir 'dfv_daemon.out.log'
$err = Join-Path $logDir 'dfv_daemon.err.log'
Start-Process -FilePath $python `
    -ArgumentList @($launch, 'dfv') `
    -WorkingDirectory 'C:\dev\AAC_fresh' `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err
Write-Output "dfv daemon launched."
