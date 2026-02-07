<#
.SYNOPSIS
    Accelerated Arbitrage Corp - Windows Startup Script

.DESCRIPTION
    Start the ACC trading system in various modes.

.PARAMETER Mode
    Execution mode: paper, dry-run, live, check

.EXAMPLE
    .\Start-ACC.ps1 -Mode paper
    .\Start-ACC.ps1 -Mode check
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("paper", "dry-run", "live", "check", "test")]
    [string]$Mode = "paper"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host @"

╔══════════════════════════════════════════════════════════════════════════════╗
║     █████╗  ██████╗ ██████╗    Accelerated Arbitrage Corp                   ║
║    ██╔══██╗██╔════╝██╔════╝    ══════════════════════════                   ║
║    ███████║██║     ██║         Windows Startup Script                        ║
║    ██╔══██║██║     ██║                                                       ║
║    ██║  ██║╚██████╗╚██████╗                                                  ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Ensure we're in the project directory
Set-Location $ScriptDir

# Check Python
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}
$pythonVersion = python --version
Write-Host "       Found: $pythonVersion" -ForegroundColor Green

# Check virtual environment
Write-Host "[2/4] Checking virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $ScriptDir ".venv"
if (Test-Path $venvPath) {
    Write-Host "       Activating existing venv..." -ForegroundColor Green
    & "$venvPath\Scripts\Activate.ps1"
} else {
    Write-Host "       Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvPath
    & "$venvPath\Scripts\Activate.ps1"
    Write-Host "       Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Check .env
Write-Host "[3/4] Checking configuration..." -ForegroundColor Yellow
$envFile = Join-Path $ScriptDir ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "       WARNING: .env not found!" -ForegroundColor Yellow
    $envExample = Join-Path $ScriptDir ".env.example"
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Host "       Created .env from .env.example - please configure it!" -ForegroundColor Yellow
    }
}

# Create data directories
Write-Host "[4/4] Ensuring directories exist..." -ForegroundColor Yellow
$dirs = @("data", "logs")
foreach ($dir in $dirs) {
    $path = Join-Path $ScriptDir $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}
Write-Host "       Ready!" -ForegroundColor Green

Write-Host ""
Write-Host "Starting ACC in $Mode mode..." -ForegroundColor Cyan
Write-Host "═" * 60

switch ($Mode) {
    "paper" {
        python main.py --paper
    }
    "dry-run" {
        python main.py --dry-run
    }
    "live" {
        Write-Host "WARNING: Live trading mode!" -ForegroundColor Red
        Write-Host "This will execute REAL trades with REAL money!" -ForegroundColor Red
        $confirm = Read-Host "Type 'YES' to confirm"
        if ($confirm -eq "YES") {
            python main.py --live
        } else {
            Write-Host "Cancelled." -ForegroundColor Yellow
        }
    }
    "check" {
        python main.py --check
    }
    "test" {
        Write-Host "Running test suite..." -ForegroundColor Cyan
        python -m pytest tests/ -v
    }
}
