@echo off
title AAC BARREN WUFFET — Full Launch
echo ============================================================
echo   AAC BARREN WUFFET / AZ SUPREME — Full Launch
echo   Gateways + Paper Trading + Matrix Monitor
echo ============================================================
echo.

:: Navigate to project root
cd /d "%~dp0"

:: Ensure logs directory exists
if not exist "logs" mkdir logs

:: Use local .venv (Python 3.10)
if not exist ".venv\Scripts\python.exe" (
    echo [!] No .venv found. Run: python setup_machine.py
    pause
    exit /b 1
)

echo Starting all systems (preflight + gateways + matrix monitor + paper engine)...
echo Press Ctrl+C to shutdown gracefully.
echo.
.venv\Scripts\python.exe launch.py all --display web

echo.
echo Engine stopped.
pause
