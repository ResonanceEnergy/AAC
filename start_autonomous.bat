@echo off
title AAC Autonomous Engine - BARREN WUFFET
echo ============================================================
echo   AAC AUTONOMOUS ENGINE - 24/7 Mode
echo   Identity: BARREN WUFFET / AZ SUPREME
echo ============================================================
echo.

:: Activate venv (Python 3.12 - REQUIRED, system 3.14 breaks ib_insync)
call "C:\Users\gripa\.aac_venv\Scripts\activate.bat"

:: Navigate to project root
cd /d "%~dp0"

:: Ensure logs directory exists
if not exist "logs" mkdir logs

echo Starting autonomous engine...
echo Press Ctrl+C to shutdown gracefully.
echo.

python core\autonomous_engine.py

echo.
echo Engine stopped.
pause
