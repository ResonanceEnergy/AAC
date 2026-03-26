@echo off
:: ═══════════════════════════════════════════════════════════════
::  BARREN WUFFET — Unified Launcher (Windows)
::  Works on QUSAR and QFORGE.
::  If .venv exists, uses it.  Otherwise uses system Python.
::
::  Usage:  launch.bat dashboard
::          launch.bat paper
::          launch.bat git-sync
::  Setup:  python setup_machine.py
:: ═══════════════════════════════════════════════════════════════
cd /d "%~dp0"

:: Prefer .venv Python if it exists
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe launch.py %*
) else (
    echo [!] No .venv found. Run: python setup_machine.py
    python launch.py %*
)
