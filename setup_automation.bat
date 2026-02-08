@echo off
REM AAC Automation Scheduler
REM Creates Windows scheduled tasks for automated AAC operations

echo ========================================
echo    AAC Automation Scheduler Setup
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [ADMIN] Running with administrator privileges
) else (
    echo [WARN] Not running as administrator - some features may not work
)

echo [+] Setting up AAC automation tasks...
echo.

REM Create daily backup task
schtasks /create /tn "AAC Daily Backup" /tr "cmd /c cd /d \"c:\Users\gripa\OneDrive\Desktop\ACC\repositories\aac-main\" && python aac_automation.py --skip-tests" /sc daily /st 02:00 /rl highest /f >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Daily backup task created (2:00 AM)
) else (
    echo [WARN] Could not create daily backup task
)

REM Create system health check task
schtasks /create /tn "AAC Health Check" /tr "cmd /c cd /d \"c:\Users\gripa\OneDrive\Desktop\ACC\repositories\aac-main\" && python -c \"import asyncio; from aac_automation import AACAutomation; automation = AACAutomation(); asyncio.run(automation._check_system_health())\"" /sc hourly /rl highest /f >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Hourly health check task created
) else (
    echo [WARN] Could not create health check task
)

REM Create dashboard startup task (runs at user login)
schtasks /create /tn "AAC Dashboard Startup" /tr "cmd /c cd /d \"c:\Users\gripa\OneDrive\Desktop\ACC\repositories\aac-main\" && python aac_automation.py --dashboard-only" /sc onlogon /rl highest /f >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Dashboard startup task created (runs at login)
) else (
    echo [WARN] Could not create dashboard startup task
)

echo.
echo ========================================
echo    Current AAC Scheduled Tasks
echo ========================================
echo.
schtasks /query /tn "AAC*" | findstr "AAC"
if %errorlevel% neq 0 (
    echo [INFO] No AAC tasks found
)

echo.
echo ========================================
echo    Automation Setup Complete!
echo ========================================
echo.
echo Available automation options:
echo.
echo 1. AAC_AUTO_LAUNCH.bat     - One-click launch + git sync
echo 2. aac_automation.py        - Full Python automation script
echo 3. Scheduled tasks          - Automatic daily operations
echo.
echo Quick commands:
echo python aac_automation.py                    - Full automation
echo python aac_automation.py --dashboard-only   - Dashboard only
echo python aac_automation.py --skip-git         - Skip git operations
echo.
pause