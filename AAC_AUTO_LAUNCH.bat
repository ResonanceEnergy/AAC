@echo off
echo ========================================
echo    AAC Auto-Launch + Git Sync System
echo ========================================
echo.
cd /d "c:\Users\gripa\OneDrive\Desktop\ACC\repositories\aac-main"
echo [+] Changed to AAC directory

echo [+] Checking git status...
git status --porcelain > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Not a git repository
    goto :error
)

echo [+] Adding changes...
git add . 2>nul

echo [+] Checking for changes...
git diff --cached --quiet
if %errorlevel% equ 0 (
    echo [INFO] No changes to commit
) else (
    echo [+] Committing changes...
    git commit -m "Auto-commit: AAC Matrix Monitor updates - %date% %time%" 2>nul
    if %errorlevel% equ 0 (
        echo [+] Commit successful
        echo [+] Pushing to GitHub...
        git push origin main 2>nul
        if %errorlevel% equ 0 (
            echo [+] Push successful
        ) else (
            echo [WARN] Push failed - check remote configuration
        )
    ) else (
        echo [WARN] Commit failed
    )
)

echo.
echo ========================================
echo    Launching AAC Matrix Monitor
echo ========================================
echo [+] Starting dashboard with browser auto-open...
python core/aac_master_launcher.py --dashboard-only --display-mode web

echo.
echo ========================================
echo    AAC System Ready!
echo ========================================
pause
exit /b 0

:error
echo [ERROR] Setup failed
pause
exit /b 1
