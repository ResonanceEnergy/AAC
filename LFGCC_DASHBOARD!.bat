@echo off
echo ========================================
echo    AAC Matrix Monitor - Dashboard Only
echo ========================================
echo.
cd /d "c:\Users\gripa\OneDrive\Desktop\ACC\repositories\aac-main"
echo [+] Changed to AAC directory
echo [+] Starting AAC Matrix Monitor (Web Dashboard)...
echo [+] Will launch on http://localhost:8080
echo [+] Browser will open automatically...
echo.
python core/aac_master_launcher.py --dashboard-only --display-mode web
echo.
echo ========================================
echo    Matrix Monitor Launch Complete!
echo    Matrix Monitor: http://localhost:8080
echo ========================================
echo.
pause