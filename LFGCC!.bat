@echo off
echo ========================================
echo    LFGCC! - AAC Master System Launcher
echo ========================================
echo.
cd /d "c:\Users\gripa\OneDrive\Desktop\ACC\repositories\aac-main"
echo [+] Changed to AAC directory
echo [+] Starting AAC Master System (Complete Launch)...
echo [+] This will launch: Doctrine Compliance + Department Agents + Trading Systems + Monitoring
echo [+] Web Dashboard will be available at: http://localhost:8080
echo [+] Opening web browser automatically...
echo.
python core/aac_master_launcher.py
start http://localhost:8080
echo.
echo ========================================
echo    AAC Master System Launch Complete!
echo    Web Dashboard: http://localhost:8080
echo ========================================
echo.
pause
