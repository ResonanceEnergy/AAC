@echo off
echo ========================================
echo    AAC Matrix Monitor - Launch Test
echo ========================================
echo.
echo Testing automatic browser opening...
echo Matrix Monitor will start and browser should open in 3 seconds
echo.
cd /d "c:\Users\gripa\OneDrive\Desktop\ACC\repositories\aac-main"
python core/aac_master_launcher.py --dashboard-only --display-mode web
echo.
echo Test complete!
pause