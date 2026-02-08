# Simple AAC LFGCC! Shortcut Creator

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\LFGCC!.lnk")
$Shortcut.TargetPath = "python.exe"
$Shortcut.Arguments = "`"$PSScriptRoot\aac_master_launcher.py`" --mode paper"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = "AAC Master Launcher - Let's F***ing Go Crypto Currency!"
$Shortcut.IconLocation = "shell32.dll,13"
$Shortcut.Save()

Write-Host "LFGCC! shortcut created on desktop!" -ForegroundColor Green