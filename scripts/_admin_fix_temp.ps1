# Temp script - run elevated to fix 3 remaining task scheduler issues
# Safe to delete after running

$log = "C:\dev\AAC_fresh\scripts\_admin_fix_log.txt"
$out = @()

# 1. Remove AAC_Startup_OnLogon (disabled, redundant with NCC-MasterLauncher)
try {
    Unregister-ScheduledTask -TaskName "AAC_Startup_OnLogon" -Confirm:$false -ErrorAction Stop
    $out += "OK: Removed AAC_Startup_OnLogon"
} catch {
    $out += "FAIL AAC_Startup_OnLogon: $($_.Exception.Message)"
}

# 2. Fix Resonance Repo Auto - Daily (wrong path + not silent)
try {
    $task = Get-ScheduledTask -TaskName "Resonance Repo Auto - Daily" -ErrorAction Stop
    $task.Actions[0].Execute = "wscript.exe"
    $task.Actions[0].Arguments = '"C:\dev\silent-launch.vbs" "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\dev\resonance-uy-py\run-auto.ps1"'
    $task | Set-ScheduledTask -ErrorAction Stop | Out-Null
    $out += "OK: Fixed Resonance Repo Auto - Daily (path + silent)"
} catch {
    $out += "FAIL Resonance Daily: $($_.Exception.Message)"
}

# 3. Remove Resonance Repo Auto - Logon (disabled, wrong path, redundant)
try {
    Unregister-ScheduledTask -TaskName "Resonance Repo Auto - Logon" -Confirm:$false -ErrorAction Stop
    $out += "OK: Removed Resonance Repo Auto - Logon"
} catch {
    $out += "FAIL Resonance Logon: $($_.Exception.Message)"
}

$out | Out-File -FilePath $log -Encoding utf8
$out | ForEach-Object { Write-Host $_ }
