Enable-ScheduledTask -TaskName "OpenD-Headless" -TaskPath "\AAC\" | Out-Null
Start-ScheduledTask -TaskName "OpenD-Headless" -TaskPath "\AAC\"
Start-Sleep -Seconds 20
Write-Host "===Process==="
Get-Process OpenD -ErrorAction SilentlyContinue | Format-Table Id, Name, StartTime -AutoSize
Write-Host "===Port 11111==="
$c = Get-NetTCPConnection -LocalPort 11111 -State Listen -ErrorAction SilentlyContinue
if ($c) { $c | Format-Table LocalAddress, LocalPort, OwningProcess -AutoSize } else { Write-Host "NOT LISTENING YET" }
Write-Host "===Latest GTW log tail==="
$log = Get-ChildItem "$env:APPDATA\com.moomoo.OpenD\Log\GTWLog_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($log) { 
    Write-Host "Log: $($log.FullName)"
    Get-Content $log.FullName -Tail 20 
} else { Write-Host "No GTW log found" }
Write-Host "===Task state==="
Get-ScheduledTask -TaskName "OpenD-Headless" -TaskPath "\AAC\" | Select-Object TaskName, State | Format-Table -AutoSize
Get-ScheduledTaskInfo -TaskName "OpenD-Headless" -TaskPath "\AAC\" | Format-List LastRunTime, LastTaskResult
