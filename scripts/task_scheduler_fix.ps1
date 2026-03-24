<#
.SYNOPSIS
    AAC/NCC Task Scheduler Audit and Cleanup
.DESCRIPTION
    Fixes all issues found in the 2026-03-23 audit.
    Run with no flags to preview. Run with -Fix to apply.
#>
param(
    [switch]$Fix
)

$ErrorActionPreference = 'Continue'
$SilentVbs = 'C:\dev\silent-launch.vbs'
$Changes = @()

function Report($action, $detail) {
    $tag = if ($Fix) { "[FIX]" } else { "[PREVIEW]" }
    $msg = "$tag $action -- $detail"
    Write-Host $msg -ForegroundColor $(if ($Fix) { 'Green' } else { 'Yellow' })
    $script:Changes += $msg
}

Write-Host ""
Write-Host "=== AAC/NCC Task Scheduler Cleanup ===" -ForegroundColor Cyan
Write-Host "Mode: $(if ($Fix) { 'APPLYING FIXES' } else { 'PREVIEW ONLY (use -Fix to apply)' })"
Write-Host ""

# 1. Fix AAC_Automated_Pipeline: cmd.exe -> wscript silent, extend timeout
$task = Get-ScheduledTask -TaskName 'AAC_Automated_Pipeline' -ErrorAction SilentlyContinue
if ($task) {
    if ($task.Actions[0].Execute -eq 'cmd.exe') {
        Report "AAC_Automated_Pipeline" "Switch cmd.exe to wscript silent + extend timeout to 30 min"
        if ($Fix) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument """$SilentVbs"" ""C:\dev\AAC_fresh\.venv\Scripts\python.exe C:\dev\AAC_fresh\automate.py --pipeline"""
            $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -Priority 7
            Set-ScheduledTask -TaskName 'AAC_Automated_Pipeline' -Action $action -Settings $settings
        }
    }
} else { Write-Host "[SKIP] AAC_Automated_Pipeline not found" -ForegroundColor DarkGray }

# 2. Fix AAC_RocketShip_DailyBriefing: cmd.exe -> wscript silent
$task = Get-ScheduledTask -TaskName 'AAC_RocketShip_DailyBriefing' -ErrorAction SilentlyContinue
if ($task) {
    if ($task.Actions[0].Execute -eq 'cmd.exe') {
        Report "AAC_RocketShip_DailyBriefing" "Switch cmd.exe to wscript silent"
        if ($Fix) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument """$SilentVbs"" ""C:\dev\AAC_fresh\.venv\Scripts\python.exe -m strategies.rocket_ship.daily_ops --once"""
            Set-ScheduledTask -TaskName 'AAC_RocketShip_DailyBriefing' -Action $action
        }
    }
} else { Write-Host "[SKIP] AAC_RocketShip_DailyBriefing not found" -ForegroundColor DarkGray }

# 3. Remove redundant NCC-Doctrine Weekly Full Backup (identical to Daily)
$task = Get-ScheduledTask -TaskName 'NCC-Doctrine Weekly Full Backup' -ErrorAction SilentlyContinue
if ($task) {
    Report "NCC-Doctrine Weekly Full Backup" "REMOVE -- identical command to Daily Backup (both -RunDataBackup)"
    if ($Fix) {
        Unregister-ScheduledTask -TaskName 'NCC-Doctrine Weekly Full Backup' -Confirm:$false
    }
}

# 4. Remove root NCC-WeeklyAudit (references non-existent trinity_audit.py)
$task = Get-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\' -ErrorAction SilentlyContinue
if ($task) {
    Report "Root NCC-WeeklyAudit" "REMOVE -- references non-existent runtime/trinity_audit.py"
    if ($Fix) {
        Unregister-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\' -Confirm:$false
    }
}

# 5. Remove disabled AAC_Startup_OnLogon (NCC-MasterLauncher handles logon)
$task = Get-ScheduledTask -TaskName 'AAC_Startup_OnLogon' -ErrorAction SilentlyContinue
if ($task) {
    Report "AAC_Startup_OnLogon" "REMOVE -- disabled, crashed (0x8007042B), NCC-MasterLauncher covers logon"
    if ($Fix) {
        Unregister-ScheduledTask -TaskName 'AAC_Startup_OnLogon' -Confirm:$false
    }
}

# 6. Fix Resonance Repo Auto - Daily: wrong path
$task = Get-ScheduledTask -TaskName 'Resonance Repo Auto - Daily' -ErrorAction SilentlyContinue
if ($task) {
    $args0 = $task.Actions[0].Arguments
    if ($args0 -match 'C:\\resonance-uy-py') {
        Report "Resonance Repo Auto - Daily" "Fix path C:\resonance-uy-py to C:\dev\resonance-uy-py + wrap silent"
        if ($Fix) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument """$SilentVbs"" ""powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\dev\resonance-uy-py\run-auto.ps1"""
            $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
            Set-ScheduledTask -TaskName 'Resonance Repo Auto - Daily' -Action $action -Principal $principal
        }
    }
}

# 7. Remove disabled Resonance Repo Auto - Logon
$task = Get-ScheduledTask -TaskName 'Resonance Repo Auto - Logon' -ErrorAction SilentlyContinue
if ($task -and $task.State -eq 'Disabled') {
    Report "Resonance Repo Auto - Logon" "REMOVE -- disabled, wrong path, redundant with Daily"
    if ($Fix) {
        Unregister-ScheduledTask -TaskName 'Resonance Repo Auto - Logon' -Confirm:$false
    }
}

# 8. Fix NCC-NightlyResearch: python.exe direct (pops window) -> wscript
$task = Get-ScheduledTask -TaskName 'NCC-NightlyResearch' -TaskPath '\NCC\' -ErrorAction SilentlyContinue
if ($task) {
    $exe = $task.Actions[0].Execute
    if ($exe -match 'python') {
        Report "NCC-NightlyResearch" "Wrap with wscript.exe to prevent window popup"
        if ($Fix) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument """$SilentVbs"" ""C:\Users\gripa\AppData\Local\Programs\Python\Python312\python.exe C:\dev\NCC-Doctrine\NCL-RND\rnd_agent.py""" -WorkingDirectory 'C:\dev\NCC-Doctrine'
            Set-ScheduledTask -TaskName 'NCC-NightlyResearch' -TaskPath '\NCC\' -Action $action
        }
    }
}

# 9. Fix NCC-WeeklyAudit under \NCC\: python.exe direct -> wscript
$task = Get-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\NCC\' -ErrorAction SilentlyContinue
if ($task) {
    $exe = $task.Actions[0].Execute
    if ($exe -match 'python') {
        Report "NCC\NCC-WeeklyAudit" "Wrap with wscript.exe to prevent window popup"
        if ($Fix) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument """$SilentVbs"" ""C:\Users\gripa\AppData\Local\Programs\Python\Python312\python.exe C:\dev\NCC-Doctrine\runtime\trinity_loop.py""" -WorkingDirectory 'C:\dev\NCC-Doctrine'
            Set-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\NCC\' -Action $action
        }
    }
}

# Summary
Write-Host ""
Write-Host "=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "Changes: $($Changes.Count)" -ForegroundColor White
if (-not $Fix) {
    Write-Host ""
    Write-Host "Run with -Fix to apply all changes." -ForegroundColor Yellow
}
