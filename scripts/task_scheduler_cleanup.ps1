<#
.SYNOPSIS
    AAC/NCC Task Scheduler Audit & Cleanup
.DESCRIPTION
    Fixes all issues found in the 2026-03-23 audit:
      1. AAC tasks use cmd.exe (pops windows) → switch to wscript.exe silent wrapper
      2. NCC-Doctrine Weekly Full Backup is identical to Daily Backup → remove
      3. Root NCC-WeeklyAudit references non-existent trinity_audit.py → remove (duplicate of \NCC\NCC-WeeklyAudit)
      4. AAC_Startup_OnLogon disabled + crashed → remove (NCC-MasterLauncher already handles logon)
      5. Resonance Repo Auto tasks point to C:\resonance-uy-py (wrong) → fix to C:\dev\resonance-uy-py
      6. Resonance Repo Auto - Logon disabled → remove
      7. NCC-NightlyResearch and \NCC\NCC-WeeklyAudit run python.exe directly (pop window) → wrap with wscript
      8. AAC_Automated_Pipeline has PT10M timeout but automate.py takes ~3 min → bump to PT30M
      9. AAC_RocketShip_DailyBriefing never ran (267011) — verify target exists
    
    All changes are logged. Run with -WhatIf to preview, -Fix to apply.
.PARAMETER WhatIf
    Preview changes without applying them.
.PARAMETER Fix
    Apply all fixes.
#>
param(
    [switch]$Fix,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Continue'
$SilentVbs = 'C:\dev\silent-launch.vbs'
$Changes = @()

function Report($action, $detail) {
    $tag = if ($Fix -and -not $WhatIf) { "[FIX]" } else { "[PREVIEW]" }
    $msg = "$tag $action - $detail"
    Write-Host $msg -ForegroundColor $(if ($Fix) { 'Green' } else { 'Yellow' })
    $script:Changes += $msg
}

Write-Host "`n=== AAC/NCC Task Scheduler Cleanup ===" -ForegroundColor Cyan
Write-Host "Mode: $(if ($Fix -and -not $WhatIf) { 'APPLYING FIXES' } else { 'PREVIEW ONLY (use -Fix to apply)' })`n"

# ── 1. Fix AAC_Automated_Pipeline: cmd.exe → wscript silent, extend timeout ──
$task = Get-ScheduledTask -TaskName 'AAC_Automated_Pipeline' -ErrorAction SilentlyContinue
if ($task) {
    if ($task.Actions[0].Execute -eq 'cmd.exe') {
        Report "AAC_Automated_Pipeline" "Switch from cmd.exe to wscript.exe silent wrapper + extend timeout to 30 min"
        if ($Fix -and -not $WhatIf) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "`"$SilentVbs`" `"C:\dev\AAC_fresh\.venv\Scripts\python.exe C:\dev\AAC_fresh\automate.py --pipeline`""
            $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -Priority 7
            Set-ScheduledTask -TaskName 'AAC_Automated_Pipeline' -Action $action -Settings $settings
        }
    }
} else { Write-Host "[SKIP] AAC_Automated_Pipeline not found" -ForegroundColor DarkGray }

# ── 2. Fix AAC_RocketShip_DailyBriefing: cmd.exe → wscript silent ──
$task = Get-ScheduledTask -TaskName 'AAC_RocketShip_DailyBriefing' -ErrorAction SilentlyContinue
if ($task) {
    if ($task.Actions[0].Execute -eq 'cmd.exe') {
        Report "AAC_RocketShip_DailyBriefing" "Switch from cmd.exe to wscript.exe silent wrapper"
        if ($Fix -and -not $WhatIf) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "`"$SilentVbs`" `"C:\dev\AAC_fresh\.venv\Scripts\python.exe -m strategies.rocket_ship.daily_ops --once`""
            Set-ScheduledTask -TaskName 'AAC_RocketShip_DailyBriefing' -Action $action
        }
    }
} else { Write-Host "[SKIP] AAC_RocketShip_DailyBriefing not found" -ForegroundColor DarkGray }

# ── 3. Remove redundant NCC-Doctrine Weekly Full Backup (identical to Daily) ──
$task = Get-ScheduledTask -TaskName 'NCC-Doctrine Weekly Full Backup' -ErrorAction SilentlyContinue
if ($task) {
    Report "NCC-Doctrine Weekly Full Backup" "REMOVE — identical command to Daily Backup (both -RunDataBackup)"
    if ($Fix -and -not $WhatIf) {
        Unregister-ScheduledTask -TaskName 'NCC-Doctrine Weekly Full Backup' -Confirm:$false
    }
}

# ── 4. Remove root NCC-WeeklyAudit (references non-existent trinity_audit.py) ──
$task = Get-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\' -ErrorAction SilentlyContinue
if ($task) {
    Report "Root \NCC-WeeklyAudit" "REMOVE — references non-existent runtime/trinity_audit.py; \NCC\NCC-WeeklyAudit is the real one"
    if ($Fix -and -not $WhatIf) {
        Unregister-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\' -Confirm:$false
    }
}

# ── 5. Remove disabled AAC_Startup_OnLogon (NCC-MasterLauncher handles logon) ──
$task = Get-ScheduledTask -TaskName 'AAC_Startup_OnLogon' -ErrorAction SilentlyContinue
if ($task) {
    Report "AAC_Startup_OnLogon" "REMOVE — disabled, last run crashed (0x8007042B), NCC-MasterLauncher already handles logon"
    if ($Fix -and -not $WhatIf) {
        Unregister-ScheduledTask -TaskName 'AAC_Startup_OnLogon' -Confirm:$false
    }
}

# ── 6. Fix Resonance Repo Auto - Daily: wrong path ──
$task = Get-ScheduledTask -TaskName 'Resonance Repo Auto - Daily' -ErrorAction SilentlyContinue
if ($task) {
    $args0 = $task.Actions[0].Arguments
    if ($args0 -match 'C:\\resonance-uy-py') {
        Report "Resonance Repo Auto - Daily" "Fix path C:\resonance-uy-py → C:\dev\resonance-uy-py + wrap with wscript"
        if ($Fix -and -not $WhatIf) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "`"$SilentVbs`" `"powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\dev\resonance-uy-py\run-auto.ps1`""
            $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
            Set-ScheduledTask -TaskName 'Resonance Repo Auto - Daily' -Action $action -Principal $principal
        }
    }
}

# ── 7. Remove disabled Resonance Repo Auto - Logon ──
$task = Get-ScheduledTask -TaskName 'Resonance Repo Auto - Logon' -ErrorAction SilentlyContinue
if ($task -and $task.State -eq 'Disabled') {
    Report "Resonance Repo Auto - Logon" "REMOVE — disabled, path wrong, redundant with Daily"
    if ($Fix -and -not $WhatIf) {
        Unregister-ScheduledTask -TaskName 'Resonance Repo Auto - Logon' -Confirm:$false
    }
}

# ── 8. Fix NCC-NightlyResearch: python.exe direct (pops window) → wscript wrapper ──
$task = Get-ScheduledTask -TaskName 'NCC-NightlyResearch' -TaskPath '\NCC\' -ErrorAction SilentlyContinue
if ($task) {
    $exe = $task.Actions[0].Execute
    if ($exe -match 'python') {
        Report "\NCC\NCC-NightlyResearch" "Wrap with wscript.exe to prevent window popup"
        if ($Fix -and -not $WhatIf) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "`"$SilentVbs`" `"C:\Users\gripa\AppData\Local\Programs\Python\Python312\python.exe C:\dev\NCC-Doctrine\NCL-RND\rnd_agent.py`"" -WorkingDirectory 'C:\dev\NCC-Doctrine'
            Set-ScheduledTask -TaskName 'NCC-NightlyResearch' -TaskPath '\NCC\' -Action $action
        }
    }
}

# ── 9. Fix \NCC\NCC-WeeklyAudit: python.exe direct (pops window) → wscript wrapper ──
$task = Get-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\NCC\' -ErrorAction SilentlyContinue
if ($task) {
    $exe = $task.Actions[0].Execute
    if ($exe -match 'python') {
        Report "\NCC\NCC-WeeklyAudit" "Wrap with wscript.exe to prevent window popup"
        if ($Fix -and -not $WhatIf) {
            $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "`"$SilentVbs`" `"C:\Users\gripa\AppData\Local\Programs\Python\Python312\python.exe C:\dev\NCC-Doctrine\runtime\trinity_loop.py`"" -WorkingDirectory 'C:\dev\NCC-Doctrine'
            Set-ScheduledTask -TaskName 'NCC-WeeklyAudit' -TaskPath '\NCC\' -Action $action
        }
    }
}

# ── Summary ──
Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "Changes: $($Changes.Count)" -ForegroundColor White
if (-not $Fix -or $WhatIf) {
    Write-Host "`nRun with -Fix to apply all changes." -ForegroundColor Yellow
}
