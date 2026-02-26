# ✅ SoT CONSOLIDATION - EXECUTION CHECKLIST

**Use this during consolidation to track progress**

---

## 🔵 PRE-CONSOLIDATION CHECKLIST

Complete these BEFORE starting any consolidation work:

### Understanding Phase
- [ ] Read [SOT_CONSOLIDATION_SUMMARY.md](SOT_CONSOLIDATION_SUMMARY.md)
- [ ] Read [SOT_CONSOLIDATION_EXECUTION_PLAN.md](SOT_CONSOLIDATION_EXECUTION_PLAN.md)
- [ ] Understand the consolidation plan (Location A → Archive, Location B → Primary)
- [ ] Understand risk mitigation (3x backups)
- [ ] Got team approval to proceed

### Discovery Phase
- [ ] Run searches from [SOT_PATH_REFERENCES_ANALYSIS.md](SOT_PATH_REFERENCES_ANALYSIS.md)
- [ ] Found all Python scripts with old references: `___________`
- [ ] Found all JSON configs with old references: `___________`
- [ ] Found all PowerShell scripts with old references: `___________`
- [ ] Documented all files needing updates (list attached)

### Backup Phase
- [ ] Created full workspace backup: `c:\MircoHydro_CONSOLIDATION_BACKUP_20260124`
- [ ] Verified backup size > 200MB (reasonable copy of workspace)
- [ ] Backup location accessible and safe
- [ ] Tested backup restore (optional but recommended)
- [ ] No errors in backup process

### Approval Phase
- [ ] Team lead approved consolidation
- [ ] Scheduled 2-3 hour maintenance window
- [ ] Notified team of planned maintenance
- [ ] Identified rollback person (who can restore backup if needed)
- [ ] Set communication channel for status updates

---

## 🟢 CONSOLIDATION EXECUTION CHECKLIST

Follow these steps IN ORDER during consolidation:

### Phase 1: Preparation Complete ✓
- [ ] All pre-consolidation items completed
- [ ] Ready to start Phase 2
- [ ] Time: ~30 minutes completed

### Phase 2: Clean Location B

**Step 1: Remove Duplicate Addons**
```powershell
$sotB = "c:\MircoHydro\MicroHydroV1 — SoT\MicroHydroV1"

# Delete duplicate addon zips
Remove-Item "$sotB\MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0 (1).zip" -Force
Remove-Item "$sotB\MicroHydroV1_DO_ALL_AddOn_v0.3.0 (1).zip" -Force
Remove-Item "$sotB\MicroHydroV1_DO_ALL_AddOn_v0.3.0 (2).zip" -Force
Remove-Item "$sotB\MicroHydroV1_DO_ALL_AddOn_v0.3.0 (3).zip" -Force
Remove-Item "$sotB\MicroHydroV1_RUN3_DoAll_Templates_Plots (1).zip" -Force
Remove-Item "$sotB\MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa (1).zip" -Force

Write-Host "Removed 6 duplicate addon files from Location B"
```

- [ ] Executed PowerShell step 1 (remove duplicates)
- [ ] Verified no errors
- [ ] Confirmed 6 files deleted
- [ ] Location B now clean

**Step 2: Copy Location A's Backup (if desired)**
```powershell
$sotA = "c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT"
$sotB = "c:\MircoHydro\MicroHydroV1 — SoT"

# Copy backup if it doesn't exist in B
if (Test-Path "$sotA\MicroHydroV1_BACKUP_20260122_005746") {
    if (-not (Test-Path "$sotB\MicroHydroV1_BACKUP_20260122_005746")) {
        Copy-Item -Path "$sotA\MicroHydroV1_BACKUP_20260122_005746" `
                  -Destination "$sotB\" -Recurse -Force
        Write-Host "Copied backup from Location A to Location B"
    }
}
```

- [ ] Executed PowerShell step 2 (copy backup)
- [ ] Verified backup copied (or already exists)
- [ ] No errors in copy process
- [ ] Location B now has backup from Location A

### Phase 3: Archive Location A

**Step 3: Archive Old Location A**
```powershell
$sotA_parent = "c:\MircoHydro\MICRO HYDRO NEW"

# Archive Location A
if (Test-Path "$sotA_parent\MicroHydroV1 — SoT") {
    Rename-Item -Path "$sotA_parent\MicroHydroV1 — SoT" `
                -NewName "MicroHydroV1 — SoT_ARCHIVED_20260124"
    Write-Host "Archived Location A as: MicroHydroV1 — SoT_ARCHIVED_20260124"
}
```

- [ ] Executed PowerShell step 3 (archive Location A)
- [ ] Verified old location renamed
- [ ] Archive folder visible: `MicroHydroV1 — SoT_ARCHIVED_20260124`
- [ ] Original Location A no longer active
- [ ] No errors in rename process

### Phase 4: Update Path References

**Step 4a: Create Update Script (or do manually)**

For each file found in reference search:

```powershell
# EXAMPLE: Update Python script
$filePath = "c:\MircoHydro\automation\import_measurements.py"
$content = Get-Content $filePath -Raw

# Replace old path with new path
$newContent = $content -replace 'MICRO HYDRO NEW\\MicroHydroV1 — SoT', 'MicroHydroV1 — SoT'

Set-Content -Path $filePath -Value $newContent
Write-Host "Updated: $filePath"
```

- [ ] Found all files with old references
- [ ] Created update script or manual update list
- [ ] Updated file #1: `___________`
- [ ] Updated file #2: `___________`
- [ ] Updated file #3: `___________`
- [ ] Updated file #4: `___________`
- [ ] (Add more as needed)
- [ ] All reference files updated
- [ ] No files missed

**Step 4b: Verify Updates**
```powershell
# Search for any remaining old references
$oldPath = "MICRO HYDRO NEW.*MicroHydroV1 — SoT"
$found = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Include *.py, *.json, *.ps1, *.sh | 
  Select-String -Pattern $oldPath

if ($found) {
    Write-Host "FOUND REMAINING REFERENCES:"
    $found
} else {
    Write-Host "✓ No old references remaining!"
}
```

- [ ] Executed verification search
- [ ] No old references found
- [ ] All paths updated correctly
- [ ] Ready for Phase 5

### Phase 5: Verification

**Step 5: Test New Location Structure**
```powershell
$newSoT = "c:\MircoHydro\MicroHydroV1 — SoT\MicroHydroV1"

Write-Host "Verifying new Location B structure:"
Write-Host "✓ Core dirs present: " + 
  (
    (Test-Path "$newSoT\automation") -and 
    (Test-Path "$newSoT\cad") -and 
    (Test-Path "$newSoT\docs") -and 
    (Test-Path "$newSoT\tests") -and 
    (Test-Path "$newSoT\tools")
  )
Write-Host "✓ MANIFEST.json present: " + (Test-Path "$newSoT\MANIFEST.json")
Write-Host "✓ Backup present: " + (Test-Path "$newSoT\MicroHydroV1_BACKUP_20260122_005746")

$addonDirs = Get-ChildItem -Path "$newSoT" -Directory | 
  Where-Object {$_.Name -like "*Importer*" -or $_.Name -like "*AutoFill*" -or $_.Name -like "*Migrator*"}
Write-Host "✓ Addon packages present: $($addonDirs.Count)"
```

- [ ] Executed verification script
- [ ] All core directories present
- [ ] MANIFEST.json present
- [ ] Backup present
- [ ] Addon packages present
- [ ] No verification errors
- [ ] Structure looks correct

**Step 6: Test Pipeline**
```powershell
# Navigate to workspace and test pipeline
cd "c:\MircoHydro\MicroHydroV1_Work"
python tools\microhydro_master_integrator.py

# Or run main pipeline:
cd "c:\MircoHydro"
python mhv1_pipeline.py --test
```

- [ ] Changed to project directory
- [ ] Executed pipeline successfully
- [ ] No path-related errors
- [ ] Pipeline produces expected output
- [ ] All imports work correctly
- [ ] SoT paths resolve correctly

### Phase 6: Final Checks

**Step 7: Document Completion**
- [ ] Consolidation executed successfully
- [ ] All references updated
- [ ] Location B is now primary SoT
- [ ] Location A archived in MICRO HYDRO NEW folder
- [ ] Old paths no longer used
- [ ] Pipeline tested and working
- [ ] Team notified of successful consolidation
- [ ] Backup preserved for rollback (if needed)

---

## 🟡 POTENTIAL ISSUES & SOLUTIONS

### Issue: PowerShell script fails to delete files
**Solution:**
- [ ] Check if files are locked (close any open editors)
- [ ] Run PowerShell as Administrator
- [ ] Verify exact file names (check for special characters)
- [ ] Try manual deletion via File Explorer first

### Issue: Files with old references not found
**Solution:**
- [ ] Run broader search: `grep -r "SoT" c:\MircoHydro`
- [ ] Check for escaped paths: `\\` vs `\`
- [ ] Check for relative paths: `..\..\` patterns
- [ ] Search in ALL file types, not just .py and .json

### Issue: Pipeline fails after consolidation
**Solution:**
- [ ] Check if MANIFEST.json accessible
- [ ] Verify core directories have all files
- [ ] Check for hardcoded absolute paths in scripts
- [ ] Restore from backup and try again
- [ ] Check error message for specific path issue

### Issue: Verification script doesn't complete
**Solution:**
- [ ] Run simplified version (one check at a time)
- [ ] Check if paths have special characters
- [ ] Use `-Verbose` flag for more info
- [ ] Try with escaped quotes: `"c:\MircoHydro\MicroHydroV1 — SoT"`

### Issue: Can't rename Location A folder
**Solution:**
- [ ] Ensure no files open in Location A
- [ ] Check file permissions
- [ ] Try cutting and pasting instead of rename
- [ ] Close all explorer windows first
- [ ] Run PowerShell as Administrator

---

## 🔴 ROLLBACK PROCEDURE

**If something goes wrong:**

```powershell
# Step 1: Stop any running processes
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Stop-Process -Name mhv1_pipeline -Force -ErrorAction SilentlyContinue

# Step 2: Restore from backup
Copy-Item -Path "c:\MircoHydro_CONSOLIDATION_BACKUP_20260124" `
          -Destination "c:\MircoHydro_ROLLBACK_TEMP" -Recurse -Force

# Step 3: Remove consolidated location (optional - keep for analysis)
# Remove-Item -Path "c:\MircoHydro" -Recurse -Force

# Step 4: Restore backup
Copy-Item -Path "c:\MircoHydro_ROLLBACK_TEMP\*" `
          -Destination "c:\MircoHydro" -Recurse -Force

Write-Host "Rollback complete! Workspace restored to pre-consolidation state."
```

- [ ] Understand rollback procedure before starting
- [ ] Know where backup is located: `c:\MircoHydro_CONSOLIDATION_BACKUP_20260124`
- [ ] Have restore PowerShell script ready (above)
- [ ] Team member ready to execute rollback if needed

---

## ✅ POST-CONSOLIDATION CHECKLIST

After successful consolidation:

- [ ] Location B is the primary SoT: `c:\MircoHydro\MicroHydroV1 — SoT\`
- [ ] Location A archived: `c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT_ARCHIVED_20260124\`
- [ ] All path references updated
- [ ] Pipeline tested and working
- [ ] Team notified of success
- [ ] Backup preserved at: `c:\MircoHydro_CONSOLIDATION_BACKUP_20260124`
- [ ] Archive folder verified (can be deleted after 30 days if desired)
- [ ] Documentation updated with new paths
- [ ] Team trained on new SoT location

---

## 📊 TIMING TRACKER

| Phase | Task | Est. Time | Actual Time | Status |
|-------|------|-----------|-------------|--------|
| Pre-Check | Understanding & approval | 20 min | _____ | □ |
| Discovery | Find references | 20 min | _____ | □ |
| Backup | Create backup | 30 min | _____ | □ |
| Clean | Remove duplicates | 5 min | _____ | □ |
| Archive | Archive Location A | 5 min | _____ | □ |
| Update | Update path references | 20 min | _____ | □ |
| Verify | Verify structure & test | 15 min | _____ | □ |
| **TOTAL** | | **~2 hours** | _____ | |

---

## 👥 TEAM ROLES DURING CONSOLIDATION

**Assign these roles:**

- **Lead:** `________________` (Overall oversight)
- **Executor:** `________________` (Runs PowerShell scripts)
- **Verifier:** `________________` (Tests results)
- **Backup Handler:** `________________` (Manages rollback if needed)
- **Communication:** `________________` (Updates team on progress)

---

## 📞 EMERGENCY CONTACTS

**If consolidation fails:**

- **Technical Issue:** Contact executor or lead
- **Rollback Needed:** Contact backup handler (execute rollback script)
- **Questions on Plan:** Review [SOT_CONSOLIDATION_EXECUTION_PLAN.md](SOT_CONSOLIDATION_EXECUTION_PLAN.md)
- **Need Approval:** Contact project manager

---

**Print this checklist and use it during the consolidation process.**

**Status:** Ready to Use  
**Created:** January 24, 2026  
**Version:** 1.0 Final

