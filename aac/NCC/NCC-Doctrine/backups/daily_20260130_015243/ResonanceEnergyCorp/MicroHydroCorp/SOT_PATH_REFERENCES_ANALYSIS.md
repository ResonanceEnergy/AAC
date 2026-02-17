# PATH REFERENCES NEEDING UPDATE

**Consolidation Status:** Ready to execute  
**Files Affected:** 20+ documentation files (analyzed)  
**Active Scripts:** To be determined

---

## 📍 REFERENCES FOUND IN AUDIT DOCUMENTATION

**These are internal audit docs (safe to update or ignore):**

1. **AUDIT_DEDUPLICATION_ROADMAP.md** (6 references)
   - Lines: 15, 26, 40, 66, 72, 390, 401, 403, 448
   - Action: Update for clarity (optional - for documentation consistency)

2. **AUDIT_QUICK_REFERENCE.md** (3 references)
   - Lines: 55, 65, 250
   - Action: Update for clarity (optional)

3. **README_AUDIT_INDEX.md** (1 reference)
   - Line: 133
   - Action: Update for clarity (optional)

4. **MASTER_AUDIT_SUMMARY.md** (2 references)
   - Lines: 239, 243
   - Action: Update for clarity (optional)

5. **00_START_HERE.md** (2 references)
   - Lines: 156, 221
   - Action: Update for clarity (optional)

6. **SOT_CONSOLIDATION_ANALYSIS.md** (3 references)
   - Lines: 15, 26, 276, 381
   - Action: Update for consistency

7. **file_inventory.csv** (10+ references)
   - Rows with SoT paths
   - Action: Can be re-generated after consolidation

---

## 🔴 CRITICAL - PRODUCTION SCRIPTS/CONFIGS TO SEARCH

**Must find and update before consolidation:**

1. **Python Scripts (.py)**
   - Check: automation/ directories
   - Check: MicroHydroV1_Work/tools/
   - Look for: hardcoded path strings

2. **JSON Configuration Files (.json)**
   - Check: automation/Import_Config.json
   - Check: MicroHydroV1_Work/tools/policy_config.json
   - Look for: path references

3. **PowerShell Scripts (.ps1, .command)**
   - Check: Install_MicroHydroV1.command
   - Check: mhv1_oneclick_pipeline.command
   - Check: Any tools/scripts/ files

4. **Batch/Shell Scripts (.bat, .sh, .command)**
   - Check: Install_MicroHydroV1.sh
   - Check: Any automation scripts

5. **Documentation (.md, .docx)**
   - Check: CHANGELOG.md
   - Check: README.md files
   - Check: docs/ directory

---

## 🔍 SEARCH COMMANDS TO EXECUTE

**After backup, run these searches:**

```powershell
# Find all references in Python scripts
Write-Host "Searching Python scripts..."
$pythonRefs = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Filter "*.py" | 
  Select-String -Pattern "MICRO HYDRO NEW|MicroHydroV1 — SoT" -List

if ($pythonRefs) {
    Write-Host "PYTHON FILES WITH REFERENCES:"
    $pythonRefs | ForEach-Object { Write-Host "  - $($_.Path)" }
}

# Find all references in JSON configs
Write-Host "`nSearching JSON configurations..."
$jsonRefs = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Filter "*.json" | 
  Select-String -Pattern "MICRO HYDRO NEW|MicroHydroV1 — SoT|SoT" -List

if ($jsonRefs) {
    Write-Host "JSON FILES WITH REFERENCES:"
    $jsonRefs | ForEach-Object { Write-Host "  - $($_.Path)" }
}

# Find all references in PowerShell scripts
Write-Host "`nSearching PowerShell scripts..."
$psRefs = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Include @("*.ps1", "*.command") | 
  Select-String -Pattern "MICRO HYDRO NEW|MicroHydroV1 — SoT" -List

if ($psRefs) {
    Write-Host "POWERSHELL FILES WITH REFERENCES:"
    $psRefs | ForEach-Object { Write-Host "  - $($_.Path)" }
}

# Find all references in shell scripts
Write-Host "`nSearching shell scripts..."
$shRefs = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Include @("*.sh", "*.bash") | 
  Select-String -Pattern "MICRO HYDRO NEW|MicroHydroV1 — SoT" -List

if ($shRefs) {
    Write-Host "SHELL FILES WITH REFERENCES:"
    $shRefs | ForEach-Object { Write-Host "  - $($_.Path)" }
}
```

---

## 📝 REFERENCE UPDATE TEMPLATE

**When references are found, use this pattern:**

**Old Path Format:**
```
Windows:  c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT
Relative: ..\..\..\..\MICRO HYDRO NEW\MicroHydroV1 — SoT
Unix:     /path/to/MICRO HYDRO NEW/MicroHydroV1 — SoT
```

**New Path Format:**
```
Windows:  c:\MircoHydro\MicroHydroV1 — SoT
Relative: ..\..\..\MicroHydroV1 — SoT
Unix:     /path/to/MicroHydroV1 — SoT
```

**Update Pattern:**
```
Find:       MICRO HYDRO NEW\MicroHydroV1 — SoT\MicroHydroV1
Replace:    MicroHydroV1 — SoT\MicroHydroV1

Find:       MICRO HYDRO NEW/MicroHydroV1 — SoT/MicroHydroV1
Replace:    MicroHydroV1 — SoT/MicroHydroV1

Find:       ..\..\..\..\MICRO HYDRO NEW\
Replace:    ..\..\
```

---

## ✅ READY FOR EXECUTION

**Next Steps:**
1. Create full backup: `c:\MircoHydro_CONSOLIDATION_BACKUP_20260124`
2. Run search commands above to find all active references
3. Update each reference systematically
4. Test updated scripts
5. Execute consolidation plan from SOT_CONSOLIDATION_EXECUTION_PLAN.md

---

**Status:** Analysis Complete - Ready for Team Review  
**Files to Update:** TBD pending search results  
**Estimated Update Time:** 30-60 minutes

