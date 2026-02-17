# SoT CONSOLIDATION - DETAILED MERGE PLAN

**Analysis Date:** January 24, 2026  
**Status:** Ready for Implementation  
**Priority:** 🔴 CRITICAL

---

## EXECUTIVE DECISION

**Recommendation:** Consolidate into Location B (MicroHydroV1 — SoT)

**Reason:**
- Location B has MANIFEST.json (version control)
- Location B is top-level (better accessibility)
- Location B has comprehensive addon packages
- Location A can become archive backup

---

## 📊 STRUCTURE COMPARISON

### Location A: MICRO HYDRO NEW/MicroHydroV1 — SoT/

**Tier 1 Contents:**
```
MicroHydroV1 — SoT/
├── MicroHydroV1/                                    [CORE PROJECT]
├── MicroHydroV1_BACKUP_20260122_005746/           [TIMESTAMPED BACKUP] ⭐
├── MicroHydroV1_FastTrack_Bundle.zip
├── MicroHydroV1_Phase3_Automation_Pack_macOS_OpenReleases.zip
├── MicroHydroV1_Phase4_SmartRuns_Pack_macOS.zip
├── MicroHydroV1_RC1.zip
├── MicroHydroV1_RunID_Migrator_v0.3.0.zip
└── MicroHydroV1_SoT_Package.zip
```

**Location A Advantages:**
- ✓ Has timestamped backup from 2026-01-22
- ✓ Has release packages (Phase 3, 4, RC1)
- ✓ Established directory hierarchy

**Location A Disadvantages:**
- ✗ No MANIFEST.json (version tracking)
- ✗ Limited addon package organization
- ✗ Nested deep in folder structure

---

### Location B: MicroHydroV1 — SoT/

**Tier 1 Contents:**
```
MicroHydroV1 — SoT/
└── MicroHydroV1/                                   [CORE PROJECT]
    ├── automation/
    ├── cad/
    ├── docs/
    ├── tests/
    ├── tools/
    ├── README.md
    │
    ├── MANIFEST.json                               [VERSION CONTROL] ⭐
    │
    ├── ADDON PACKAGES (Directories):
    ├── MicroHydroV1_Importer_RunID_Patch_v0.3.0/
    ├── MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0/
    ├── MicroHydroV1_RUN3_AutoFill_Evidence_RUNID_BASELINE_v0.3.0/
    ├── MicroHydroV1_RunID_Migrator_v0.3.0/
    │
    ├── ADDON PACKAGES (Zip Archives - 13 files):
    ├── MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0.zip
    ├── MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0 (1).zip    [DUP]
    ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0.zip
    ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0 (1).zip                   [DUP]
    ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0 (2).zip                   [DUP]
    ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0 (3).zip                   [DUP]
    ├── MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_T002_v0.3.0.zip
    ├── MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0.zip
    ├── MicroHydroV1_RUN3_AutoFill_Evidence_v0.3.0.zip
    ├── MicroHydroV1_RUN3_DoAll_Templates_Plots.zip
    ├── MicroHydroV1_RUN3_DoAll_Templates_Plots (1).zip            [DUP]
    ├── MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa.zip
    └── MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa (1).zip        [DUP]
    
    └── MicroHydroV1_SharePoint_Version_Control_Configuration_Guide_v1.0.docx
```

**Location B Advantages:**
- ✓ HAS MANIFEST.json (version control)
- ✓ Addon packages well organized
- ✓ Top-level location (better accessibility)
- ✓ SharePoint integration documentation
- ✓ More comprehensive tooling

**Location B Disadvantages:**
- ✗ Duplicate addon packages (multiple (1).zip, (2).zip files)
- ✗ No timestamped backup (has archives instead)
- ✗ More files to manage

---

## 🎯 CONSOLIDATION PLAN - 3 PHASES

### PHASE 1: Pre-Consolidation Analysis (Complete ✓)

**Completed:**
- [x] Identified both SoT locations
- [x] Analyzed structure differences
- [x] Found critical differences (MANIFEST, addons)
- [x] Documented advantages/disadvantages

**Finding:** Location B is the ENHANCED/AUTHORITATIVE version.

---

### PHASE 2: Preparation & Verification

**Step 1: Create Pre-Consolidation Backup**
```powershell
# Create full backup before any changes
Copy-Item -Path "c:\MircoHydro" `
          -Destination "c:\MircoHydro_PRE_CONSOLIDATION_20260124" `
          -Recurse -Force
```

**Step 2: Analyze MANIFEST.json**
```
File: c:\MircoHydro\MicroHydroV1 — SoT\MicroHydroV1\MANIFEST.json
Action: Read and document version information
Purpose: Understand versioning scheme
```

**Step 3: Identify Duplicate Addons**
```
Current Duplicates in Location B:
- MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0.zip
- MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0 (1).zip  ← DELETE

- MicroHydroV1_DO_ALL_AddOn_v0.3.0.zip
- MicroHydroV1_DO_ALL_AddOn_v0.3.0 (1).zip                 ← DELETE
- MicroHydroV1_DO_ALL_AddOn_v0.3.0 (2).zip                 ← DELETE
- MicroHydroV1_DO_ALL_AddOn_v0.3.0 (3).zip                 ← DELETE

- MicroHydroV1_RUN3_DoAll_Templates_Plots.zip
- MicroHydroV1_RUN3_DoAll_Templates_Plots (1).zip          ← DELETE

- MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa.zip
- MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa (1).zip       ← DELETE
```

**Step 4: Check for Script References**
```
Search entire workspace for:
- References to "MICRO HYDRO NEW\MicroHydroV1 — SoT"
- References to Location A's path
- Update scripts/configs to use new location

Commands to find references:
grep -r "MICRO HYDRO NEW.*MicroHydroV1.*SoT" c:\MircoHydro
grep -r "MicroHydroV1.*SoT" c:\MircoHydro --include="*.py"
grep -r "MicroHydroV1.*SoT" c:\MircoHydro --include="*.json"
```

---

### PHASE 3: Consolidation Execution

**Decision Point: Which structure to keep?**

**OPTION 1: RECOMMENDED - Keep Location B, Archive Location A**

```
KEEP: c:\MircoHydro\MicroHydroV1 — SoT\  ← Primary SoT
├── MicroHydroV1/                         ← Core project
├── MicroHydroV1_Importer_RunID_Patch_v0.3.0/    ← Keep all addons
├── MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0/
├── MicroHydroV1_RUN3_AutoFill_Evidence_RUNID_BASELINE_v0.3.0/
├── MicroHydroV1_RunID_Migrator_v0.3.0/
├── [Cleaned addon zips - no (1), (2), (3)]
└── MANIFEST.json

ARCHIVE: c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1_Archive_Backup\
└── Keep ONLY:
    └── MicroHydroV1_BACKUP_20260122_005746/  ← Timestamped backup
```

**OPTION 2: Alternative - Move everything to Location A, delete Location B**

```
KEEP: c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT\
└── Copy from Location B:
    ├── MANIFEST.json
    ├── All addon packages
    ├── SharePoint config

DELETE: c:\MircoHydro\MicroHydroV1 — SoT\  ← Remove entirely
```

**RECOMMENDATION: OPTION 1**

**Justification:**
- Location B is already at top level (better)
- Maintains all Location A's backup capability
- Simpler directory cleanup
- MANIFEST.json already in place
- Fewer file operations to perform

---

## 📋 CONSOLIDATION EXECUTION STEPS

### Step 1: Full Backup (Safety)
```powershell
# Back up entire workspace
Copy-Item -Path "c:\MircoHydro" `
          -Destination "c:\MircoHydro_CONSOLIDATION_BACKUP_20260124" `
          -Recurse -Force

Write-Host "Backup complete: c:\MircoHydro_CONSOLIDATION_BACKUP_20260124"
```

### Step 2: Clean Duplicate Addons from Location B
```powershell
$lotB = "c:\MircoHydro\MicroHydroV1 — SoT\MicroHydroV1"

# Delete duplicate addon zips
Remove-Item "$sotB\MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0 (1).zip" -Force
Remove-Item "$sotB\MicroHydroV1_DO_ALL_AddOn_v0.3.0 (1).zip" -Force
Remove-Item "$sotB\MicroHydroV1_DO_ALL_AddOn_v0.3.0 (2).zip" -Force
Remove-Item "$sotB\MicroHydroV1_DO_ALL_AddOn_v0.3.0 (3).zip" -Force
Remove-Item "$sotB\MicroHydroV1_RUN3_DoAll_Templates_Plots (1).zip" -Force
Remove-Item "$sotB\MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa (1).zip" -Force

Write-Host "Removed 6 duplicate addon files from Location B"
```

### Step 3: Copy Location A's Backup to Location B (Optional)
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

### Step 4: Archive Location A
```powershell
$sotA_parent = "c:\MircoHydro\MICRO HYDRO NEW"
$archivePath = "$sotA_parent\MicroHydroV1_Archive_SoT_Backup_20260124"

# Move Location A to archive folder
if (Test-Path "$sotA_parent\MicroHydroV1 — SoT") {
    Rename-Item -Path "$sotA_parent\MicroHydroV1 — SoT" `
                -NewName "MicroHydroV1 — SoT_ARCHIVED_20260124"
    Write-Host "Archived Location A"
}
```

### Step 5: Update References in Scripts/Config
```
Search for files that reference Location A path:
- Python scripts (*.py)
- JSON configs (*.json)
- Batch/PowerShell scripts (*.bat, *.ps1)
- Documentation (*.md)

Update paths from:
"c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT"
To:
"c:\MircoHydro\MicroHydroV1 — SoT"

Updated from:
"MICRO HYDRO NEW/MicroHydroV1 — SoT"
To:
"MicroHydroV1 — SoT"
```

### Step 6: Verification
```powershell
# Test new Location B paths
$newSoT = "c:\MircoHydro\MicroHydroV1 — SoT\MicroHydroV1"

Write-Host "Verifying new Location B:"
Write-Host "✓ Core dirs: $((Test-Path "$newSoT\automation") -and (Test-Path "$newSoT\cad") -and (Test-Path "$newSoT\docs") -and (Test-Path "$newSoT\tests") -and (Test-Path "$newSoT\tools"))"
Write-Host "✓ MANIFEST: $(Test-Path "$newSoT\MANIFEST.json")"
Write-Host "✓ Addons: $((Get-ChildItem -Path "$newSoT" -Directory | Where-Object {$_.Name -like "*AddOn*" -or $_.Name -like "*Patch*"}).Count) addon dirs"
```

### Step 7: Test Pipeline
```
1. Run: mhv1_pipeline.py from MicroHydroV1_Work
2. Verify all imports work
3. Check test execution
4. Validate output generation
```

---

## ✅ FINAL CONSOLIDATION STATE

**After successful consolidation:**

```
c:\MircoHydro\
├── MICRO HYDRO NEW/
│   ├── Installation scripts
│   ├── MICRO HYDRO - OLD/        [Legacy]
│   ├── MicroHydroV1/             [Active Dev]
│   └── MicroHydroV1 — SoT_ARCHIVED_20260124/  [Old Location - archived]
│
├── MicroHydroV1 — SoT/           [PRIMARY SoT ⭐]
│   └── MicroHydroV1/
│       ├── automation/
│       ├── cad/
│       ├── docs/
│       ├── tests/
│       ├── tools/
│       ├── README.md
│       ├── MANIFEST.json        [VERSION CONTROL]
│       ├── MicroHydroV1_Backup_20260122_005746/   [BACKUP]
│       ├── MicroHydroV1_Importer_RunID_Patch_v0.3.0/
│       ├── MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0/
│       ├── MicroHydroV1_RUN3_AutoFill_Evidence_RUNID_BASELINE_v0.3.0/
│       ├── MicroHydroV1_RunID_Migrator_v0.3.0/
│       ├── [Cleaned addon zips - no duplicates]
│       └── MicroHydroV1_SharePoint_Version_Control_Configuration_Guide_v1.0.docx
│
├── MicroHydroV1_Work/            [Unchanged]
│
└── MicroHydroV1 — SoT_Archive/   [Safe copy, point-in-time]
```

---

## 🎯 BENEFITS AFTER CONSOLIDATION

✅ **Single Authoritative SoT**
- Clear which version is current
- MANIFEST.json maintained
- Reduced confusion

✅ **Cleaner Structure**
- One SoT location (top-level)
- Old Location archived (audit trail)
- Duplicate addons removed

✅ **Better Version Control**
- MANIFEST.json always at: `c:\MircoHydro\MicroHydroV1 — SoT\MicroHydroV1\`
- Scripts can reliably reference it
- Deployment stages easier to manage

✅ **Storage Optimization**
- Removed 6 duplicate addon zips (~30-50 MB)
- No redundant copies of core project
- Archive is separate, can compress

✅ **Reduced Redundancy**
- 289 duplicate files → focus on deduplication
- MicroHydroV1_Work can now use symlinks
- Better path navigation

---

## 📊 RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| Script breakage | Search & update all references before consolidation |
| Data loss | Full backups at 3 points: original, pre-consolidation, archive |
| Pipeline failure | Test in staging with Location B before deploying |
| MANIFEST corruption | Keep backup of MANIFEST from Location B |
| Reference issues | Update all paths systematically with script |

---

## 📋 PRE-CONSOLIDATION CHECKLIST

Before executing consolidation:

- [ ] Read this entire plan
- [ ] Create backup: `c:\MircoHydro_CONSOLIDATION_BACKUP_20260124`
- [ ] Find all references to Location A path
- [ ] Create script to update all paths
- [ ] Test script on backup first
- [ ] Read MANIFEST.json in Location B
- [ ] Document any differences between core projects
- [ ] Get team sign-off on this plan
- [ ] Plan rollback procedure
- [ ] Schedule maintenance window
- [ ] Notify users of planned consolidation

---

## 🚀 NEXT STEP

**Proceed to:**
1. Create full backup
2. Search for all references to Location A
3. Create path update script
4. Execute consolidation with team oversight

---

**Status:** ✅ Plan Ready for Implementation  
**Owner:** Development Team  
**Timeline:** 1-2 hours execution + testing  
**Risk Level:** LOW (with backups in place)

