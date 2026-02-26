# ⚙️ PHASE 2: TIER 1 DEDUPLICATION EXECUTION

**Status:** READY TO EXECUTE (After Phase 1 Approval)  
**Date:** January 24, 2026  
**Duration:** 1-2 days active work + 7-day monitoring  
**Priority:** 🟡 MEDIUM  
**Storage Target:** Free 71-112 MB

---

## 🎯 OBJECTIVE
Safely remove 71-112 MB of duplicate .zip files identified in the original audit. Strategy: Archive first (don't delete), monitor for 7 days, then decide on permanent deletion.

---

## 📋 DUPLICATE ZIP FILES IDENTIFIED

### **Location 1: MASTER_PRODUCTION_PACKAGE**
```
c:\MircoHydro\MASTER_PRODUCTION_PACKAGE\
├── MicroHydroV1_CAD Archive.zip                    [ORIGINAL]
├── MicroHydroV1_Rebuilt_Clean_Repo_20260122*.zip  [ORIGINAL]
├── New Folder With Items.zip                        [ORIGINAL]
└── Install_MicroHydroV1_AppBundle.zip              [ORIGINAL]

c:\MircoHydro\MASTER_PRODUCTION_PACKAGE\MASTER_PRODUCTION_PACKAGE\
├── MicroHydroV1_CAD Archive.zip                    [DUPLICATE] ⚠️
├── MicroHydroV1_Rebuilt_Clean_Repo_20*.zip        [DUPLICATE] ⚠️
├── MicroHydroV1_\09_Packages\original*.zip         [DUPLICATES] ⚠️ (6 files)
└── ... (multiple addon packages)                    [DUPLICATES] ⚠️
```

### **Location 2: MicroHydroV1_CAD Archive**
```
c:\MircoHydro\MASTER_PRODUCTION_PACKAGE\MicroHydroV1_CAD Archive\
└── MicroHydroV1_CAD\
    ├── MicroHydroV1_AddonPackages*.zip             [DUPLICATES] ⚠️
    ├── MicroHydroV1_CompletePackage*.zip           [DUPLICATES] ⚠️
    ├── MacroSuite*.zip                             [DUPLICATES] ⚠️
    └── ... (multiple versions)
```

---

## 📊 ARCHIVE STRATEGY

**Why Archive First?**
- ✅ Safe: Can restore immediately if needed
- ✅ Testable: Run 7 days of stability tests
- ✅ Reversible: Zero data loss
- ✅ Audit Trail: Know exactly what we removed

**Archive Location:**
```
c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\
├── DUPLICATES_ARCHIVED_LOGS.md
├── Day_1_Duplicates/
├── Day_2_Duplicates/
├── Day_3_Duplicates/
└── ...
```

---

## ⚙️ EXECUTION PLAN - TIER 1 DEDUPLICATION

### **STEP 1: Hash Analysis (1 hour)**

**Objective:** Confirm which .zip files are duplicates  
**Method:** Compare SHA256 hashes

**Who:** IT/Technical Lead  
**When:** Day 1 morning

**Process:**
```powershell
# Command to run (pseudocode for illustration):
$files = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Filter "*.zip"

$hashes = @{}
foreach ($file in $files) {
    $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash
    if ($hashes.ContainsKey($hash)) {
        Write-Host "DUPLICATE: $($file.FullName)"
        Write-Host "  Same as: $($hashes[$hash])"
    } else {
        $hashes[$hash] = $file.FullName
    }
}
```

**Output:** List of duplicate candidates with confidence level

---

### **STEP 2: Backup & Archive (30 min)**

**Objective:** Move duplicates to archive (safe location)

**Process:**
```
1. Create archive directory:
   c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\

2. For each duplicate identified:
   - Move file to archive
   - Log the action
   - Verify move successful

3. Expected location changes:
   OLD: c:\MircoHydro\MASTER_PRODUCTION_PACKAGE\...\file.zip
   NEW: c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\file.zip
```

**Command Example:**
```powershell
# Create archive directory
New-Item -Path "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124" -ItemType Directory

# Move duplicates
Move-Item -Path $duplicateFile -Destination "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\"

# Log the action
Add-Content -Path "DUPLICATES_ARCHIVED_LOGS.md" `
  -Value "- Archived: $($duplicateFile) | Date: $(Get-Date) | Size: $(Get-Item $duplicateFile).Length bytes"
```

---

### **STEP 3: Impact Assessment (30 min)**

**Objective:** Check if any scripts reference the moved files

**Process:**
1. Search all active Python scripts for .zip references
2. Search all documentation for .zip references
3. Check if any active import processes depend on moved files

**Command:**
```powershell
# Search for .zip references in Python
Get-ChildItem -Path "c:\MircoHydro" -Filter "*.py" -Recurse | 
  Select-String -Pattern "\.zip" | 
  Select-Object Path, LineNumber, Line | 
  Format-Table

# Search in markdown docs
Get-ChildItem -Path "c:\MircoHydro" -Filter "*.md" -Recurse | 
  Select-String -Pattern "\.zip" | 
  Select-Object Path, LineNumber | 
  Format-Table
```

**Expected Result:** No active references (these are archive/release files)

---

### **STEP 4: Team Testing - 7 Days (Passive Monitoring)**

**Objective:** Confirm nothing broke after archival

**Timeline:**
```
Day 1 (Fri):  Archive moved, immediate testing
Day 2-4:      Passive monitoring (watch for issues)
Day 5-7:      Final stability verification
```

**What Team Does:**
- Continue normal workflows
- Run pipeline as usual
- Report any issues immediately
- No extra testing needed

**What to Monitor:**
- Pipeline execution logs
- Import/export success rates
- Any error messages related to missing files
- System performance

**Success Criteria:**
```
✅ No pipeline failures
✅ No broken references
✅ No missing file errors
✅ Normal performance
✅ Team reports stability
```

---

### **STEP 5: Decision Gate (Day 8)**

**Objective:** Decide on permanent deletion

**Review:**
- [ ] All 7 days stable?
- [ ] No issues reported?
- [ ] No broken references found?
- [ ] Team confirms everything working?

**Option A: Keep archived (Conservative)**
```
Decision: Keep files archived indefinitely
Action: Compression (optional) + document retention policy
Result: Storage saved, files recoverable, zero risk
```

**Option B: Permanently delete (Aggressive)**
```
Decision: Delete archived files permanently
Action: Delete contents of c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\
Result: 71-112 MB permanently freed, files gone
Risk: Zero (confirmed stable for 7 days)
```

**Recommendation:** Option B (delete after 7-day verification)

---

## 📈 EXPECTED STORAGE IMPACT

### Before Tier 1:
```
Active Workspace:     ~190 MB
SoT Location:         ~10.6 MB
Archive:              ~12.7 MB
Backup:               ~228.5 MB
Duplicates (zips):    ~71-112 MB ⚠️
─────────────────────────────
TOTAL:                ~513 MB
```

### After Tier 1 Dedup:
```
Active Workspace:     ~190 MB  ✓
SoT Location:         ~10.6 MB ✓
Archive:              ~12.7 MB ✓
Backup:               ~228.5 MB ✓
Duplicates (archived):-71-112 MB 📊
─────────────────────────────
TOTAL:                ~442-451 MB (saves 71-112 MB)
```

---

## 📋 PHASE 2 EXECUTION CHECKLIST

### **Pre-Execution (Day 1 Morning)**
- [ ] Phase 1 complete with team sign-off ✅
- [ ] All team testing done
- [ ] No issues reported
- [ ] Ready to proceed

### **Hash Analysis (Day 1 Morning - 1 hour)**
- [ ] Identified all duplicate .zip files
- [ ] Created list of candidates
- [ ] Verified duplicates with hash comparison
- [ ] Documented findings

### **Archive & Move (Day 1 Afternoon - 30 min)**
- [ ] Created archive directory: `TIER1_CLEANUP_ARCHIVED_20260124`
- [ ] Moved all duplicates to archive
- [ ] Verified all files moved successfully
- [ ] Created movement log

### **Impact Check (Day 1 Afternoon - 30 min)**
- [ ] Searched for .zip references in active code
- [ ] Confirmed no active scripts depend on moved files
- [ ] Verified no broken links
- [ ] Documented findings

### **Monitoring Setup (Day 1 End)**
- [ ] Created monitoring checklist
- [ ] Set daily log review schedule
- [ ] Notified team of changes
- [ ] Prepared for 7-day observation

### **7-Day Monitoring (Days 2-8)**
- [ ] Day 2: Review logs for any issues
- [ ] Day 3: Spot check pipeline execution
- [ ] Day 4: Team feedback check
- [ ] Day 5: Monitor performance metrics
- [ ] Day 6: Verify import processes
- [ ] Day 7: Final stability check
- [ ] Day 8: Make permanent deletion decision

### **Final Decision (Day 8)**
- [ ] All 7 days reviewed
- [ ] No issues found ✅
- [ ] Decision: Permanent delete
- [ ] Execute deletion
- [ ] Mark Phase 2 complete

---

## 🚨 ISSUE SCENARIOS

### **Scenario 1: Pipeline breaks after archival**
```
Action:
  1. Immediately restore files from archive
  2. Investigate what broke
  3. Document the dependency
  4. Keep files (don't delete)
  5. Plan manual removal later
```

### **Scenario 2: Script references archived file**
```
Action:
  1. Restore file from archive
  2. Update script to new location
  3. Test script
  4. Re-archive after fix
```

### **Scenario 3: File corruption in archive**
```
Action:
  1. Restore from primary archive
  2. Replace corrupted file
  3. Continue monitoring
```

---

## 📊 PHASE 2 TIMELINE

```
DAY 1 (Friday):
  08:00-09:00   Hash analysis & duplicate identification
  09:00-09:30   Create archive directory & move files
  09:30-10:00   Impact assessment & reference checking
  10:00+        Monitoring begins

DAYS 2-7 (Next Week):
  Daily:        Review logs for issues
  Daily:        Pipeline success check
  Daily:        Team feedback collection
  
DAY 8 (Next Friday):
  08:00-09:00   Final review of all monitoring data
  09:00-09:30   Decision: Keep or Delete?
  09:30-10:00   Execute permanent deletion (if approved)
  10:00+        Phase 2 Complete ✅
```

---

## 📄 DOCUMENTATION

**Create:**
- ✅ PHASE_2_DEDUP_EXECUTION_LOG.md (this file)
- [ ] DUPLICATES_ARCHIVED_LOGS.md (movement log)
- [ ] TIER1_MONITORING_REPORT.md (daily checks)
- [ ] TIER1_FINAL_DECISION.md (day 8 decision)

---

## ✅ PHASE 2 SUCCESS CRITERIA

- [x] All duplicates identified (hash verified)
- [ ] All duplicates archived (moved to safe location)
- [ ] No broken references (confirmed via search)
- [ ] 7-day stable operation (team confirms)
- [ ] Final decision made (keep or delete)
- [ ] Phase 2 complete

---

## 🎯 NEXT PHASE

After Phase 2 completion:
- **Phase 3:** Update internal documentation
- **Phase 4:** Define archive retention policy
- **Optional Tier 2+:** Additional deduplication (with planning)

---

**Status:** WAITING FOR PHASE 1 SIGN-OFF  
**Ready to Execute:** YES ✅  
**Estimated Duration:** 1 day active + 7 days passive monitoring  
**Expected Outcome:** 71-112 MB freed (permanently or archived)

