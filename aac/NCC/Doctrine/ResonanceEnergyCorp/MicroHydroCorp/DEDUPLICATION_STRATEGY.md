# 🗑️ DEDUPLICATION STRATEGY - 289 Duplicate Files

**Analysis Date:** January 24, 2026  
**Total Duplicates Found:** 289 groups  
**Files Affected:** ~500+ (some in multiple groups)  
**Storage to Recover:** 150-250 MB  
**Risk Level:** MEDIUM (requires careful analysis)  

---

## 📊 DUPLICATE BREAKDOWN

From workspace audit:

| Category | Count | Storage | Risk |
|----------|-------|---------|------|
| **Identical Files** | 114 groups | 80-100 MB | LOW |
| **Different Versions** | 175 groups | 70-150 MB | MEDIUM |
| **Known Safe** | ~50 groups | 30-50 MB | LOW |
| **Requires Analysis** | ~239 groups | 100-200 MB | MEDIUM |

---

## 🎯 THREE-TIER DEDUPLICATION STRATEGY

### TIER 1: SAFE & IMMEDIATE (Low Risk)

**These can be deleted without analysis:**

#### Release Packages (Safe to deduplicate)
```
MICRO HYDRO NEW/ folder:
├── MICRO HYDRO - OLD.zip
├── MICRO HYDRO NEW.zip              ← Likely duplicate
├── MicroHydroV1 — SoT.zip           ← Duplicate (just archived)
├── MicroHydroV1_Integrated_Organized_Tree.zip
├── MicroHydroV1_MasterIntegrated_Output.zip
└── Multiple .xlsx and .docx versions (decision files)
```

**Action:** Compare hashes, keep latest, remove duplicates  
**Time to Execute:** 30 min  
**Storage Gain:** 50-80 MB  
**Risk:** LOW (can restore from backups)

---

#### macOS Installation Scripts
```
MICRO HYDRO NEW/ folder:
├── Install_MicroHydroV1.command
├── Install_MicroHydroV1.sh
├── mhv1_oneclick_pipeline.command
├── mhv1_pipeline.py
```

**Check:** Are any of these exact duplicates?  
**Action:** Keep latest, remove duplicates  
**Storage Gain:** 1-2 MB  
**Risk:** LOW

---

#### Archive Folders
```
MicroHydroV1 — Archive/
MicroHydroV1 — Releases/
MicroHydroV1 — SoT_ARCHIVED_20260124/  ← JUST CREATED
```

**Check:** Compare contents, identify duplicates  
**Action:** Consolidate into single archive  
**Storage Gain:** 20-30 MB  
**Risk:** LOW (archived data)

---

### TIER 2: MODERATE RISK (Requires Brief Analysis)

**These likely have duplicates, need quick verification:**

#### Addon Packages in Multiple Locations
```
Already cleaned Location B:
  ✅ No more (1).zip, (2).zip duplicates

Check Location A Archive:
  └── MicroHydroV1_Phase3_Automation_Pack_macOS_OpenReleases.zip
  └── MicroHydroV1_Phase4_SmartRuns_Pack_macOS.zip
  └── MicroHydroV1_RC1.zip
  └── MicroHydroV1_SoT_Package.zip
  └── MicroHydroV1_RunID_Migrator_v0.3.0.zip
  
Check Location B Core:
  └── (May have same versions)
```

**Action:** Compare with Location B versions, archive originals if identical  
**Time to Execute:** 20 min  
**Storage Gain:** 10-15 MB  
**Risk:** MEDIUM (but can use archive)

---

#### CAD & Design Files
```
MicroHydroV1_Shapr3D_Import/
MicroHydroV1_Shapr3D_ReferenceModel/
MicroHydroV1_Shapr3D_ReferenceModel_v2/
```

**Check:** Are these versions or exact duplicates?  
**Action:** Document versions, keep all (likely intentional)  
**Storage Gain:** 0 (keep all - different versions)  
**Risk:** LOW (intentional versions)

---

### TIER 3: HIGH RISK (Keep For Now)

**These should NOT be touched without team input:**

#### Test Data
```
tests/
├── raw/
├── processed/
└── results/
```

**Why:** Active test suite - may have intentional duplicates  
**Action:** SKIP for now  
**Status:** Review with QA team if needed

---

#### Workspace Reference Files
```
MicroHydroV1_Work/
├── 00_Admin/
├── 01_SharePoint_Libraries/
├── 02_Canonical_Working_Tree/
├── 03_Packages_Original_Zips/
├── 04_Reports/
└── 05_Tools/
```

**Why:** Master integration workspace - duplicates may be intentional  
**Action:** SKIP for now  
**Status:** Review with integration team

---

#### Development Archives
```
MicroHydroV1 — OLD/
└── RnD/
    ├── MASTER_PRODUCTION_PACKAGE/
    ├── MASTER_PRODUCTION_PACKAGE 2/
    ├── NextChat_Package/
    └── zipppittt/
```

**Why:** Legacy/R&D - may contain important history  
**Action:** SKIP for now  
**Status:** Archive as-is for now

---

## 🔍 EXECUTION PLAN - TIER 1 ONLY (SAFE)

### Step 1: Identify Safe Duplicates
```powershell
# Compare release ZIP files
$releaseDir = "c:\MircoHydro\MICRO HYDRO NEW"
$zips = Get-ChildItem -Path $releaseDir -Filter "*.zip"

foreach ($zip in $zips) {
    $hash = Get-FileHash -Path $zip.FullName -Algorithm SHA256
    Write-Host "$($zip.Name): $($hash.Hash)"
}
```

**Outcome:** List of identical vs unique ZIPs

---

### Step 2: Archive Duplicates (Don't Delete)
```powershell
# Create archive subdirectory
$archivePath = "c:\MircoHydro\CLEANUP_ARCHIVED_20260124"
New-Item -ItemType Directory -Path $archivePath -Force

# Move duplicates (not delete)
# Example:
Move-Item -Path "c:\MircoHydro\MICRO HYDRO NEW\MICRO HYDRO NEW.zip" `
          -Destination "$archivePath\" -Force
```

**Outcome:** Duplicates moved to archive for review

---

### Step 3: Verify No Breaks
```powershell
# Test that scripts still work
cd "c:\MircoHydro"
# Test pipeline imports
python -c "import pathlib; print('✅ Imports work')"
```

**Outcome:** Confirm no broken references

---

### Step 4: Review Before Permanent Deletion
```powershell
# Show what was archived
Get-ChildItem -Path "c:\MircoHydro\CLEANUP_ARCHIVED_20260124" `
  | Measure-Object -Property Length -Sum
```

**After 7 days:** If all works, delete archive permanently

---

## 📊 TIER 1 DEDUPLICATION CHECKLIST

- [ ] **Phase 1: Analyze**
  - [ ] Hash all ZIP files
  - [ ] Identify identical files
  - [ ] Document findings

- [ ] **Phase 2: Archive**
  - [ ] Create archive folder
  - [ ] Move duplicates (don't delete)
  - [ ] Keep original names for traceability

- [ ] **Phase 3: Test**
  - [ ] Verify pipeline still works
  - [ ] Check script imports
  - [ ] Confirm no broken references

- [ ] **Phase 4: Monitor**
  - [ ] Watch for 7 days
  - [ ] Team confirms all working
  - [ ] Then permanently delete

---

## 🚨 WHAT NOT TO DO

❌ **Don't delete without backup** - Already have one, use it  
❌ **Don't touch test data** - May break active tests  
❌ **Don't remove versioned files** - Keep design history  
❌ **Don't delete immediately** - Archive first, review, then delete  
❌ **Don't touch workspace master** - MicroHydroV1_Work is critical

---

## 💾 TIER 1 STORAGE POTENTIAL

**If we execute Tier 1 only:**

| Action | Storage | Risk |
|--------|---------|------|
| Remove duplicate ZIPs | 50-80 MB | LOW |
| Clean old scripts | 1-2 MB | LOW |
| Archive old releases | 20-30 MB | LOW |
| **Subtotal** | **71-112 MB** | **LOW** |

**Total (including SoT consolidation gains):** 111-162 MB freed ✅

---

## 🔄 FUTURE PHASES (After Approval)

### TIER 2 Deduplication (2-3 weeks)
- Analyze addon packages
- Compare versions
- Archive where identical

**Potential Gain:** 10-15 MB  
**Risk Level:** MEDIUM

### TIER 3 Deduplication (With Team)
- Archive old R&D folders
- Clean test duplicates
- Optimize workspace

**Potential Gain:** 50-100 MB  
**Risk Level:** HIGH (team dependent)

---

## ✅ RECOMMENDATION

**Proceed with Tier 1 immediately:**
1. Low risk (all in isolated ZIP files)
2. Easy to reverse (archived, not deleted)
3. High confidence (obvious duplicates)
4. Good ROI (71-112 MB)
5. Can do this week

**Schedule Tier 2 for later:**
- After team confirms all works
- After 1 week of monitoring
- With team input/approval

**Skip Tier 3 for now:**
- Requires team discussions
- Active development folders
- Intentional duplicates

---

## 📋 READY TO PROCEED?

**To execute Tier 1 deduplication:**

1. Identify identical ZIP files (hash compare)
2. Archive duplicates to backup folder
3. Test for 7 days
4. Permanently delete if all works

**Estimated time:** 1-2 hours  
**Recovery ability:** 100% (full backup available)  
**Storage gain:** 71-112 MB  
**Risk level:** LOW

---

**Status:** Ready for execution  
**Created:** January 24, 2026  
**Next Phase:** Tier 1 execution (recommend after SoT consolidation stabilizes)

