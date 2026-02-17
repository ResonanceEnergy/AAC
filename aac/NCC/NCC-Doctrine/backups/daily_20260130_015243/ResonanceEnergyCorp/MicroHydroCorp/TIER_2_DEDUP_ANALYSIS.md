# Tier 2 Deduplication Analysis

**Date:** January 24, 2026  
**Status:** Planning Phase  
**Execution Target:** February 1-21, 2026

---

## Overview

Tier 2 targets duplicate CAD files, spreadsheets, and documentation across the workspace — medium-priority items with moderate storage impact but lower risk profile.

---

## Target Categories

### 1. Duplicate CAD Files (.step, .iges, .FCStd)

**Identified Locations:**
- `cad/exports/` — Multiple STEP export versions
- `MicroHydroV1_Shapr3D_Import/` — Assembly duplicates
- `tests/` — Reference CAD model backups

**Examples of Duplicates:**
- `Diffuser.step` (v1, v2, v3, v4 - minor revisions)
- `FullAssembly.step` (multiple exports from different dates)
- `VanePack_Reference.step` (3+ versions)
- FreeCAD project files (.FCStd, multiple saves)

**Estimated Count:** 15-20 duplicate files  
**Estimated Size:** ~3-5 MB  
**Risk Level:** MEDIUM (need reference versions, but duplicates are safe to remove)

---

### 2. Duplicate Spreadsheets (.xlsx, .xls, .csv)

**Identified Locations:**
- `data/` — RnD export versions
- `tests/results/` — Data analysis spreadsheets
- `docs/templates/` — Template backups
- `automation/` — Config and measurement imports

**Examples of Duplicates:**
- `MicroHydroV1_RnD_Export.xlsx` (dated versions v1.0, v1.1, v1.2)
- `Measurements_Import_Template.csv` (5+ template versions)
- Analysis spreadsheets with "_OLD", "_BACKUP", "_ARCHIVE" suffixes
- Merged/consolidated versions alongside originals

**Estimated Count:** 12-18 duplicate files  
**Estimated Size:** ~2-3 MB  
**Risk Level:** MEDIUM (data files, need care)

---

### 3. Duplicate Documentation (.pdf, .md, .docx)

**Identified Locations:**
- `docs/` — PDF exports of procedures
- `docs/reports/` — Multiple versions of reports
- `docs/procedures/` — Superseded procedure documents
- `docs/master/` — Outdated master documents

**Examples of Duplicates:**
- Assembly procedure (v1, v2, v3, final, final_revised)
- System design documentation (multiple PDF exports)
- Meeting notes (cumulative, redundant copies)
- Procedure revisions (old versions not deleted)

**Estimated Count:** 8-14 duplicate files  
**Estimated Size:** ~3-4 MB  
**Risk Level:** LOW (documentation, easily recoverable)

---

### 4. Reference Model Exports (.DXF, .STL)

**Identified Locations:**
- `MicroHydroV1_Shapr3D_ReferenceModel/DXF/` — Multiple DXF exports
- `MicroHydroV1_Shapr3D_ReferenceModel_v2/STL_Reference/` — STL versions

**Examples of Duplicates:**
- Assembly.DXF (v1 and v2 - old and new reference)
- Component STLs (multiple format conversions, redundant)

**Estimated Count:** 5-10 duplicate files  
**Estimated Size:** ~2-3 MB  
**Risk Level:** LOW (reference only, not used in production)

---

## Aggregated Tier 2 Summary

| Category | Files | Size | Risk |
|----------|-------|------|------|
| **CAD Files** | 15-20 | 3-5 MB | MEDIUM |
| **Spreadsheets** | 12-18 | 2-3 MB | MEDIUM |
| **Documentation** | 8-14 | 3-4 MB | LOW |
| **Reference Models** | 5-10 | 2-3 MB | LOW |
| **TOTAL** | 40-62 | ~14 MB | MEDIUM |

---

## Comparison to Tier 1

| Aspect | Tier 1 | Tier 2 |
|--------|--------|--------|
| **File Type** | .zip packages | .step, .xlsx, .pdf, .dxf |
| **Count** | 91 files | ~40-62 files |
| **Size** | 25.9 MB | ~14 MB |
| **Risk** | LOW (already archived) | MEDIUM (active references) |
| **Observation Period** | 7 days | 14 days |
| **Decision Date** | Jan 31 | Feb 15 |
| **Execution Date** | Feb 1 | Feb 16 |

---

## Execution Plan

### Phase 1: Hash Analysis (Feb 1)

**Goal:** Identify exact duplicates using SHA256 hashing

```powershell
# Scan all CAD, spreadsheet, and document files
$files = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Include @("*.step", "*.iges", "*.xlsx", "*.xls", "*.csv", "*.pdf", "*.md", "*.docx", "*.dxf", "*.stl")

# Build hash table
$hashes = @{}
$duplicates = @()

foreach ($file in $files) {
    $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash
    
    if ($hashes.ContainsKey($hash)) {
        $duplicates += $file
    } else {
        $hashes[$hash] = $file.FullName
    }
}

# Results expected: 40-62 duplicates
```

**Expected Result:** 40-62 duplicates identified, ~14 MB total

**Time:** 10-15 minutes

---

### Phase 2: Archive Creation & Movement (Feb 1)

**Goal:** Move duplicates to safe archive (not delete)

```powershell
# Create archive directory
$archiveDir = "c:\MircoHydro\TIER2_CLEANUP_ARCHIVED_20260201"
New-Item -Path $archiveDir -ItemType Directory -Force

# Move all duplicates
foreach ($file in $duplicates) {
    Move-Item -Path $file.FullName -Destination $archiveDir -Force
}
```

**Expected Result:** All 40-62 files moved, archive created

**Time:** 5-10 minutes

---

### Phase 3: Observation Period (Feb 1-14)

**Goal:** Monitor system health for 14 days

**Daily Checks:**
- Archive file count stable
- CAD tools operational
- Data analysis processes normal
- Documentation accessible
- Pipeline execution normal
- Backup accessible

**Monitoring Tool:** Same as Tier 1 (extended for 14 days)

**Expected Result:** All checks pass, ready for Feb 15 decision

**Time:** 5 min/day × 14 days = 70 minutes total

---

### Phase 4: Decision & Deletion (Feb 15-16)

**Decision Criteria:**
- [ ] 14-day observation complete
- [ ] All daily checks passed
- [ ] No system issues reported
- [ ] No team objections
- [ ] CAD tools unaffected
- [ ] Data analysis unaffected

**If Approved:**
```powershell
# Permanent delete archived files
Remove-Item -Path "c:\MircoHydro\TIER2_CLEANUP_ARCHIVED_20260201\*" -Recurse -Force
```

**Expected Result:** 14 MB freed, archive directory kept for audit

**If Rejected:**
- Extend observation 30 more days (until Mar 15)
- Investigate issues and resolve
- Re-evaluate on Mar 15

---

## Risk Mitigation

### Pre-Archive Checklist

- [ ] All files backed up in safety backup (228.5 MB)
- [ ] SHA256 hashes recorded in TIER2_ARCHIVE_MANIFEST.md
- [ ] Original locations documented
- [ ] Backup copy created before archive
- [ ] Team notified of upcoming archive

### Monitoring Safeguards

- [ ] Daily archive integrity checks
- [ ] Weekly performance metrics
- [ ] Emergency rollback capability (5 min restore)
- [ ] Full backup available entire time
- [ ] Escalation procedures defined

### Approval Safeguards

- [ ] Technical review of duplicate detection
- [ ] CAD team sign-off on CAD file safety
- [ ] Data team sign-off on spreadsheet safety
- [ ] 14-day observation (longer than Tier 1)
- [ ] Executive approval before deletion

---

## Team Impact Assessment

### During Archive Period (Feb 1)
- ✅ No impact — files archived, not deleted
- ✅ Work normally — system fully operational
- ✅ Safe to revert — archive easily restored

### During Observation Period (Feb 1-14)
- ✅ No interruptions — all systems functional
- ✅ Monitoring transparent — team can see progress
- ✅ Rollback available — restore in 5 minutes if needed

### After Approval (Feb 16+)
- ✅ 14 MB freed — permanent storage savings
- ✅ System responsive — no performance degradation
- ⏳ Tier 3 planning — next phase begins Feb 22

---

## Dependencies & Prerequisites

**Before Executing Tier 2:**

- [ ] Tier 1 must be completed successfully (Jan 31 decision)
- [ ] Tier 1 monitoring must show all green (7/7 days pass)
- [ ] Team approval must be obtained for Tier 1 deletion
- [ ] Full backup must still be available (228.5 MB)
- [ ] Safety systems must be operational

**Blocking Conditions:**

- ❌ Tier 1 fails (archive files missing): DO NOT PROCEED
- ❌ SoT corrupted during Tier 1: DO NOT PROCEED
- ❌ Critical issue discovered during Tier 1: DO NOT PROCEED
- ❌ Team objects to Tier 1 deletion: DO NOT PROCEED

---

## Success Metrics

**Phase 1 (Hash Analysis):**
- Duplicates identified: 40-62 files
- Total size: ~14 MB
- Analysis time: < 20 minutes
- Accuracy: 100% (hash verification)

**Phase 2 (Archive & Move):**
- Files moved: 40-62/40-62 (100%)
- Failures: 0
- Movement time: < 15 minutes
- Archive created: ✅

**Phase 3 (Observation):**
- Monitoring days: 14/14 complete
- Daily checks passing: 14/14 (100%)
- Critical issues: 0
- Rollback need: 0

**Phase 4 (Decision & Deletion):**
- Team approval: ✅
- Storage freed: ~14 MB
- Deletion time: < 5 minutes
- Audit trail: Preserved

---

## Timeline

| Date | Event | Duration |
|------|-------|----------|
| **Feb 1** | Hash analysis + archive + move | 20-25 min |
| **Feb 1-14** | 14-day observation period | 14 days |
| **Feb 15** | Decision review meeting | 30 min |
| **Feb 16** | Permanent deletion (if approved) | 5 min |
| **Feb 22+** | Tier 3 execution begins | - |

---

## Documentation

All files created in: `c:\MircoHydro\`

- **TIER_2_DEDUP_ANALYSIS.md** (this file)
- **TIER2_ARCHIVE_MANIFEST.md** (created during Phase 1)
- **TIER2_MONITORING_CHECKLIST.md** (created during Phase 3)

---

## Questions & Support

- **Overall strategy:** See STORAGE_OPTIMIZATION_ROADMAP_90DAY.md
- **Retention policy:** See ARCHIVE_RETENTION_POLICY.md
- **Tier 1 results:** See CONSOLIDATION_SUMMARY.md
- **Team communication:** See STORAGE_STRATEGY_SUMMARY.md

