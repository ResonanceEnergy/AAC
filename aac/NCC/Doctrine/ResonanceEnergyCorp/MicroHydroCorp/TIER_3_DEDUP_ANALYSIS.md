# Tier 3 Deduplication Analysis

**Date:** January 24, 2026  
**Status:** Planning Phase  
**Execution Target:** February 22 - March 31, 2026

---

## Overview

Tier 3 targets the largest deduplication opportunity: old project backups, research archives, legacy test data, and redundant installer packages. High-impact (100+ MB) but also highest-risk due to historical significance.

---

## Target Categories

### 1. Old Project Backups (~40 MB)

**Identified Locations:**
- `MicroHydroV1 — Archive/` (entire folder)
- `MicroHydroV1_InitGui_PATCH/` (patch archives)
- Dated backup folders in `MicroHydroV1_Work/`
- Snapshot archives (e.g., `*_20260122_005746_BACKUP`)

**Examples:**
- `MicroHydroV1_BACKUP_20260122_005746/` (full backup, no longer needed)
- `MicroHydroV1_InitGui_PATCH/` (patch superseded by v0.3.0)
- Intermediate work directories (completed, archived)
- Old build artifacts

**Why Duplicates?**
- Created for backup/safety during development
- Never integrated back into main project
- Superseded by newer versions
- Left behind after migrations

**Estimated Count:** 20-40 duplicate directories/files  
**Estimated Size:** ~40 MB  
**Risk Level:** **HIGH** (historical backups, need careful review)

---

### 2. Duplicate RnD Exports (~25 MB)

**Identified Locations:**
- `data/runs/` — Multiple run output versions
- `data/reference/` — Old reference data
- RnD subdirectories with dated exports

**Examples:**
- `RUN_001_results/` (v1, v2, v3 with minor differences)
- `analysis_export_20260115/` (old analysis)
- `parameter_study_outputs/` (multiple incomplete runs)
- Consolidated data with originals still present

**Why Duplicates?**
- Multiple runs of same experiment (optimization iterations)
- Old analysis versions kept "just in case"
- Research exploratory branches never cleaned up
- Export consolidation without cleanup

**Estimated Count:** 30-50 duplicate files/directories  
**Estimated Size:** ~25 MB  
**Risk Level:** **HIGH** (research data, peer review sensitive)

---

### 3. Legacy Test Data (~15 MB)

**Identified Locations:**
- `tests/raw/` — Old test recordings
- `tests/processed/` — Processed test data backups
- `tests/results/` — Archived test reports

**Examples:**
- Test data from v0.1.0, v0.2.0 (superseded)
- Multiple recorded runs of same test scenario
- Backup copies with `_OLD`, `_ARCHIVE`, `_BACKUP` suffixes
- Pre-consolidation test datasets

**Why Duplicates?**
- Testing iterations not cleaned up
- Multiple data formats (CSV, JSON, pickle) of same test
- Version control artifacts left in filesystem
- Backup copies pre-dating modern archive strategy

**Estimated Count:** 40-60 duplicate files  
**Estimated Size:** ~15 MB  
**Risk Level:** **MEDIUM** (test data, regenerable if needed)

---

### 4. Old Installer Packages (~20 MB)

**Identified Locations:**
- `MASTER_PRODUCTION_PACKAGE/` (multiple, redundant)
- `MASTER_PRODUCTION_PACKAGE 2/` (alternative/backup)
- `MicroHydroV1_CAD Archive/` (CAD-specific installers)
- Old setup files and installers

**Examples:**
- Multiple complete package zips
- CAD-specific installer packages (v0.1, v0.2, v0.3)
- macOS and Windows install variants (duplicated)
- Build artifacts from different compilation passes

**Why Duplicates?**
- Each version/platform archived separately
- Multiple build attempts left in place
- Consolidation packages created without deleting originals
- Installation testing artifacts

**Estimated Count:** 10-20 duplicate files  
**Estimated Size:** ~20 MB  
**Risk Level:** **MEDIUM** (installers, can be rebuilt if needed)

---

## Aggregated Tier 3 Summary

| Category | Size | Count | Risk | Notes |
|----------|------|-------|------|-------|
| **Old Backups** | ~40 MB | 20-40 | HIGH | Historical significance, review needed |
| **RnD Exports** | ~25 MB | 30-50 | HIGH | Research data, peer review sensitive |
| **Test Data** | ~15 MB | 40-60 | MEDIUM | Regenerable, less critical |
| **Installers** | ~20 MB | 10-20 | MEDIUM | Rebuild-able, not production |
| **TOTAL** | **~100 MB** | **100-170** | **HIGH** | Largest opportunity, most care needed |

---

## Comparison to Tiers 1 & 2

| Aspect | Tier 1 | Tier 2 | Tier 3 |
|--------|--------|--------|--------|
| **File Type** | .zip packages | .step, .xlsx, .pdf | Backups, data, test, installers |
| **Count** | 91 files | ~40-62 files | ~100-170 directories |
| **Size** | 25.9 MB | ~14 MB | **~100 MB** |
| **Risk** | **LOW** | MEDIUM | **HIGH** |
| **Observation** | 7 days | 14 days | **30 days** |
| **Decision** | Jan 31 | Feb 15 | **Mar 23** |
| **Community Review** | None | Optional | **Required** |
| **Exec Approval** | Tech Lead | TL + Dev Mgr | **TL + Proj Mgr** |

---

## Risk Assessment

### Why Tier 3 is Higher Risk

1. **Historical Significance:**
   - Backups contain project evolution history
   - Research data may be needed for reproducibility
   - Can't recreate old experimental conditions

2. **Peer Review Sensitivity:**
   - RnD data might be cited in papers
   - Test data used for validation
   - Deletion could complicate audits

3. **Regulatory Compliance:**
   - May need to preserve for compliance
   - Research institution requirements
   - Engineering design traceability

4. **Irreplaceability:**
   - Old hardware/software no longer available
   - Environmental conditions changed
   - Original experimenters moved on

### Mitigation Strategy

**NEVER delete without:**
- [ ] 30-day observation (vs. 7-14 for Tiers 1-2)
- [ ] Explicit community review (research team feedback)
- [ ] Executive approval (Project Manager + Team Lead sign-off)
- [ ] Peer review of research data deletion (if applicable)
- [ ] Permanent archive record (what was deleted, when, why)

---

## Execution Plan

### Phase 1: Inventory & Community Review (Feb 22)

**Goal:** Create detailed inventory and solicit team feedback

```powershell
# Scan and categorize all potential Tier 3 duplicates
$tier3Files = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Include @("*BACKUP*", "*ARCHIVE*", "*OLD*", "*RUN_*", "*_export*") | Where-Object { $_.CreationTime -lt (Get-Date).AddMonths(-3) }

# Calculate totals
$totalSize = ($tier3Files | Measure-Object -Property Length -Sum).Sum / 1MB
$totalCount = $tier3Files.Count

# Create inventory manifest
# Expected: ~100-170 items, ~100 MB
```

**Deliverable:** `TIER3_INVENTORY_AND_REVIEW.md`

**Steps:**
1. Create detailed inventory with file descriptions
2. Email to research team: "These files may be deleted, please review"
3. Collection period: Feb 22-28 (1 week feedback window)
4. Summary: What team wants to keep, what's safe to delete

**Expected Result:** Community consensus on what can safely delete

**Time:** 30 minutes creation + 1 week feedback = 8 days calendar

---

### Phase 2: Hash Analysis & Categorization (Mar 1)

**Goal:** Identify specific duplicates after feedback

```powershell
# After community input, refine deletion list
# Hash-analyze final list of "safe to delete" files
# Create Tier 3 archive with only approved items

# Expected: 50-100 MB (conservative, respecting feedback)
```

**Deliverable:** `TIER3_ARCHIVE_MANIFEST.md`

**Expected Result:** 50-100 MB of approved duplicates identified

**Time:** 20-30 minutes

---

### Phase 3: Archive & Move (Mar 1)

**Goal:** Move approved duplicates to archive

```powershell
# Create Tier 3 archive directory
$archiveDir = "c:\MircoHydro\TIER3_CLEANUP_ARCHIVED_20260301"
New-Item -Path $archiveDir -ItemType Directory -Force

# Move only approved duplicates
# Expected: 50-100 MB moved
```

**Expected Result:** All 50-100 MB safely archived

**Time:** 10-20 minutes

---

### Phase 4: Observation Period (Mar 1-31)

**Goal:** Monitor for 30 days (longest observation, highest risk)

**Daily Checks (5 min/day):**
- Archive files intact
- Research systems operational
- No peer review issues reported
- Test data accessible if needed
- Backup available

**Monitoring Tool:** Extended checklist for 30 days

**Expected Result:** All checks pass, ready for Mar 31 decision

**Time:** 5 min/day × 30 days = 150 minutes total

---

### Phase 5: Executive Review & Decision (Mar 31)

**Decision Criteria:**
- [ ] 30-day observation complete (all days pass)
- [ ] Community feedback satisfied
- [ ] Research team confirms safety
- [ ] No critical issues during observation
- [ ] Project Manager approves deletion
- [ ] Team Lead approves deletion

**If Approved:**
```powershell
# Permanent delete archived files
Remove-Item -Path "c:\MircoHydro\TIER3_CLEANUP_ARCHIVED_20260301\*" -Recurse -Force
```

**Expected Result:** 50-100 MB freed

**If Rejected:**
- Keep archived indefinitely
- No further deletion attempts
- Archive becomes permanent historical record

---

## Alternative: Hybrid Approach

**If team hesitant about deleting research data:**

```
Option A: Delete only installers (~20 MB) immediately
Option B: Keep backups and test data for 1 more year
Option C: Compress instead of delete (saves 50-80% space)
```

---

## Team Communication Strategy

### Announcement (Feb 1)
"In February, we'll review ~100 MB of old backups and test data for potential cleanup. We want your input."

### Inventory Review (Feb 22-28)
"Here's the list of files we're considering for deletion. Please flag anything you need."

### Archive (Mar 1)
"We're archiving approved items for 30-day observation. Nothing deleted yet, full rollback available."

### Decision (Mar 31)
"After 30 days of successful monitoring, we'll permanently delete unless you object."

---

## Dependencies & Prerequisites

**Before Executing Tier 3:**

- [ ] Tier 1 successfully deleted (Feb 1 complete)
- [ ] Tier 2 successfully deleted (Feb 16 complete) — **OR** Tier 2 skipped
- [ ] Community feedback period complete (Feb 28)
- [ ] No critical issues during Tier 1-2
- [ ] Full backup still available (228.5 MB minimum)
- [ ] Safety systems operational

**Blocking Conditions:**

- ❌ Tier 1 deletion failed: DO NOT PROCEED
- ❌ Tier 2 issues unresolved: PAUSE Tier 3
- ❌ Research team objects: DO NOT DELETE
- ❌ Peer review concern: DO NOT DELETE
- ❌ Compliance issue discovered: DO NOT DELETE

---

## Success Metrics

**Phase 1 (Inventory & Review):**
- Inventory items: 100-170 files
- Total size: ~100 MB
- Community feedback: 100% of team responds
- Consensus achieved: Clear "safe to delete" list

**Phase 2 (Hash Analysis):**
- Duplicates identified: 50-100 items
- Approved for deletion: 50-100 MB
- Analysis time: < 30 minutes
- Accuracy: 100%

**Phase 3 (Archive & Move):**
- Files moved: 50-100 MB
- Failures: 0
- Movement time: < 20 minutes
- Archive created: ✅

**Phase 4 (Observation):**
- Monitoring days: 30/30 complete
- Daily checks: 30/30 passing
- Critical issues: 0
- Rollback need: 0

**Phase 5 (Decision & Deletion):**
- Team approval: ✅ (explicit)
- Exec approval: ✅ (PM + TL)
- Storage freed: 50-100 MB
- Deletion time: < 10 minutes
- Audit trail: Preserved

---

## Timeline

| Date | Event | Duration |
|------|-------|----------|
| **Feb 22** | Inventory + community review start | 30 min |
| **Feb 22-28** | Feedback period | 7 days |
| **Mar 1** | Finalize list + archive analysis + move | 50 min |
| **Mar 1-31** | 30-day observation period | 30 days |
| **Mar 31** | Decision review + approvals | 1 hour |
| **Apr 1+** | Permanent deletion (if approved) | 10 min |

**Total active time:** ~3 hours spread over 67 days

---

## Alternative Scenarios

### Scenario A: Team Requests Full Preservation
- **Decision:** Keep everything indefinitely
- **Result:** Tier 3 not executed
- **Impact:** No additional storage savings beyond Tier 1-2

### Scenario B: Team Approves Partial Deletion
- **Decision:** Delete only installers + test data (~35 MB)
- **Keep:** Research backups + RnD exports (conservation approach)
- **Result:** ~35 MB saved instead of 50-100 MB

### Scenario C: Compression Instead of Deletion
- **Decision:** Compress old backups to 20% of size (saves 80 MB)
- **Benefit:** Preserves data while freeing space
- **Result:** Same space savings with higher recovery confidence

### Scenario D: Gradual Deletion
- **Decision:** Delete 25 MB immediately, review before deleting rest
- **Timeline:** Two decision points instead of one
- **Benefit:** Lower risk, incremental confidence building

---

## Success Criteria for Phase 3 Approval

For Tier 3 to proceed, **ALL** must be true:

1. ✅ Tier 1 & 2 deleted successfully
2. ✅ Community 100% consulted
3. ✅ Research team confirmed "safe"
4. ✅ 30-day observation all green
5. ✅ Project Manager approval obtained
6. ✅ Team Lead approval obtained
7. ✅ Full backup still available
8. ✅ No critical system issues

---

## Documentation

All files created in: `c:\MircoHydro\`

**Phase 1 (Inventory):**
- `TIER3_INVENTORY_AND_REVIEW.md`

**Phase 2 (Analysis):**
- `TIER3_ARCHIVE_MANIFEST.md`

**Phase 4 (Monitoring):**
- `TIER3_MONITORING_CHECKLIST_30DAY.md`

---

## Questions & Support

- **Overall strategy:** STORAGE_OPTIMIZATION_ROADMAP_90DAY.md
- **Retention policy:** ARCHIVE_RETENTION_POLICY.md
- **Community review:** Your research team lead
- **Executive approval:** Project Manager

