# Archive Retention and Deletion Policy

**Date:** January 24, 2026  
**Policy Version:** 1.0  
**Status:** Active

---

## Purpose

This policy defines when, how, and under what conditions archived files are retained, monitored, or permanently deleted from the MicroHydroV1 workspace.

---

## Core Principles

1. **Archive Before Delete** — Never delete files directly; always archive first
2. **Observe and Monitor** — Required observation period before permanent deletion
3. **Safety First** — Full backup available during entire observation period
4. **Data Integrity** — Hash verification ensures no data loss
5. **Team Approval** — Explicit sign-off before permanent deletion

---

## Tier 1: Duplicate .zip Files (Active)

**Archive Date:** January 24, 2026  
**Archive Location:** `c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\`  
**Files:** 91 duplicate .zip files  
**Total Size:** 25.9 MB  

### Retention Timeline

| Date Range | Phase | Action | Duration |
|-----------|-------|--------|----------|
| Jan 24-31 | **Observation** | Monitor archive daily | **7 days** |
| Jan 31 | **Decision** | Team reviews results | **1 day** |
| Feb 1+ | **Permanent Delete** | If approved, delete files | **Immediate** |

### Observation Period (Jan 24-31)

**Daily Monitoring (5 min/day):**
- [ ] Archive file count = 91 (no changes)
- [ ] Archive total size = 25.9 MB (no changes)
- [ ] Primary SoT operational (no errors)
- [ ] Pipeline execution (completes normally)
- [ ] Backup accessible (can restore if needed)

**Monitoring Tools:**
- Script: `DAILY_MONITORING_CHECK.ps1`
- Log: Daily results in `MONITORING_CHECKLIST_7DAY.md`
- Dashboard: Summary in `CONSOLIDATION_SUMMARY.md`

### Approval Criteria (Jan 31 Decision)

**All of the following must be TRUE:**

- [ ] **Observation complete:** 7 days of monitoring done
- [ ] **All daily checks passed:** 0 failures in monitoring logs
- [ ] **Team approval:** No objections from core team
- [ ] **No critical issues:** Zero system degradation
- [ ] **Archive intact:** All 91 files still present

**If ALL criteria met:**
```
✅ APPROVED: Execute permanent deletion
Estimated savings: 25.9 MB
Expected completion: Feb 1, 2026
```

**If ANY criteria fails:**
```
⏳ CONTINUE: Extend monitoring for 30 days
Re-evaluate: Feb 28, 2026
```

### Permanent Deletion (If Approved)

**Command:**
```powershell
Remove-Item -Path "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\*" -Recurse -Force
```

**Expected Result:**
- 91 files permanently deleted
- 25.9 MB storage freed
- Archive directory kept for audit trail

**Verification:**
```powershell
(Get-ChildItem -Path "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\" -Recurse).Count
# Should return: 0 (empty directory)
```

### Emergency Rollback (If Needed)

**ANY TIME during observation period (Jan 24 - Jan 31):**

```powershell
# Restore from safety backup (5 minutes)
Copy-Item -Path "c:\MircoHydro_CONSOLIDATION_BACKUP_20260124\*" `
          -Destination "c:\MircoHydro\" -Recurse -Force
```

---

## Tier 2: CAD, Spreadsheet, Document Duplicates (Planned)

**Target Archive Date:** February 1, 2026 (if Tier 1 approved)  
**Expected Files:** 30-50 duplicates  
**Expected Size:** ~14 MB

### Retention Timeline

| Date Range | Phase | Action | Duration |
|-----------|-------|--------|----------|
| Feb 1-14 | **Observation** | Monitor daily | **14 days** |
| Feb 15 | **Decision** | Review results | **1 day** |
| Feb 16+ | **Permanent Delete** | If approved | **Immediate** |

### Observation Rules

- Daily checks same as Tier 1
- Risk level: MEDIUM (affects reference materials)
- Longer observation: 14 days (vs. 7 for Tier 1)

### Approval Criteria

Same as Tier 1, but with 14-day observation window.

---

## Tier 3: Old Backups, Test Data (Planned)

**Target Archive Date:** February 22, 2026 (if Tier 2 approved)  
**Expected Files:** 100-150 duplicates  
**Expected Size:** ~100 MB

### Retention Timeline

| Date Range | Phase | Action | Duration |
|-----------|-------|--------|----------|
| Feb 22 - Mar 22 | **Observation** | Monitor daily | **30 days** |
| Mar 23 | **Decision** | Review results | **1 day** |
| Mar 24+ | **Permanent Delete** | If approved | **Immediate** |

### Observation Rules

- Daily checks same as Tier 1
- Risk level: HIGH (affects research/historical record)
- Longer observation: 30 days (vs. 7 or 14)
- **Community review option:** Optional team feedback
- **Executive approval:** Team lead sign-off required

### Approval Criteria

Same as Tier 1 & 2, plus:
- [ ] **Community feedback considered:** No strong objections
- [ ] **Executive approval:** Team lead approves deletion
- [ ] **30-day observation complete:** Full month of monitoring

---

## Full Backup Retention

**Location:** `c:\MircoHydro_CONSOLIDATION_BACKUP_20260124\`  
**Size:** 228.5 MB  
**Retention Period:** 90 days minimum (through April 23, 2026)

### Backup Policy

- **Keep during observation:** Yes (all 3 tiers)
- **Delete after Tier 3:** No — keep as historical archive
- **Review schedule:** Quarterly
- **Restore capability:** Always available

**If emergency rollback needed:**
```powershell
# Restore entire backup (5-10 minutes)
Copy-Item -Path "c:\MircoHydro_CONSOLIDATION_BACKUP_20260124\*" `
          -Destination "c:\MircoHydro\" -Recurse -Force
```

---

## Archive Directory Retention

**Archive directories themselves are kept permanently:**

1. `c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124\` — After files deleted, keep empty dir
2. `c:\MircoHydro\TIER2_CLEANUP_ARCHIVED_20260215\` — Same (if Tier 2 executed)
3. `c:\MircoHydro\TIER3_CLEANUP_ARCHIVED_20260222\` — Same (if Tier 3 executed)
4. `c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT_ARCHIVED_20260124\` — Keep for audit

**Purpose:** Audit trail showing what was deleted, when, and why.

---

## Monitoring and Compliance

### Daily Health Checks

**Script:** `DAILY_MONITORING_CHECK.ps1`  
**Duration:** 3-5 minutes  
**Frequency:** Once per day (morning recommended)  
**Log:** `MONITORING_CHECKLIST_7DAY.md`

### Weekly Summary

**Script:** `WEEKLY_SUMMARY_REPORT.ps1`  
**Duration:** 5 minutes  
**Frequency:** Once per week  
**Output:** Summary of 7-day trends

### Archive Integrity Checks

**Script:** `ARCHIVE_INTEGRITY_CHECK.ps1`  
**Duration:** 10 minutes  
**Frequency:** On-demand or weekly  
**Method:** SHA256 hash verification

---

## Exception Handling

### What if a check fails?

**Option 1: Investigate and Continue**
- Root cause analysis
- If fixable, resolve and continue monitoring
- Resume from same checkpoint

**Option 2: Extend Monitoring**
- Continue for 14 additional days
- Re-evaluate on new decision date
- Escalate if issues persist

**Option 3: Rollback**
- Restore from backup immediately
- Cancel deletion phase
- Plan alternative approach

---

## Escalation Procedure

**If archive file count drops unexpectedly:**
1. ⚠️ Pause all deletion operations
2. 🔄 Restore from backup
3. 📋 Investigate cause
4. 📞 Escalate to team lead
5. 🔍 Review for data loss

**If SoT corruption detected:**
1. ⚠️ Emergency rollback
2. 🔄 Restore primary SoT from backup
3. 📞 Immediate escalation
4. 🔍 Full audit of affected systems
5. 📋 Document incident

**If backup becomes inaccessible:**
1. ⚠️ Stop all deletion operations
2. 📞 Immediate IT escalation
3. 🔄 Assess restore capability
4. 📋 Do not proceed with Tier 1+ until resolved

---

## Deletion Authority

### Tier 1 (25.9 MB - Low Risk)
**Approval Required From:** Technical Lead or Dev Manager  
**Notice Period:** 1 day (team informed)  
**Reversibility:** Full backup available for 90 days

### Tier 2 (14 MB - Medium Risk)
**Approval Required From:** Technical Lead + Team Lead  
**Notice Period:** 3 days (team consulted)  
**Reversibility:** Full backup available for 90 days

### Tier 3 (100 MB - High Risk)
**Approval Required From:** Team Lead + Project Manager  
**Notice Period:** 7 days (community feedback)  
**Reversibility:** Full backup available permanently

---

## Documentation and Auditing

### Pre-Deletion Documentation
- Archive manifest created (file list + hashes)
- Backup verified and accessible
- Team notified of impending deletion
- Monitoring logs reviewed

### Post-Deletion Documentation
- Deletion command executed and logged
- Freed storage verified
- Integrity check passed
- Archive directory kept for audit

### Permanent Record
- All deleted files: Listed in archive manifest
- All dates: Recorded in monitoring logs
- All approvals: Documented in commit messages/emails
- All hashes: Stored in MANIFEST.json for verification

---

## Schedule Overview

| Phase | Archive Date | Observation | Decision | Deletion | Storage Freed |
|-------|--------------|-------------|----------|----------|---------------|
| **Tier 1** | Jan 24 | Jan 24-31 | Jan 31 | Feb 1 | 25.9 MB |
| **Tier 2** | Feb 1 | Feb 1-14 | Feb 15 | Feb 16 | +14 MB |
| **Tier 3** | Feb 22 | Feb 22-Mar22 | Mar 23 | Mar 24 | +100 MB |
| **TOTAL** | - | - | Mar 23 | Mar 24+ | **~140 MB** |

---

## Policy Review

**Version:** 1.0  
**Created:** January 24, 2026  
**Last Updated:** January 24, 2026  
**Next Review:** February 1, 2026  

**Changes in this version:**
- Initial policy creation
- Tier 1-3 retention rules
- Observation periods defined
- Approval authorities established
- Escalation procedures documented

---

## Questions & Support

- **Policy questions:** See STORAGE_STRATEGY_SUMMARY.md
- **Technical questions:** See MONITORING_SETUP.md
- **Approval process:** See PHASE_4_LONGTERM_STRATEGY_EXECUTION.md
- **Current status:** See CONSOLIDATION_SUMMARY.md

