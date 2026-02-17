# Tier 2 Overview

**Status:** 🔄 Monitoring (Day 1 of 14)  
**Archive Location:** `c:\MircoHydro\TIER2_CLEANUP_ARCHIVED_20260124\`  
**Files:** 39 duplicate CAD/spreadsheet/document files  
**Size:** 0.2 MB  
**Decision Date:** February 8, 2026

---

## What's Archived (Tier 2)

**39 duplicate files including:**

**Spreadsheets & Data Files (30+ items):**
- MicroHydroV1_RnD_Export.xlsx (multiple versions)
- Bench_Test_Tracker.xlsx
- DAQ_Logger_Config_RUN3.xlsx
- ELC_Frequency_Response.xlsx
- Measurements_Import_Template.csv
- Various test data CSV files
- T001, T002, T003, T004 data logs
- Mac metadata files (._*.xlsx, ._*.csv)

**Total:** 39 files, 0.2 MB (mostly metadata)

---

## Monitoring Schedule

**14-Day Monitoring Period:** Jan 24 - Feb 7

| Week | Dates | Status | Action |
|------|-------|--------|--------|
| Week 1 | Jan 24-30 | ⏳ In Progress | Run daily checks |
| Week 2 | Jan 31-Feb 6 | ⏳ Pending | Run daily checks |
| Day 14 | Feb 7 | ⏳ Pending | Final check |
| Day 15 | Feb 8 | ⏳ Pending | DECISION MEETING |

---

## Daily Check Command

```powershell
@(Get-ChildItem c:\MircoHydro\TIER2_CLEANUP_ARCHIVED_20260124 -Recurse -File).Count
# Should return: 39
```

---

## What We're Monitoring

✅ **Archive file count:** Should always be 39  
✅ **Archive total size:** Should always be ~0.2 MB  
✅ **SoT operational:** Should be accessible  
✅ **CAD tools functional:** No corruption issues  
✅ **Data analysis tools:** Pipeline execution normal  
✅ **Backup accessible:** Should be available  

---

## Decision Criteria (Feb 8)

**If 14 days ALL PASS:**
- ✅ File count never changed (stayed 39)
- ✅ File size never changed (stayed 0.2 MB)
- ✅ SoT operational entire time
- ✅ CAD tools working normally
- ✅ Data analysis processes normal
- ✅ No team issues reported
- ✅ Backup always accessible

**Decision:**
→ **YES:** Delete Feb 9 (0.2 MB freed)  
→ **NO:** Keep archived, extend 30 days

---

## If Approved for Deletion (Feb 9)

**Delete command:**
```powershell
Remove-Item -Path "c:\MircoHydro\TIER2_CLEANUP_ARCHIVED_20260124\*" -Recurse -Force
```

**Verify deletion:**
```powershell
@(Get-ChildItem -Path "c:\MircoHydro\TIER2_CLEANUP_ARCHIVED_20260124" -Recurse).Count
# Should return: 0
```

**Result:** 0.2 MB freed (cumulative: 5.9 MB from both tiers)

---

## Red Flags (Stop Immediately)

❌ Archive file count drops below 39  
❌ Archive file size decreases  
❌ SoT becomes inaccessible  
❌ CAD tools show errors  
❌ Pipeline execution fails  
❌ Backup becomes unavailable  

**If ANY occur:** Stop and escalate to team lead immediately.

---

## Risk Assessment

**Risk Level:** MEDIUM
- Files are reference materials (not production)
- Small size (0.2 MB) reduces impact
- 14-day observation (longer than Tier 1)
- Duplicate spreadsheets are regenerable

**Mitigation:**
- Full backup available (228.5 MB)
- CAD tools monitoring during observation
- Longer observation period
- Team feedback considered

---

## What Happens Next?

**During Tier 2 Monitoring (Jan 24 - Feb 7):**
- Tier 1 decision happens (Jan 31)
- If Tier 1 approved: Delete Feb 1
- Tier 2 continues monitoring (Feb 1-7)

**After Tier 2 Decision (Feb 8+):**
- If approved: Delete Feb 9 (0.2 MB freed)
- Start Tier 3 planning (Feb 10-21)
- Community review period (Feb 22-28)

---

## Storage Impact

| Scenario | Result |
|----------|--------|
| Phase 1 | 40-50 MB freed ✅ |
| Tier 1 (if approved) | +5.7 MB freed |
| Tier 2 (if approved) | +0.2 MB freed |
| Total (both approved) | 45.9-55.9 MB freed |

---

## Next Steps

1. **Jan 25-30:** Continue daily monitoring (Tier 1)
2. **Jan 31:** Tier 1 decision meeting
3. **Feb 1-7:** Continue daily monitoring (Tier 2)
4. **Feb 8:** Tier 2 decision meeting (10 min)
5. **Feb 9:** Delete (if approved)

---

**Questions?** See README.md or MASTER_TODO_LIST.md

