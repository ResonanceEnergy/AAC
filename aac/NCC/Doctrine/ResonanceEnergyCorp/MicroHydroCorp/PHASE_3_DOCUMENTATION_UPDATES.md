# 📝 PHASE 3: DOCUMENTATION & REFERENCE UPDATES

**Status:** READY TO EXECUTE (Parallel with Phase 2)  
**Date:** January 24, 2026  
**Duration:** 1-2 days  
**Priority:** 🟡 MEDIUM

---

## 🎯 OBJECTIVE
Update internal documentation to reflect the new SoT location and remove references to deprecated paths.

---

## 📋 TASK 1: SEARCH FOR OLD PATH REFERENCES

### **Objective:** Find all files referencing old SoT location

### **Old Path to Search For:**
```
MICRO HYDRO NEW\MicroHydroV1 — SoT
c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT
```

### **Execution:**

**Search in all documentation:**
```powershell
$oldPaths = @(
    "MICRO HYDRO NEW.*MicroHydroV1 — SoT",
    "MICRO HYDRO NEW.*MicroHydroV1.*SoT",
    "MicroHydroV1 — SoT.*MICRO HYDRO"
)

Get-ChildItem -Path "c:\MircoHydro" -Filter "*.md" -Recurse | 
  ForEach-Object {
    foreach ($pattern in $oldPaths) {
      if (Select-String -Path $_.FullName -Pattern $pattern) {
        Write-Host "Found in: $($_.FullName)"
      }
    }
  }
```

**Search in active code:**
```powershell
Get-ChildItem -Path "c:\MircoHydro" -Filter "*.py" -Recurse | 
  ForEach-Object {
    if (Select-String -Path $_.FullName -Pattern "MICRO HYDRO NEW") {
      Write-Host "Found in: $($_.FullName)"
    }
  }
```

### **Expected Results:**
- Documentation files: 5-10 references (can update)
- Active code: 0-1 references (should be minimal)
- Archive files: Many (leave as-is)

---

## 📋 TASK 2: UPDATE ACTIVE REFERENCES

### **New SoT Location to Use:**
```
c:\MircoHydro\MicroHydroV1 — SoT
MicroHydroV1 — SoT (for relative references)
```

### **Files to Update (Examples)**

**Priority 1: Active Documentation**
- [ ] README.md (if references old SoT)
- [ ] CONSOLIDATION_* files (verify paths)
- [ ] Any team procedures (if reference old paths)

**Priority 2: Configuration Files**
- [ ] Import_Config.json (check paths)
- [ ] Python scripts (verify references)
- [ ] Batch files (if any exist)

### **Update Process:**

**For Markdown Files:**
```
OLD: c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT
NEW: c:\MircoHydro\MicroHydroV1 — SoT

OLD: ..\MICRO HYDRO NEW\MicroHydroV1 — SoT\
NEW: ..\MicroHydroV1 — SoT\
```

**For Python Files:**
```
OLD: r"c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT"
NEW: r"c:\MircoHydro\MicroHydroV1 — SoT"
```

---

## 📋 TASK 3: CREATE TEAM REFERENCE GUIDE

### **Objective:** Help team understand new location and structure

### **Content to Create:**

**File:** `TEAM_REFERENCE_GUIDE.md`

```markdown
# 🗂️ TEAM REFERENCE GUIDE - SoT Consolidation

## Quick Reference

### **New SoT Location**
**Path:** `c:\MircoHydro\MicroHydroV1 — SoT\`

### **What's Inside**
- `automation/` - Import tools and configuration
- `cad/` - CAD models and designs
- `docs/` - Project documentation
- `tests/` - Test data and results
- `tools/` - Build and pipeline scripts
- `MANIFEST.json` - Version control

### **Common Paths**

| What | Path |
|------|------|
| CAD Files | `...\MicroHydroV1 — SoT\cad\` |
| Configuration | `...\automation\Import_Config.json` |
| Pipeline Script | `...\tools\run_pipeline.py` |
| Documentation | `...\docs\` |
| Test Data | `...\tests\` |

### **What Changed**
- ✅ Location unified (was 2 locations, now 1)
- ✅ Old location archived (preserved but not active)
- ✅ Scripts still work (relative paths)
- ✅ Your workflows unchanged

### **What Didn't Change**
- ✅ Your code (no updates needed)
- ✅ Your access (everything the same)
- ✅ Your productivity (zero impact)

### **Frequently Asked Questions**

**Q: Where's the old location?**
A: Archived at `c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT_ARCHIVED_20260124\` for audit trail

**Q: Do I need to update my code?**
A: No, scripts use relative paths. They still work.

**Q: What if something breaks?**
A: Report immediately. We have full backup and can restore.

**Q: How do I access the new location?**
A: Same as before: `c:\MircoHydro\MicroHydroV1 — SoT\`

**Q: Can we still access the old location?**
A: Yes, it's archived if needed for reference.
```

---

## 📋 PHASE 3 EXECUTION CHECKLIST

### **Task 1: Search for Old References (1 hour)**
- [ ] Search documentation for "MICRO HYDRO NEW"
- [ ] Search code for old SoT references
- [ ] Document all findings
- [ ] Identify files that need updates
- [ ] Create update list

### **Task 2: Update Active References (1-2 hours)**
- [ ] Update README.md (if needed)
- [ ] Update CONSOLIDATION_* docs (verify paths correct)
- [ ] Update any procedures (if they reference old paths)
- [ ] Update Python scripts (if any references)
- [ ] Update JSON configs (if needed)
- [ ] Test updated files (verify still work)

### **Task 3: Create Reference Guide (30 min)**
- [ ] Create TEAM_REFERENCE_GUIDE.md
- [ ] Include common paths
- [ ] Add FAQ section
- [ ] Include before/after comparison
- [ ] Share with team

### **Documentation Complete (15 min)**
- [ ] Mark all updates complete
- [ ] Archive old reference files
- [ ] Create completion log
- [ ] Notify team of updates

---

## 📊 EXPECTED OUTCOMES

| Item | Before | After |
|------|--------|-------|
| Documentation | May reference old path | References new path ✓ |
| Team clarity | "Which SoT?" | "Use MicroHydroV1 — SoT" ✓ |
| Active code | Working with relative | Still working ✓ |
| Team reference | Generic docs | Specific guide ✓ |

---

## 🎯 NEXT PHASE (After Completion)

Once Phase 3 complete:
- **Phase 4:** Archive retention policy
- **Optional:** Tier 2 deduplication planning

---

**Status:** READY TO EXECUTE  
**Duration:** 2-4 hours spread over 1-2 days  
**Parallel Execution:** Can run alongside Phase 2

