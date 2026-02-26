# MicroHydro File Structure - Deduplication & Organization Guide

**Audit Date:** 2026-01-24  
**Analysis:** 772 files, 289 duplicate groups, 0.22 GB total  
**Status:** Complete audit with remediation roadmap

---

## CRITICAL ISSUES REQUIRING ACTION

### 🔴 Issue #1: REDUNDANT SoT COPIES (Immediate Priority)

**Problem:** SoT directory exists in TWO locations
```
Location A: MICRO HYDRO NEW/MicroHydroV1 — SoT/MicroHydroV1/
Location B: MicroHydroV1 — SoT/MicroHydroV1/
```

**Impact:**
- Tooling duplicated (opt_config.json in both, 6x copies total)
- Same files exist in 4-5 places when including Work/ copies
- Unclear which version is authoritative

**Recommendation:**
```
KEEP:    MICRO HYDRO NEW/MicroHydroV1 — SoT/  (Location A - Primary)
DELETE:  MicroHydroV1 — SoT/                  (Location B - Remove entire directory)
```

**Files Affected:** opt_config.json, optimize_params.py, run_pipeline.py, validate_repo.py, policy_config.json, etc.

---

### 🔴 Issue #2: THREE-LEVEL CIRCULAR REDUNDANCY

**Problem:** MicroHydroV1_Work contains copies of entire SoT structure

```
Level 0 (Original):
└── MICRO HYDRO NEW/MicroHydroV1 — SoT/MicroHydroV1/

Level 1 (First Copy):
└── MicroHydroV1_Work/MasterIntegrated_Output/01_SharePoint_Libraries/
    MicroHydroV1 — SoT/MicroHydroV1/

Level 2 (Second Copy):
└── MicroHydroV1_Work/MasterIntegrated_Output/02_Canonical_Working_Tree/
    MicroHydroV1/

Level 3 (Third Copy):
└── MicroHydroV1_Work/MasterIntegrated_Output/_stage/top/
    MicroHydroV1 — SoT/MicroHydroV1/
```

**Impact:** 
- 100+ MB of redundant storage
- Risk of out-of-sync versions
- Maintenance nightmare

**Recommendation:**
Replace all copies with **symbolic links** to primary:

```powershell
# INSTEAD OF COPIES, use:
mklink /D "MicroHydroV1_Work/01_SharePoint_Libraries/MicroHydroV1 — SoT" `
        "..\..\..\..\MICRO HYDRO NEW\MicroHydroV1 — SoT"

mklink /D "MicroHydroV1_Work/02_Canonical_Working_Tree/MicroHydroV1" `
        "..\..\..\MICRO HYDRO NEW\MicroHydroV1"

mklink /D "MicroHydroV1_Work/_stage/top/MicroHydroV1 — SoT" `
        "..\..\..\..\MICRO HYDRO NEW\MicroHydroV1 — SoT"
```

---

### 🔴 Issue #3: ACTIVE BACKUP AMBIGUITY

**Problem:** Recent backup marked as historical but may still be in use

```
MicroHydroV1_BACKUP_20260122_005746/
    └── Created: 2026-01-22 (just 2 days before report date!)
    └── Size: ~220 MB (nearly identical to current)
    └── Contains: Full project with same tools, docs, tests
```

**Critical Question:** Is this backup ACTIVE or HISTORICAL?

**Recommendation:**
1. Clarify status with development team
2. If **ACTIVE:** Move to primary development location
3. If **HISTORICAL:** 
   - Move to `MICRO HYDRO NEW/MicroHydroV1_Backups/2026-01-22/`
   - Create `BACKUP_MANIFEST.json` with metadata
   - Document why backup was needed

**Implications for Next Actions:** Cannot proceed with deduplication until this is clarified!

---

### 🔴 Issue #4: MASTER PRODUCTION PACKAGE VERSIONING

**Problem:** Two complete production packages, unclear which is current

```
Package v1: MASTER_PRODUCTION_PACKAGE/
├── Status: Original package
├── Size: ~150 MB
└── Contains: Design phases 01-14, CAD archives, documentation

Package v2: MASTER_PRODUCTION_PACKAGE 2/
├── Status: "2" implies newer
├── Size: ~155 MB
└── Contains: Same structure, some file differences
```

**Key Differences Found:**
- `params.json` - Different versions (CAD evolution)
- `Nozzle_PreSwirl_BOM.xlsx` - Different content
- `Nozzle_Profile_Revolve.dxf` - Different versions
- Others appear identical

**Recommendation:**
1. Create `MASTER_PRODUCTION_PACKAGE_v1.0/` (original)
2. Rename `MASTER_PRODUCTION_PACKAGE 2` → `MASTER_PRODUCTION_PACKAGE_v2.0/`
3. Create version manifest:

```json
{
  "v1.0": {
    "created": "YYYY-MM-DD",
    "description": "Original production package",
    "key_changes": []
  },
  "v2.0": {
    "created": "YYYY-MM-DD",
    "description": "Updated production package",
    "key_changes": [
      "Updated CAD parameters (params.json)",
      "Refined nozzle design (Nozzle_PreSwirl_BOM.xlsx)",
      "etc..."
    ]
  }
}
```

---

### 🔴 Issue #5: CAD MODEL VERSION AMBIGUITY

**Problem:** Two Shapr3D reference models with unclear relationship

```
v1: MicroHydroV1_Shapr3D_ReferenceModel/
    ├── params.json
    └── DXF/ (contains original DXF files)

v2: MicroHydroV1_Shapr3D_ReferenceModel_v2/
    ├── params.json (different content)
    ├── DXF_R12/ (R12 format DXF files)
    └── STL_Reference/ (added 3D print models)
```

**Content Comparison:**
- Some DXF files IDENTICAL: VanePack_Sketch_10deg.dxf, etc.
- Some files DIFFERENT: Nozzle_Profile_Revolve.dxf, RetainerCap_Sketch.dxf
- v2 adds: STL Reference models

**Recommendation:**
1. Document v2 as **current** reference
2. Create CAD_DESIGN_EVOLUTION.md explaining changes
3. Archive v1 with clear deprecation notice
4. Add version identifier to all model files

```
Directory Structure:
MicroHydroV1_CAD_MODELS/
├── CURRENT/ → MicroHydroV1_Shapr3D_ReferenceModel_v2/
├── v1_DEPRECATED/ → MicroHydroV1_Shapr3D_ReferenceModel/
├── DESIGN_EVOLUTION.md
└── CAD_VERSION_MANIFEST.json
```

---

## SAFE DEDUPLICATION ACTIONS (114 Identical Groups)

These files are IDENTICAL copies - can be safely removed, keeping ONE master:

### Tier 1: Eliminate Immediately (Critical Configuration Files)

**Files:** High-risk if duplicated

| File | Current Copies | Keep Where | Delete From |
|------|----------------|------------|-------------|
| `opt_config.json` | 6 | `tools/optimize/` | All 5 other copies |
| `optimize_params.py` | 6 | `tools/optimize/` | All 5 other copies |
| `run_pipeline.py` | 6 | `tools/` | 5 copies in Work/ |
| `validate_repo.py` | 6 | `tools/validate/` | All 5 other copies |
| `policy_config.json` | 2 | `tools/` | `Work/05_Tools/` |

**Storage Recovery:** ~50 MB

**Risk:** LOW - Create backup before deletion

---

### Tier 2: Consolidate (Test Data & Templates)

**Files:** Identical test data files in multiple test runs

| File | Copies | Consolidation Strategy |
|------|--------|------------------------|
| `T001_JetCoherence_Run1.csv` | 8 | Keep in `/data/reference/`, symlink from test runs |
| `T002_TankRipple_Run1.csv` | 8 | Keep in `/data/reference/`, symlink from test runs |
| `T003_ELCStability_Run1.csv` | 8 | Keep in `/data/reference/`, symlink from test runs |
| `T004_Power_Run1.csv` | 8 | Keep in `/data/reference/`, symlink from test runs |
| `PT_Calibration_*.xlsx` | 2 | Keep in `/docs/templates/`, symlink in Work/ |
| `T002_TankRipple_TimeSeries_Template.csv` | 2 | Keep in `/docs/templates/`, symlink in Work/ |

**Storage Recovery:** ~80 MB

**Risk:** LOW - Verify symlinks work before removal

---

### Tier 3: Archive (Documentation & Reports)

**Files:** Identical across deployment stages

| File | Copies | Action |
|------|--------|--------|
| `PowerAutomate_Flows_SoT_to_Release.md` | 2 | Keep SoT version, delete Work copy |
| `README_Calibration_and_AutoEmbed.md` | 2 | Keep SoT version, delete Work copy |
| `README_DO_ALL.md` | 2 | Keep SoT version, delete Work copy |
| `RUN3_Evidence_Template.md` | 2 | Keep SoT version, delete Work copy |
| `RUN3_Operator_Card.md` | 2 | Keep SoT version, delete Work copy |
| `run_closed_loop.py` | 2 | Keep SoT version, delete Work copy |

**Storage Recovery:** ~30 MB

**Risk:** MEDIUM - Verify no external references before deletion

---

## PRESERVE (Must Keep - Different Versions)

These files have DIFFERENT content across locations - preserve all for audit trail:

### Category A: Configuration Evolution (24 copies of params.json)

```
params.json appears in:
1. CAD Archive/CAD/params.json
2. CAD Archive/MicroHydroV1_CAD/params.json
3. CAD Archive/MicroHydroV1_CAD_FIXED/params.json
4. CAD Archive/MicroHydroV1_FULL_Workspace/CAD/Params/params.json
5. MASTER_PRODUCTION_PACKAGE/params.json
6. MASTER_PRODUCTION_PACKAGE 2/params.json
7. MicroHydroV1/cad/params/params.json
8. MicroHydroV1 — SoT/cad/macros/params.json
9. MicroHydroV1 — SoT/cad/params/params.json
10-24. Various copies in Work/ deployment stages
```

**Action:** 
- Create `CAD_PARAMS_EVOLUTION.md` documenting changes
- Store hashes in audit file for version tracking
- Do NOT delete - represents design iteration

**Audit Trail Recommendation:**
```
docs/version_history/
├── params_json_evolution.md
├── params_hash_v1.txt (hash from original)
├── params_hash_v2.txt (hash from revision 1)
└── ... (additional versions)
```

---

### Category B: Document Revisions (19 copies of README.md)

**Locations with DIFFERENT content:**
- MicroHydroV1/README.md
- MicroHydroV1/automation/README.md
- MicroHydroV1/cad/README.md
- MicroHydroV1/data/README.md
- MicroHydroV1/docs/README.md
- MicroHydroV1/tests/README.md
- MicroHydroV1 — SoT versions (some differ from main)
- Various Work/ deployment versions

**Action:** Keep all - represents documentation evolution
**Preserve:** Version dates in README headers

---

### Category C: Approval Document Evolution (7 versions PASS_A_FULL_RevA.docx)

```
PASS_A_FULL_RevA.docx versions found in:
├── MASTER_PRODUCTION_PACKAGE/14_Reports/
├── MASTER_PRODUCTION_PACKAGE 2/14_Reports/
├── MicroHydroV1 — SoT/docs/
├── MicroHydroV1 — SoT/BACKUP/docs/
└── Work/ deployment stages (3 more)
```

**Status:** Different content = different approval versions
**Action:** Keep all for compliance/audit
**Metadata:** Add version dates to filename:
- `PASS_A_FULL_RevA_2026-01-20.docx`
- `PASS_A_FULL_RevA_2026-01-22.docx`

---

### Category D: Procedure Evolution (6 versions Tank_Ripple_Procedure.docx)

**Evolution Track:**
1. Original procedure (MASTER_PRODUCTION_PACKAGE v1)
2. Updated procedure (MASTER_PRODUCTION_PACKAGE v2)
3. SoT version (potentially updated)
4. Deployment versions (Work staging)

**Action:** Create procedure change log

```
PROCEDURE_REVISIONS.md
- Rev A (date): Original procedure
- Rev B (date): Updated sensor mounting guidance
- Rev C (date): Changed calibration sequence
```

---

## ORGANIZATION ROADMAP

### Phase 1: Assessment (COMPLETE ✓)
- [x] Analyzed 772 files
- [x] Identified 289 duplicate groups
- [x] Classified: 114 identical, 175 different
- [x] Created audit reports

### Phase 2: Immediate Actions (0-3 Days)
- [ ] **CLARIFY:** Which SoT is authoritative? (Location A or B?)
- [ ] **CLARIFY:** Is backup still active?
- [ ] **CLARIFY:** Which MASTER_PRODUCTION_PACKAGE version is current?
- [ ] Create decision log

### Phase 3: Quick Wins (3-7 Days)
- [ ] Delete second SoT location (if confirmed redundant)
- [ ] Delete 114 identical copies (with backup)
- [ ] Storage recovered: ~150-200 MB
- [ ] Document baseline metrics

### Phase 4: Restructure (1-2 Weeks)
- [ ] Reorganize to canonical structure
- [ ] Implement symlinks for deployments
- [ ] Replace copies in Work/ with links
- [ ] Verify all links functional

### Phase 5: Version Control (2-3 Weeks)
- [ ] Add MANIFEST.json to all versioned items
- [ ] Document CAD evolution
- [ ] Document package versions
- [ ] Create version tracking spreadsheet

### Phase 6: Audit & Verify (Ongoing)
- [ ] Test pipeline execution
- [ ] Verify deployments work with new structure
- [ ] Document any issues
- [ ] Schedule quarterly audits

---

## RECOMMENDED COMMAND SEQUENCE

**Execute in order (with backups!)**

### Step 1: Backup Everything
```powershell
Copy-Item -Path "c:\MircoHydro" -Destination "c:\MircoHydro_BACKUP_20260124" -Recurse
```

### Step 2: Remove Redundant SoT (if confirmed)
```powershell
Remove-Item -Path "c:\MircoHydro\MicroHydroV1 — SoT" -Recurse -Force
# After confirming MICRO HYDRO NEW/MicroHydroV1 — SoT is primary
```

### Step 3: Replace Work Copies with Symlinks
```powershell
# Remove old copies
Remove-Item -Path "c:\MircoHydro\MicroHydroV1_Work\MasterIntegrated_Output\01_SharePoint_Libraries\MicroHydroV1 — SoT" -Recurse -Force
Remove-Item -Path "c:\MircoHydro\MicroHydroV1_Work\MasterIntegrated_Output\02_Canonical_Working_Tree\MicroHydroV1" -Recurse -Force
Remove-Item -Path "c:\MircoHydro\MicroHydroV1_Work\MasterIntegrated_Output\_stage\top\MicroHydroV1 — SoT" -Recurse -Force

# Create symlinks
New-Item -ItemType SymbolicLink -Path "c:\MircoHydro\MicroHydroV1_Work\01_SharePoint_Libraries" -Name "MicroHydroV1 — SoT" -Target "..\..\MICRO HYDRO NEW\MicroHydroV1 — SoT"
New-Item -ItemType SymbolicLink -Path "c:\MircoHydro\MicroHydroV1_Work\02_Canonical_Working_Tree" -Name "MicroHydroV1" -Target "..\..\MICRO HYDRO NEW\MicroHydroV1"
New-Item -ItemType SymbolicLink -Path "c:\MircoHydro\MicroHydroV1_Work\_stage\top" -Name "MicroHydroV1 — SoT" -Target "..\..\..\..\MICRO HYDRO NEW\MicroHydroV1 — SoT"
```

### Step 4: Delete Identical Copies
```powershell
# Delete redundant tool files (kept only in tools/)
Remove-Item -Path "c:\MircoHydro\...\opt_config.json" -Exclude "c:\MircoHydro\MicroHydroV1\tools\optimize\opt_config.json"

# Similar for other identical files...
```

---

## VERIFICATION CHECKLIST

After reorganization, verify:

- [ ] SoT directory is unique and authoritative
- [ ] All deployment links point to correct locations
- [ ] Pipeline execution works end-to-end
- [ ] All test data accessible
- [ ] Documentation builds without errors
- [ ] CAD macros load correctly
- [ ] Backup location confirmed and labeled
- [ ] Version evolution documented

---

## STORAGE IMPACT SUMMARY

| Action | Current | After | Savings |
|--------|---------|-------|---------|
| **Remove redundant SoT** | 220 MB | 110 MB | 110 MB |
| **Replace copies with symlinks** | 110 MB | 30 MB | 80 MB |
| **Delete identical configs** | 30 MB | 5 MB | 25 MB |
| **Consolidate test data** | 5 MB | 2 MB | 3 MB |
| **TOTAL POTENTIAL SAVINGS** | **220 MB** | **~50-70 MB** | **~150 MB** |

**Note:** Preserves all DIFFERENT versions for audit trail

---

## QUESTIONS REQUIRING TEAM INPUT

1. **Which SoT is authoritative?**
   - Location A: `MICRO HYDRO NEW/MicroHydroV1 — SoT/`
   - Location B: `MicroHydroV1 — SoT/`

2. **Is the backup still active?**
   - `MicroHydroV1_BACKUP_20260122_005746/`
   - If yes: Move to active development
   - If no: Move to archive

3. **Which MASTER_PRODUCTION_PACKAGE is current?**
   - v1: `MASTER_PRODUCTION_PACKAGE/`
   - v2: `MASTER_PRODUCTION_PACKAGE 2/`

4. **Are Work/ deployment stages still needed?**
   - If yes: Convert to symlinks
   - If no: Archive entirely

5. **What's the CAD model status?**
   - Is v2 the current reference?
   - Should v1 be archived?

---

**Report prepared by:** Automated Audit System  
**Date:** 2026-01-24  
**Next review:** 2026-02-24 (monthly audit)

