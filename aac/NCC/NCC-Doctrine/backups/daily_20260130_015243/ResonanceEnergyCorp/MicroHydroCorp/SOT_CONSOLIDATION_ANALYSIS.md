# SoT CONSOLIDATION ANALYSIS & MERGE PLAN

**Analysis Date:** January 24, 2026  
**Status:** ✅ READY FOR CONSOLIDATION  
**Action Required:** MERGE Location B into Location A

---

## 🎯 EXECUTIVE SUMMARY

**Finding:** Two Source of Truth (SoT) locations exist with DIFFERENT content.

| Aspect | Location A | Location B | Status |
|--------|-----------|-----------|--------|
| **Path** | MICRO HYDRO NEW/MicroHydroV1 — SoT/ | MicroHydroV1 — SoT/ | Duplicate |
| **Core Project** | ✓ MicroHydroV1 (6 dirs) | ✓ MicroHydroV1 (6 dirs) | Same |
| **Addon Packages** | ✗ None | ✓ 15+ addons | **Different** |
| **Manifest** | ✗ Missing | ✓ MANIFEST.json | **Different** |
| **Backup** | ✓ MicroHydroV1_BACKUP_20260122_005746 | ✗ None | **Different** |
| **Size** | ~250 MB | ~300 MB | Substantial |

---

## 🔍 DETAILED COMPARISON

### Location A: MICRO HYDRO NEW/MicroHydroV1 — SoT/

**Purpose:** Contains main SoT and backup  
**Contents:**
```
MicroHydroV1 — SoT/
├── MicroHydroV1/                    ← Main project directory
│   ├── automation/                  ← Automation scripts
│   ├── cad/                        ← CAD files & macros
│   ├── docs/                       ← Documentation
│   ├── tests/                      ← Test data
│   ├── tools/                      ← Tools
│   └── README.md
│
├── MicroHydroV1_BACKUP_20260122_005746/    ← Full project backup
│   └── [Complete copy of MicroHydroV1 + tools]
│
└── Package Zip Files (7):
    ├── MicroHydroV1_FastTrack_Bundle.zip
    ├── MicroHydroV1_Phase3_Automation_Pack_macOS_OpenReleases.zip
    ├── MicroHydroV1_Phase4_SmartRuns_Pack_macOS.zip
    ├── MicroHydroV1_RC1.zip
    ├── MicroHydroV1_RunID_Migrator_v0.3.0.zip
    ├── MicroHydroV1_SoT_Package.zip
    └── [Likely duplicates of Location B content]
```

**Key Characteristics:**
- Standard directory structure (automation, cad, docs, tests, tools)
- Contains a dated backup from 2026-01-22
- Has packaged releases (7 zip files)
- Missing: MANIFEST.json (version control)
- Missing: Addon packages/tools organized as directories

**Size:** ~250 MB

---

### Location B: MicroHydroV1 — SoT/

**Purpose:** Contains main SoT with addon packages and version manifest  
**Contents:**
```
MicroHydroV1 — SoT/
├── MicroHydroV1/                    ← Main project directory
│   ├── automation/                  ← Automation scripts
│   ├── cad/                        ← CAD files & macros
│   ├── docs/                       ← Documentation
│   ├── tests/                      ← Test data
│   ├── tools/                      ← Tools
│   ├── README.md
│   │
│   ├── MANIFEST.json               ← ✓ VERSION TRACKING
│   │
│   ├── ADDON PACKAGES (organized as directories):
│   ├── MicroHydroV1_Importer_RunID_Patch_v0.3.0/
│   ├── MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0/
│   ├── MicroHydroV1_RUN3_AutoFill_Evidence_RUNID_BASELINE_v0.3.0/
│   ├── MicroHydroV1_RunID_Migrator_v0.3.0/
│   │
│   ├── ADDON PACKAGES (as zip files - 13 files):
│   ├── MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0 (1).zip
│   ├── MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0.zip
│   ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0 (1).zip
│   ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0 (2).zip
│   ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0 (3).zip
│   ├── MicroHydroV1_DO_ALL_AddOn_v0.3.0.zip
│   ├── MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_T002_v0.3.0.zip
│   ├── MicroHydroV1_RUN3_AutoFill_Evidence_v0.3.0.zip
│   ├── MicroHydroV1_RUN3_DoAll_Templates_Plots (1).zip
│   ├── MicroHydroV1_RUN3_DoAll_Templates_Plots.zip
│   ├── MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa (1).zip
│   └── MicroHydroV1_v0.3.0_FULL_LOCKED_PressurePa.zip
│
├── SharePoint Configuration:
│   └── MicroHydroV1_SharePoint_Version_Control_Configuration_Guide_v1.0.docx
```

**Key Characteristics:**
- Has MANIFEST.json (version tracking) ✓
- Organized addon packages (as both directories and zips)
- More comprehensive addon/tool packages
- Better version control documentation
- Appears to be more "mature" SoT
- No separate backup directory (has zip backups instead)

**Size:** ~300 MB

---

## 🚨 CRITICAL FINDINGS

### Finding 1: Location B is "Enhanced SoT"
Location B contains:
- ✓ MANIFEST.json (critical for version tracking)
- ✓ 15+ addon packages
- ✓ SharePoint configuration
- ✓ Better organized version control

**Interpretation:** Location B is the NEWER/ENHANCED version with more tooling and metadata.

---

### Finding 2: Location A has Backup, Location B has Archives
Location A contains:
- ✓ Dated backup: MicroHydroV1_BACKUP_20260122_005746
- ✓ Release packages (Phase 3, Phase 4, RC1)

Location B contains:
- ✓ Zip archives of releases
- ✓ Addon package archives

**Interpretation:** Location A maintains development continuity with timestamped backup. Location B has release archives.

---

### Finding 3: Duplicate Addon Packages
In Location B, many addons exist as BOTH:
- Directory form: `MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0/`
- Zip form: `MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0.zip`

**And some exist with version duplicates:**
- `MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0.zip` (original)
- `MicroHydroV1_Calibration_AutoEmbed_AddOn_v0.3.0 (1).zip` (duplicate)

**Interpretation:** Packaging is redundant - need cleanup during consolidation.

---

### Finding 4: Core Project Structure is IDENTICAL
Both locations have identical core:
- automation/
- cad/
- docs/
- tests/
- tools/
- README.md

**Interpretation:** The project files themselves need hash comparison, but structure is same.

---

## 📊 CONSOLIDATION DECISION MATRIX

| Aspect | Location A | Location B | Recommendation |
|--------|-----------|-----------|-----------------|
| **Version Control** | ✗ Missing | ✓ MANIFEST.json | **Keep B's MANIFEST** |
| **Addons/Tools** | ✗ Few | ✓ 15+ organized | **Keep B's addons** |
| **Backup Strategy** | ✓ Dated backup | ✗ Archives only | **Keep A's backup approach** |
| **Release Packages** | ✓ Phase releases | ✓ Zips | **Consolidate both** |
| **Documentation** | Basic | SharePoint config | **Keep B's config** |
| **Location** | Nested in "MICRO HYDRO NEW" | Top level | **Better: Keep B location (top-level)** |

---

## 🎯 CONSOLIDATION STRATEGY

### Recommended Approach: **MERGE A into B**

**Why:**
1. Location B has MANIFEST.json (version control)
2. Location B is top-level (easier access)
3. Location B has comprehensive addon packages
4. Location B has SharePoint integration docs
5. Location A's backup can be recreated from Location B

---

## 📋 CONSOLIDATION PLAN

### Phase 1: Analysis & Backup (Pre-Consolidation)
```
✓ Already complete:
  - Both locations identified
  - Structure analyzed
  - Differences documented

Next: Create pre-consolidation backups
```

### Phase 2: Hash & Content Verification
```
Tasks:
1. Calculate SHA256 hashes of core project files:
   - All files in MicroHydroV1/automation/
   - All files in MicroHydroV1/cad/
   - All files in MicroHydroV1/docs/
   - All files in MicroHydroV1/tests/
   - All files in MicroHydroV1/tools/
   
2. Compare Location A files vs Location B files
   - Check if core projects are identical
   - If different: Document which is "current"
   - If identical: Any one copy is sufficient

3. Check MANIFEST.json in Location B
   - Extract version information
   - Verify against file contents
```

### Phase 3: Decision Point
```
Based on hash comparison:

IF core projects are IDENTICAL:
  → Location B is authoritative (has MANIFEST)
  → Delete Location A's MicroHydroV1 core
  → Keep Location A's backup separately
  
IF core projects are DIFFERENT:
  → Document differences
  → Decide which is "current"
  → Update MANIFEST.json accordingly
```

### Phase 4: Consolidation
```
Option 1: RECOMMENDED - Keep Location B as primary
   └── Preserve:
       ├── Location B/MicroHydroV1/      ← Keep all (has MANIFEST)
       ├── Location B/MicroHydroV1_*/    ← Keep all addons
       │
       └── From Location A, keep only:
           ├── MicroHydroV1_BACKUP_20260122_005746/  ← Backup
           └── Release zip packages                  ← Releases

Option 2: Alternative - Consolidate into Location A
   └── Copy from B to A:
       ├── MANIFEST.json
       ├── All addon packages
       └── SharePoint config
       
   └── Delete Location B entirely

RECOMMENDATION: Option 1
```

### Phase 5: Cleanup
```
1. Remove duplicate addon packages:
   ├── Remove (1).zip duplicates
   ├── Remove (2).zip duplicates  
   ├── Keep one version of each addon
   
2. Organize addons structure:
   ├── Create /addons/ subdirectory
   ├── Move organized addon dirs there
   ├── Keep zip versions for distribution
   
3. Archive Location A contents:
   └── Move Location A to: MICRO HYDRO NEW/MicroHydroV1_Archive_SoT_Backup/
       (keeps for audit trail)
```

### Phase 6: Verification
```
1. Test all tooling works from new location
2. Run pipeline end-to-end
3. Verify MANIFEST.json is accurate
4. Document new structure
5. Update all references/scripts
```

---

## 📝 CRITICAL FILES TO PRESERVE

### From Location A (PRESERVE)
```
✓ MicroHydroV1_BACKUP_20260122_005746/  
  └── Reason: Timestamped backup for history

✓ Package zip files (if not in Location B)
  └── Reason: Release artifacts

✓ README files explaining the structure
  └── Reason: Documentation
```

### From Location B (PRESERVE - AUTHORITATIVE)
```
✓ MANIFEST.json
  └── Reason: Version control & metadata

✓ MicroHydroV1_Importer_RunID_Patch_v0.3.0/
✓ MicroHydroV1_RUN3_AutoFill_Evidence_PLUS_Workbook_v0.3.0/
✓ MicroHydroV1_RUN3_AutoFill_Evidence_RUNID_BASELINE_v0.3.0/
✓ MicroHydroV1_RunID_Migrator_v0.3.0/
  └── Reason: Active addon packages

✓ SharePoint configuration
  └── Reason: Integration documentation

✓ All addon zip files (consolidate duplicates)
  └── Reason: Distribution packages
```

---

## 🔐 RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Core files differ between A & B | LOW | MEDIUM | Hash verification first |
| Scripts reference Location A | MEDIUM | MEDIUM | Search & update references |
| Pipeline breaks after move | LOW | HIGH | Test in staging first |
| Loss of backup | LOW | HIGH | Keep A's backup directory |
| Addon packages conflict | LOW | MEDIUM | Deduplicate carefully |

---

## ✅ CONSOLIDATION READINESS CHECKLIST

Before starting consolidation:

- [ ] Read and understand this plan
- [ ] Create full backup: `c:\MircoHydro_PRE_CONSOLIDATION_20260124/`
- [ ] Hash all core files in both locations
- [ ] Document which core is "current"
- [ ] Identify all scripts that reference Location A
- [ ] Test Location B separately in staging
- [ ] Get team approval on this plan

---

## 📋 NEXT IMMEDIATE STEPS

### Step 1: Hash Verification (15 minutes)
Create comprehensive hash comparison of both core projects

### Step 2: File Analysis (30 minutes)  
Analyze MANIFEST.json, identify addon duplicates

### Step 3: Script References (20 minutes)
Search workspace for scripts/configs referencing Location A

### Step 4: Create Detailed Merge Plan (30 minutes)
Based on findings, create step-by-step merge instructions

### Step 5: Present to Team (30 minutes)
Show findings, get approval on consolidation approach

---

## 🎯 FINAL RECOMMENDATION

**Consolidate Location A into Location B:**

1. **Keep Location B as primary SoT**
   - Has MANIFEST.json (version control)
   - Top-level location (better accessibility)
   - More comprehensive tooling
   - Better organized

2. **Archive Location A contents**
   - Move to: `MICRO HYDRO NEW/MicroHydroV1_Archive_SoT_Backup/`
   - Keep for audit trail and backup continuity
   - Reduce to: Only keep MicroHydroV1_BACKUP_20260122_005746

3. **Clean up duplicates**
   - Remove addon package duplicates ((1).zip, (2).zip variations)
   - Organize addons directory
   - Archive zip releases properly

4. **Update all references**
   - Search for scripts/configs using Location A path
   - Update to use Location B path
   - Test thoroughly

5. **Create new documentation**
   - Update SOT structure docs
   - Add migration notes
   - Document new location

---

**Status:** ✅ Ready for detailed hash verification and team approval

**Next:** Proceed with Phase 2 (Hash & Content Verification)

