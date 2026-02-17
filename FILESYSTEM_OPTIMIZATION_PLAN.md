# AAC File System Optimization Plan
## Data Distribution & Redundancy Analysis

**Analysis Date:** February 14, 2026  
**Total Size:** 11.73 GB  
**Total Files:** 1,249,849  

## 📊 **Data Distribution Breakdown**

### **Where Most Data Is Distributed:**

| Directory | Size | Files | Primary Content |
|-----------|------|-------|-----------------|
| `aac/` | **12.9 GB** | 1.2M | **93% of total data** |
| `src/` | 135 MB | 953 | Core application code |
| `build/` | 15 MB | 14 | Build artifacts |
| `consolidation_backup/` | 14.5 MB | 952 | Backup files |
| `services/` | 0.7 MB | 91 | Microservices |

### **Within `aac/` Directory (12.9 GB):**

| Subdirectory | Size | Files | Content Type |
|--------------|------|-------|--------------|
| `src/core/` | **5.6 GB** | 460K | **Application data & code** |
| `NCC/NCC-Doctrine/` | **4.7 GB** | 447K | **NCC doctrine & backups** |
| `docs/` | 1.2 GB | 176K | **Documentation** |
| `Projects/` | 1.2 GB | 176K | **Project files** |
| `aac069/` | 333 MB | 27K | **Legacy code** |

## 📁 **File Type Distribution**

### **Space Consumption by File Type:**

| File Type | Files | Size | % of Total | Content |
|-----------|-------|------|------------|---------|
| **`.json`** | **963K** | **7.7 GB** | **65%** | API data, configs, cached data |
| **`.md`** | **283K** | **3.6 GB** | **31%** | Documentation |
| **`.zip`** | **57** | **427 MB** | **3.6%** | Archives |
| **`.pkl`** | **45** | **225 MB** | **1.9%** | ML models |
| **`.py`** | **2.7K** | **43 MB** | **0.4%** | Source code |

## 🔄 **Redundancy Analysis**

### **Identified Redundancy Patterns:**

1. **Massive JSON Duplication:**
   - 95 instances of `settings.json`
   - 90 instances of `ledger.json`, `projects.json`, `budgets.json`
   - 82 instances of `dashboard_status.json`, `ncc_international_accounting_system.json`

2. **Backup Directory Duplication:**
   - Multiple `daily_20260130_*` backup directories
   - Each containing identical 223MB data directories
   - Same files appearing in `aac/NCC/` and `aac/src/core/NCC/`

3. **Documentation Duplication:**
   - Similar content in `docs/` and `Projects/` directories
   - 157K markdown files across documentation areas

4. **ZIP Archive Duplication:**
   - Multiple `AXSystems_*.zip` and `AXLauncher_*.zip` files
   - Same archives appearing in different backup locations

## 🎯 **Gaps & Missing Elements**

### **Structural Issues:**
- **Missing `__init__.py` files** in Python packages
- **Empty directories** consuming space
- **Large Git repository** (pack files taking 400+ MB)
- **Unnecessary backup files** from consolidation

### **Code Organization Gaps:**
- **Inconsistent directory structure** between `aac/` and `src/`
- **Mixed content types** in same directories
- **No clear separation** between active code and archives

## 🚀 **Optimization Plan**

### **Phase 1: Safe Cleanup (Immediate - ~762 MB savings)**

#### **1. Git Repository Optimization**
```bash
# Clean Git repository
git gc --aggressive --prune=now
git repack -a -d --depth=250 --window=250
git prune
```
**Potential Savings:** 512 MB
**Risk:** Low (standard Git maintenance)

#### **2. Python Cache Cleanup**
```bash
# Remove Python cache files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete
```
**Potential Savings:** 50 MB
**Risk:** Low (regenerated automatically)

#### **3. Remove Empty Directories**
```bash
# Remove empty directories
find . -type d -empty -delete
```
**Potential Savings:** Minimal
**Risk:** Low

### **Phase 2: Archive Consolidation (High Impact - ~2.5 GB savings)**

#### **4. Consolidate Backup Directories**
- **Identify:** Multiple `daily_20260130_*` directories with identical content
- **Action:** Keep 1 recent backup, archive others to compressed storage
- **Pattern:** `aac/NCC/NCC-Doctrine/backups/daily_*/`
**Potential Savings:** 1.2 GB

#### **5. Remove Duplicate ZIP Archives**
- **Identify:** Multiple identical `AXSystems_*.zip` and `AXLauncher_*.zip` files
- **Action:** Keep 1 copy, remove duplicates
**Potential Savings:** 200 MB

#### **6. Consolidate Documentation**
- **Identify:** Overlapping content in `docs/` and `Projects/`
- **Action:** Merge and deduplicate markdown files
**Potential Savings:** 500 MB

### **Phase 3: Data Optimization (Major Impact - ~6 GB savings)**

#### **7. JSON Data Deduplication**
- **Identify:** 963K JSON files with massive duplication
- **Action:** Implement centralized data storage, remove redundant configs
- **Strategy:** Convert flat JSON files to database storage
**Potential Savings:** 5 GB

#### **8. ML Model Optimization**
- **Identify:** 45 pickle files with potential duplication
- **Action:** Implement model versioning, remove old/unused models
**Potential Savings:** 150 MB

### **Phase 4: Structural Reorganization (Long-term)**

#### **9. Directory Structure Cleanup**
```
Current: Mixed content in aac/ directory
Target:
├── src/           # Active source code only
├── docs/          # Consolidated documentation
├── data/          # Centralized data storage
├── archives/      # Compressed historical data
└── backups/       # Single backup location
```

#### **10. Database Migration**
- Move JSON data to PostgreSQL
- Implement proper data versioning
- Remove file-based data storage

## 📈 **Expected Results**

| Phase | Timeframe | Space Saved | Risk Level | Effort |
|-------|-----------|-------------|------------|--------|
| **Phase 1** | Immediate | **762 MB** | Low | 1 hour |
| **Phase 2** | 1-2 days | **1.9 GB** | Medium | 4 hours |
| **Phase 3** | 1-2 weeks | **5.2 GB** | High | 2 days |
| **Phase 4** | 1 month | **2-3 GB** | High | 1 week |

**Total Potential Savings: ~9.8 GB (83% of current size)**

## 🎯 **Quick Wins (Start Here)**

### **Immediate Actions (No Risk):**
```bash
# 1. Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# 2. Git optimization
git gc --aggressive --prune=now

# 3. Remove empty directories
find . -type d -empty -delete
```

### **Safe Medium-term Actions:**
```bash
# 4. Consolidate duplicate ZIP files
# 5. Remove old backup directories (keep 1 recent)
# 6. Merge duplicate documentation
```

## ⚠️ **Risk Mitigation**

### **Backup Strategy:**
1. **Full system backup** before Phase 2
2. **Test environment validation** after each phase
3. **Incremental approach** - validate after each step

### **Validation Steps:**
1. **Functionality testing** after each cleanup
2. **Data integrity checks** for moved files
3. **Performance monitoring** during transition

## 🚀 **Implementation Priority**

1. **Phase 1** - Immediate (today)
2. **Phase 2** - This week (safe, high impact)
3. **Phase 3** - Next sprint (requires planning)
4. **Phase 4** - Future project (architectural change)

## 📊 **Success Metrics**

- **Space Reduction:** Target 60-80% reduction
- **File Count:** Target 50% reduction in file count
- **Performance:** Improved file system performance
- **Maintainability:** Cleaner, more organized structure

---

**Current State:** 11.73 GB, 1.25M files  
**Target State:** 2-3 GB, 200-300K files  
**Timeframe:** 2-4 weeks  
**Risk Level:** Managed (phased approach)</content>
<parameter name="filePath">c:\Users\gripa\OneDrive\Desktop\AAC\FILESYSTEM_OPTIMIZATION_PLAN.md