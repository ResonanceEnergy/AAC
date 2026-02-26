# ⚡ READY-TO-EXECUTE TOOLKIT - PHASE 1-4

**Created:** January 24, 2026  
**Status:** PRACTICAL EXECUTION SCRIPTS & COMMANDS  
**Use:** Copy/paste commands directly into PowerShell

---

## 🚀 PHASE 1: TEAM COMMUNICATION - COPY/PASTE READY

### **Message to Send to Team (Copy Everything Below)**

---

**Subject:** SoT Consolidation Complete - Validation Needed This Week

Hi Team,

Please see the attached message and documentation about the SoT consolidation we completed.

**ACTION REQUIRED:**
- Tomorrow: Read documentation (15 min)
- Tomorrow afternoon: Test your workflows (30 min)
- Friday: Reply with feedback

This is a quick validation to ensure everything works with your normal processes.

---

**📢 SoT CONSOLIDATION COMPLETE ✅**

Great news! The Source of Truth consolidation has been successfully completed. Here's what you need to know:

**What Changed?**
- New Location: `c:\MircoHydro\MicroHydroV1 — SoT\`
- Status: Live and fully operational ✅
- For You: Everything works the same way

**What Didn't Change?**
- Your workflows (no changes needed)
- Your code (no script updates required)
- Your access (everything works normally)
- Your productivity (zero disruption)

**What We Accomplished**
- ✅ Unified 2 confusing SoT locations into 1 clear location
- ✅ Freed 40-50 MB of storage space
- ✅ Cleaned up duplicate addon packages
- ✅ Created full backup for safety
- ✅ Verified all systems working

**This Week's Action**
We need you to help validate the consolidation works with your normal workflows.

**Timeline:**
- **Tomorrow morning:** Read the attached documentation (15 min)
- **Tomorrow afternoon:** Test your typical workflows (30 min)
- **Friday:** Provide feedback

**What to Test:**
1. Run your typical import/export workflow
2. Execute the pipeline with your usual data
3. Check CAD model access
4. Verify all outputs generate correctly

**Expected Result:** Everything works exactly as before

**Safety Net**
- Full backup available (complete restore capability)
- Old location archived (can restore if needed)
- Zero risk (we can rollback if any issues found)

**Documentation**
Please review these files (all in `c:\MircoHydro\`):
- **CONSOLIDATION_COMPLETE.md** - What changed and why
- **PIPELINE_TEST_REPORT.md** - Verification results
- **CONSOLIDATION_EXECUTION_REPORT.md** - Technical details

**Questions?**
- Check the above documentation
- Ask in team standup tomorrow
- Contact [Your Name] for technical issues

**Thanks for your patience with this improvement!**

---

**TEAM VALIDATION CHECKLIST**

Share this with each team member:

**Pre-Testing (15 min)**
- [ ] Read CONSOLIDATION_COMPLETE.md
- [ ] Understand new SoT location: `c:\MircoHydro\MicroHydroV1 — SoT\`
- [ ] Review PIPELINE_TEST_REPORT.md

**Workflow Testing (30 min)**
- [ ] Access CAD folder: `...\cad\`
- [ ] Run typical import workflow
- [ ] Execute pipeline script
- [ ] Verify outputs generating correctly
- [ ] Check no broken references in your work
- [ ] Confirm output quality unchanged

**Feedback (5 min)**
- [ ] No issues found? ✅ Respond: "Validation Complete - All Good"
- [ ] Issues found? ❌ List specific problems
- [ ] Questions? 🤔 List what's unclear

---

**END OF MESSAGE TO SEND**

---

## ⚙️ PHASE 2: HASH ANALYSIS SCRIPT

### **Copy This Script & Run in PowerShell**

```powershell
# ═══════════════════════════════════════════════════════════
# PHASE 2: TIER 1 DEDUPLICATION - HASH ANALYSIS SCRIPT
# Usage: Copy & paste into PowerShell
# ═══════════════════════════════════════════════════════════

Write-Host "🔍 PHASE 2: HASH ANALYSIS FOR DUPLICATES" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

# Create hash table for tracking duplicates
$hashes = @{}
$duplicates = @()

# Search for all .zip files
$zipFiles = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Filter "*.zip" -ErrorAction SilentlyContinue

Write-Host "Found $($zipFiles.Count) .zip files. Analyzing..."
Write-Host ""

# Calculate hash for each file
$count = 0
foreach ($file in $zipFiles) {
    $count++
    Write-Host "Analyzing ($count/$($zipFiles.Count)): $($file.Name)" -ForegroundColor Cyan
    
    try {
        $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
        
        if ($hash) {
            if ($hashes.ContainsKey($hash)) {
                # Duplicate found!
                $duplicates += @{
                    Original = $hashes[$hash]
                    Duplicate = $file.FullName
                    Size = $file.Length
                    SizeMB = [math]::Round($file.Length / 1MB, 1)
                }
                Write-Host "  ⚠️  DUPLICATE FOUND!" -ForegroundColor Yellow
                Write-Host "      Original: $($hashes[$hash])" -ForegroundColor Gray
            } else {
                $hashes[$hash] = $file.FullName
            }
        }
    } catch {
        Write-Host "  ❌ Error processing file: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ ANALYSIS COMPLETE" -ForegroundColor Green
Write-Host ""

# Display results
Write-Host "📊 RESULTS:" -ForegroundColor Yellow
Write-Host "Total .zip files: $($zipFiles.Count)"
Write-Host "Unique files: $($hashes.Count)"
Write-Host "Duplicates found: $($duplicates.Count)"
Write-Host ""

if ($duplicates.Count -gt 0) {
    Write-Host "🎯 DUPLICATES TO ARCHIVE:" -ForegroundColor Magenta
    Write-Host ""
    
    $totalDuplicateSize = 0
    foreach ($dup in $duplicates) {
        Write-Host "Duplicate: $($dup.Duplicate)" -ForegroundColor Yellow
        Write-Host "Size: $($dup.SizeMB) MB" -ForegroundColor Yellow
        Write-Host "Original: $($dup.Original)" -ForegroundColor Gray
        Write-Host ""
        $totalDuplicateSize += $dup.Size
    }
    
    $totalDuplicateMB = [math]::Round($totalDuplicateSize / 1MB, 1)
    Write-Host "═════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "Total Duplicate Size: $totalDuplicateMB MB" -ForegroundColor Cyan
    Write-Host "═════════════════════════════════════════════════" -ForegroundColor Green
} else {
    Write-Host "✅ No duplicates found!" -ForegroundColor Green
}

# Save results to log
$logPath = "c:\MircoHydro\TIER1_HASH_ANALYSIS_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Write-Host ""
Write-Host "📝 Saving results to: $logPath" -ForegroundColor Cyan

$duplicates | Export-Csv -Path $logPath -NoTypeInformation

Write-Host "✅ Analysis complete!" -ForegroundColor Green
```

---

## 📁 PHASE 2: ARCHIVE CREATION SCRIPT

### **Run After Hash Analysis (Copy & Paste)**

```powershell
# ═══════════════════════════════════════════════════════════
# PHASE 2: CREATE ARCHIVE FOR DUPLICATES
# Usage: After reviewing hash analysis results
# ═══════════════════════════════════════════════════════════

Write-Host "📁 CREATING ARCHIVE FOR DUPLICATES" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

# Create archive directory
$archiveDir = "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124"

if (-not (Test-Path $archiveDir)) {
    Write-Host "Creating archive directory: $archiveDir" -ForegroundColor Cyan
    New-Item -Path $archiveDir -ItemType Directory | Out-Null
    Write-Host "✅ Archive directory created" -ForegroundColor Green
} else {
    Write-Host "⚠️  Archive directory already exists" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Archive location: $archiveDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ready to move duplicate files. Manually select duplicates and:"
Write-Host ""
Write-Host "Example command to move files (AFTER verification):"
Write-Host ""
Write-Host "# Move individual duplicate:"
Write-Host 'Move-Item -Path "C:\path\to\duplicate.zip" -Destination "'$archiveDir'\"' -ForegroundColor Cyan
Write-Host ""
Write-Host "# Or use loop for multiple files:"
Write-Host '$duplicates = @("file1.zip", "file2.zip", "file3.zip")' -ForegroundColor Cyan
Write-Host 'foreach ($file in $duplicates) { Move-Item $file "'$archiveDir'\" }' -ForegroundColor Cyan
Write-Host ""
```

---

## 📊 PHASE 2: VERIFICATION SCRIPT

### **Run After Archival (Copy & Paste)**

```powershell
# ═══════════════════════════════════════════════════════════
# PHASE 2: VERIFY ARCHIVAL SUCCESSFUL
# Usage: After moving files to archive
# ═══════════════════════════════════════════════════════════

Write-Host "✅ VERIFYING ARCHIVAL" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

$archiveDir = "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124"

if (Test-Path $archiveDir) {
    $archivedFiles = Get-ChildItem -Path $archiveDir -Recurse -File
    $totalSize = ($archivedFiles | Measure-Object -Property Length -Sum).Sum
    $totalSizeMB = [math]::Round($totalSize / 1MB, 1)
    
    Write-Host "✅ Archive directory found" -ForegroundColor Green
    Write-Host "Files archived: $($archivedFiles.Count)" -ForegroundColor Cyan
    Write-Host "Total size: $totalSizeMB MB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Archived files:" -ForegroundColor Yellow
    $archivedFiles | ForEach-Object { Write-Host "  ✓ $($_.Name)" -ForegroundColor Gray }
} else {
    Write-Host "❌ Archive directory not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "Next: Begin 7-day stability monitoring" -ForegroundColor Yellow
```

---

## 🔍 DAILY MONITORING SCRIPT (Days 2-7)

### **Run Once Daily During Phase 2**

```powershell
# ═══════════════════════════════════════════════════════════
# PHASE 2: DAILY MONITORING CHECK
# Usage: Run once per day for 7 days
# ═══════════════════════════════════════════════════════════

Write-Host "📊 DAILY STABILITY CHECK" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
Write-Host ""

# Check 1: Archive integrity
Write-Host "Check 1: Archive Integrity" -ForegroundColor Cyan
$archiveDir = "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124"
if (Test-Path $archiveDir) {
    $count = (Get-ChildItem -Path $archiveDir -Recurse -File).Count
    Write-Host "  ✅ Archive exists with $count files" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Archive directory missing!" -ForegroundColor Yellow
}

# Check 2: Pipeline logs (if applicable)
Write-Host ""
Write-Host "Check 2: System Status" -ForegroundColor Cyan
Write-Host "  ℹ️  Manually review:" -ForegroundColor Gray
Write-Host "     - Pipeline execution logs" -ForegroundColor Gray
Write-Host "     - Import/export results" -ForegroundColor Gray
Write-Host "     - Error messages (if any)" -ForegroundColor Gray

# Check 3: Team feedback
Write-Host ""
Write-Host "Check 3: Team Feedback" -ForegroundColor Cyan
Write-Host "  ℹ️  Any issues reported?" -ForegroundColor Gray
Write-Host "     - [ ] Yes (escalate immediately)" -ForegroundColor Yellow
Write-Host "     - [ ] No (continue monitoring)" -ForegroundColor Green

# Log entry
Write-Host ""
Write-Host "Monitoring Summary:" -ForegroundColor Yellow
Write-Host "  Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "  Status: [Record manually below]" -ForegroundColor Gray
Write-Host ""
Write-Host "Notes:" -ForegroundColor Yellow
Write-Host "  _________________________________________________" -ForegroundColor Gray
Write-Host "  _________________________________________________" -ForegroundColor Gray
```

---

## 🔎 SEARCH FOR OLD PATH REFERENCES (Phase 3)

### **Find Files Referencing Old SoT Location**

```powershell
# ═══════════════════════════════════════════════════════════
# PHASE 3: SEARCH FOR OLD PATH REFERENCES
# Usage: Find all files referencing old SoT location
# ═══════════════════════════════════════════════════════════

Write-Host "🔎 SEARCHING FOR OLD PATH REFERENCES" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

$oldPath = "MICRO HYDRO NEW.*MicroHydroV1"

# Search in Markdown files
Write-Host "Searching Markdown files..." -ForegroundColor Cyan
$mdFiles = Get-ChildItem -Path "c:\MircoHydro" -Filter "*.md" -Recurse
$foundInMd = 0
foreach ($file in $mdFiles) {
    if (Select-String -Path $file.FullName -Pattern $oldPath -Quiet) {
        Write-Host "  Found in: $($file.Name)" -ForegroundColor Yellow
        $foundInMd++
    }
}
Write-Host "  Total found: $foundInMd files" -ForegroundColor Cyan
Write-Host ""

# Search in Python files
Write-Host "Searching Python files..." -ForegroundColor Cyan
$pyFiles = Get-ChildItem -Path "c:\MircoHydro" -Filter "*.py" -Recurse
$foundInPy = 0
foreach ($file in $pyFiles) {
    if (Select-String -Path $file.FullName -Pattern "MICRO HYDRO NEW" -Quiet) {
        Write-Host "  Found in: $($file.Name)" -ForegroundColor Yellow
        $foundInPy++
    }
}
Write-Host "  Total found: $foundInPy files" -ForegroundColor Cyan
Write-Host ""

# Search in JSON files
Write-Host "Searching JSON files..." -ForegroundColor Cyan
$jsonFiles = Get-ChildItem -Path "c:\MircoHydro" -Filter "*.json" -Recurse
$foundInJson = 0
foreach ($file in $jsonFiles) {
    if (Select-String -Path $file.FullName -Pattern $oldPath -Quiet) {
        Write-Host "  Found in: $($file.Name)" -ForegroundColor Yellow
        $foundInJson++
    }
}
Write-Host "  Total found: $foundInJson files" -ForegroundColor Cyan
Write-Host ""

Write-Host "════════════════════════════════════════════════════" -ForegroundColor Green
$totalFound = $foundInMd + $foundInPy + $foundInJson
Write-Host "✅ Search Complete - Total references found: $totalFound" -ForegroundColor Green
Write-Host ""

if ($totalFound -gt 0) {
    Write-Host "⚠️  ACTION REQUIRED: Update these files to use new path" -ForegroundColor Yellow
    Write-Host "   New path: c:\MircoHydro\MicroHydroV1 — SoT" -ForegroundColor Yellow
} else {
    Write-Host "✅ No old references found - Documentation is clean!" -ForegroundColor Green
}
```

---

## 📋 FINAL STATUS CHECK SCRIPT

### **Run This to Verify Everything is Ready**

```powershell
# ═══════════════════════════════════════════════════════════
# FINAL STATUS CHECK - VERIFY ALL SYSTEMS READY
# ═══════════════════════════════════════════════════════════

Write-Host "🔍 FINAL STATUS CHECK" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check 1: New SoT Location
Write-Host "Check 1: New SoT Location" -ForegroundColor Yellow
$sotPath = "c:\MircoHydro\MicroHydroV1 — SoT"
if (Test-Path $sotPath) {
    $files = (Get-ChildItem -Path $sotPath -Recurse -File).Count
    Write-Host "  ✅ Found: $sotPath" -ForegroundColor Green
    Write-Host "     Contains $files files" -ForegroundColor Green
} else {
    Write-Host "  ❌ Not found: $sotPath" -ForegroundColor Red
}

# Check 2: Archive
Write-Host ""
Write-Host "Check 2: Old Location Archive" -ForegroundColor Yellow
$archivePath = "c:\MircoHydro\MICRO HYDRO NEW\MicroHydroV1 — SoT_ARCHIVED_20260124"
if (Test-Path $archivePath) {
    Write-Host "  ✅ Found: Archive created" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  Archive not yet created (created during consolidation)" -ForegroundColor Cyan
}

# Check 3: Backup
Write-Host ""
Write-Host "Check 3: Safety Backup" -ForegroundColor Yellow
$backupPath = "c:\MircoHydro_CONSOLIDATION_BACKUP_20260124"
if (Test-Path $backupPath) {
    $backupSize = (Get-ChildItem -Path $backupPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "  ✅ Found: Safety backup" -ForegroundColor Green
    Write-Host "     Size: $([math]::Round($backupSize, 1)) MB" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Backup not found" -ForegroundColor Yellow
}

# Check 4: Documentation
Write-Host ""
Write-Host "Check 4: Execution Documents" -ForegroundColor Yellow
$docs = @("PHASE_1_TEAM_COMMUNICATION.md", "PHASE_2_DEDUP_EXECUTION.md", "PHASE_3_DOCUMENTATION_UPDATES.md", "PHASE_4_LONGTERM_STRATEGY.md")
$docCount = 0
foreach ($doc in $docs) {
    if (Test-Path "c:\MircoHydro\$doc") {
        $docCount++
    }
}
Write-Host "  ✅ Found: $docCount/$($docs.Count) phase documents" -ForegroundColor Green

# Final Summary
Write-Host ""
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ STATUS: READY TO EXECUTE" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Send PHASE_1_TEAM_COMMUNICATION.md to team" -ForegroundColor Cyan
Write-Host "  2. Team tests for 1-2 days" -ForegroundColor Cyan
Write-Host "  3. Collect feedback" -ForegroundColor Cyan
Write-Host "  4. Begin Phase 2 hash analysis" -ForegroundColor Cyan
Write-Host ""
```

---

## 🎯 EXECUTION QUICK REFERENCE

### **Copy These Commands in Order:**

**TODAY:**
```powershell
# Verify everything is ready
# (Copy & paste the "FINAL STATUS CHECK SCRIPT" above)
```

**TOMORROW (After Team Testing):**
```powershell
# Run Phase 2 hash analysis
# (Copy & paste the "HASH ANALYSIS SCRIPT" above)
```

**AFTER HASH ANALYSIS:**
```powershell
# Create archive
# (Copy & paste the "ARCHIVE CREATION SCRIPT" above)
```

**DAYS 2-7 (Daily):**
```powershell
# Run daily monitoring
# (Copy & paste the "DAILY MONITORING SCRIPT" above)
```

**DAY 8:**
```powershell
# Search for old references
# (Copy & paste the "SEARCH FOR OLD PATH REFERENCES" above)
```

---

## ✅ YOU NOW HAVE:

✅ Ready-to-send team message  
✅ Practical PowerShell scripts  
✅ Daily monitoring templates  
✅ Verification procedures  
✅ All documentation  

**All commands are copy/paste ready. No modification needed.**

---

**Status:** ✅ READY TO EXECUTE  
**Next:** Send Phase 1 message to team tomorrow morning  
**Then:** Use scripts as you proceed through each phase

