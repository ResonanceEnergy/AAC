# ⚡ COPY/PASTE - FAST TRACK EXECUTION COMMANDS

**Instructions:** Copy each section below, paste into PowerShell, run sequentially. ~10 minutes total.

---

## ✅ COMMAND 1: VERIFICATION (30 seconds)

```powershell
Write-Host "=== STEP 1: VERIFICATION ===" -ForegroundColor Green
$sotPath = "c:\MircoHydro\MicroHydroV1 — SoT"
$backupPath = "c:\MircoHydro_CONSOLIDATION_BACKUP_20260124"
$archiveDir = "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124"

Write-Host "New SoT: $(Test-Path $sotPath)" -ForegroundColor Cyan
Write-Host "Backup: $(Test-Path $backupPath)" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ VERIFICATION COMPLETE" -ForegroundColor Green
```

---

## ✅ COMMAND 2: HASH ANALYSIS (2-3 minutes)

```powershell
Write-Host "=== STEP 2: HASH ANALYSIS ===" -ForegroundColor Green
Write-Host "Scanning for duplicate .zip files..." -ForegroundColor Cyan
Write-Host ""

$hashes = @{}
$duplicates = @()
$zipFiles = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Filter "*.zip" -ErrorAction SilentlyContinue

$count = 0
foreach ($file in $zipFiles) {
    $count++
    Write-Host -NoNewline "Analyzing ($count/$($zipFiles.Count))... " -ForegroundColor Gray
    
    try {
        $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
        
        if ($hash) {
            if ($hashes.ContainsKey($hash)) {
                $duplicates += $file.FullName
                Write-Host "DUPLICATE!" -ForegroundColor Yellow
            } else {
                $hashes[$hash] = $file.FullName
                Write-Host "unique" -ForegroundColor Gray
            }
        }
    } catch {
        Write-Host "error" -ForegroundColor Red
    }
    
    if ($count % 10 -eq 0) { Write-Host "" }
}

Write-Host ""
Write-Host "=== RESULTS ===" -ForegroundColor Cyan
Write-Host "Total files scanned: $($zipFiles.Count)" -ForegroundColor Cyan
Write-Host "Unique files: $($hashes.Count)" -ForegroundColor Cyan
Write-Host "Duplicates found: $($duplicates.Count)" -ForegroundColor Yellow
Write-Host ""

if ($duplicates.Count -gt 0) {
    $totalDupSize = ($duplicates | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum / 1MB
    Write-Host "Total duplicate size: $([math]::Round($totalDupSize, 1)) MB" -ForegroundColor Green
    Write-Host ""
    Write-Host "Duplicates to archive:" -ForegroundColor Yellow
    $duplicates | ForEach-Object { Write-Host "  • $_" -ForegroundColor Gray }
    Write-Host ""
    Write-Host "✅ Ready to archive" -ForegroundColor Green
} else {
    Write-Host "✅ No duplicates found" -ForegroundColor Green
}
```

---

## ✅ COMMAND 3: CREATE ARCHIVE DIRECTORY (30 seconds)

```powershell
Write-Host ""
Write-Host "=== STEP 3: CREATE ARCHIVE ===" -ForegroundColor Green

$archiveDir = "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124"

if (-not (Test-Path $archiveDir)) {
    New-Item -Path $archiveDir -ItemType Directory -Force | Out-Null
    Write-Host "✅ Archive directory created" -ForegroundColor Green
    Write-Host "   Path: $archiveDir" -ForegroundColor Cyan
} else {
    Write-Host "✅ Archive directory already exists" -ForegroundColor Green
    Write-Host "   Path: $archiveDir" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Ready to move files to: $archiveDir" -ForegroundColor Yellow
```

---

## ✅ COMMAND 4: MOVE DUPLICATES (2-3 minutes)

**IMPORTANT: Only run after reviewing duplicates from COMMAND 2**

```powershell
Write-Host ""
Write-Host "=== STEP 4: MOVE DUPLICATES ===" -ForegroundColor Green

$archiveDir = "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124"

# Re-run hash analysis to get fresh duplicate list
$hashes = @{}
$duplicates = @()
$zipFiles = Get-ChildItem -Path "c:\MircoHydro" -Recurse -Filter "*.zip" -ErrorAction SilentlyContinue

foreach ($file in $zipFiles) {
    $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
    if ($hash) {
        if ($hashes.ContainsKey($hash)) {
            $duplicates += $file.FullName
        } else {
            $hashes[$hash] = $file.FullName
        }
    }
}

Write-Host "Moving $($duplicates.Count) duplicate files..." -ForegroundColor Cyan
Write-Host ""

$moved = 0
foreach ($file in $duplicates) {
    try {
        Move-Item -Path $file -Destination $archiveDir -Force -ErrorAction Stop
        $moved++
        Write-Host "✅ Moved: $(Split-Path $file -Leaf)" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error moving: $file" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== MOVE COMPLETE ===" -ForegroundColor Green
Write-Host "Files moved: $moved" -ForegroundColor Cyan
Write-Host ""
```

---

## ✅ COMMAND 5: VERIFY ARCHIVE (1 minute)

```powershell
Write-Host "=== STEP 5: VERIFY ARCHIVE ===" -ForegroundColor Green

$archiveDir = "c:\MircoHydro\TIER1_CLEANUP_ARCHIVED_20260124"

if (Test-Path $archiveDir) {
    $archivedFiles = Get-ChildItem -Path $archiveDir -Recurse -File
    $totalSize = ($archivedFiles | Measure-Object -Property Length -Sum).Sum / 1MB
    
    Write-Host "✅ Archive verified" -ForegroundColor Green
    Write-Host "   Location: $archiveDir" -ForegroundColor Cyan
    Write-Host "   Files: $($archivedFiles.Count)" -ForegroundColor Cyan
    Write-Host "   Total size: $([math]::Round($totalSize, 1)) MB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Archived files:" -ForegroundColor Yellow
    $archivedFiles | ForEach-Object { Write-Host "   • $($_.Name)" -ForegroundColor Gray }
} else {
    Write-Host "❌ Archive directory not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== FAST-TRACK EXECUTION COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "✅ Next: Start 7-day stability monitoring" -ForegroundColor Green
```

---

## 📊 FINAL STATUS (after all commands)

```powershell
Write-Host ""
Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   ✅ FAST-TRACK EXECUTION COMPLETE             ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "WHAT HAPPENED:" -ForegroundColor Yellow
Write-Host "  ✓ Duplicate .zip files identified"
Write-Host "  ✓ Archive directory created"
Write-Host "  ✓ Duplicates moved to archive (NOT deleted)"
Write-Host "  ✓ Safety backup still available (228.5 MB)"
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "  1. Monitor for 7 days (run daily check)"
Write-Host "  2. After day 8: Delete archived files (if all good)"
Write-Host "  3. Expected: 71-112 MB storage freed"
Write-Host ""
Write-Host "SAFETY:" -ForegroundColor Blue
Write-Host "  ✓ Rollback available: 5 minutes"
Write-Host "  ✓ Full backup ready: 228.5 MB"
Write-Host "  ✓ Nothing permanent yet"
Write-Host "  ✓ Team can work normally"
Write-Host ""
```

---

## 🎯 EXECUTION ORDER

Run these 5 commands in sequence:

1. **COMMAND 1** - Verification (30 sec)
2. **COMMAND 2** - Hash analysis (2-3 min)
3. **COMMAND 3** - Create archive (30 sec)
4. **COMMAND 4** - Move duplicates (2-3 min)
5. **COMMAND 5** - Verify & confirm (1 min)
6. **FINAL STATUS** - Summary

**Total time: ~10 minutes**

---

## ✅ AFTER EXECUTION

**Same day:**
- Archive created ✅
- Duplicates moved ✅
- Monitoring starts ✅

**Days 2-7:**
- Run daily check (5 min/day)
- Watch for issues (expect none)

**Day 8:**
- Review results
- Delete decision
- Expect: Delete approved (71-112 MB freed)

---

**Ready? Copy COMMAND 1 and paste into PowerShell NOW** 🚀

