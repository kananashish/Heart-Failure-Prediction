# Cleanup Script & Verification Guide

## Option A: Using PowerShell (Windows)

### Step 1: Backup First (Safety Check)
```powershell
# Create a backup before cleanup
Compress-Archive -Path . -DestinationPath "backup_before_cleanup.zip" -Force
```

### Step 2: Delete Virtual Environments
```powershell
# Remove virtual environment directories
Remove-Item -Path "heart_failure_env" -Recurse -Force
Remove-Item -Path "heart_failure_env_new" -Recurse -Force
Remove-Item -Path ".venv" -Recurse -Force

Write-Host "✅ Virtual environments removed"
```

### Step 3: Delete Development Artifacts
```powershell
# Remove coverage reports
Remove-Item -Path "htmlcov" -Recurse -Force

# Remove CatBoost training cache
Remove-Item -Path "catboost_info" -Recurse -Force

# Remove backup model
Remove-Item -Path "models/preprocessor_v2.pkl" -Force

Write-Host "✅ Development artifacts removed"
```

### Step 4: Delete Large Data Files
```powershell
# Remove raw CSV data files from data directory
Remove-Item -Path "data/combined_heart.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/heart.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/uci_heart.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/train_original.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/hospitals.csv" -Force -ErrorAction SilentlyContinue

# Remove duplicate model files
Remove-Item -Path "data/best_model.pkl" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/preprocessor.pkl" -Force -ErrorAction SilentlyContinue

Write-Host "✅ Large data files removed"
```

### Step 5: Check Remaining Files
```powershell
# List remaining files in data/
Write-Host "Remaining in data/ folder:"
Get-ChildItem -Path "data" | Select-Object Name, @{Name="Size";Expression={"{0:N2} MB" -f ($_.Length/1MB)}}

# Total size
Write-Host "`nProject size:"
"{0:N2} MB" -f ((Get-ChildItem -Path . -Recurse | Measure-Object -Property Length -Sum).Sum/1MB)
```

---

## Option B: Complete Cleanup Script (Automated)

Save as `cleanup_project.ps1`:

```powershell
# ============================================
# Heart Failure Prediction - Cleanup Script
# ============================================

param(
    [switch]$DryRun = $false,  # Show what would be deleted without deleting
    [switch]$Backup = $true     # Create backup before cleanup
)

Write-Host "🧹 Heart Failure Prediction Project Cleanup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Define items to delete
$itemsToDelete = @(
    # Virtual environments
    "heart_failure_env",
    "heart_failure_env_new",
    ".venv",
    
    # Development artifacts
    "htmlcov",
    "catboost_info",
    "models/preprocessor_v2.pkl",
    
    # Large data files
    "data/combined_heart.csv",
    "data/heart.csv",
    "data/uci_heart.csv",
    "data/train_original.csv",
    "data/hospitals.csv",
    "data/best_model.pkl",
    "data/preprocessor.pkl"
)

# Calculate size before cleanup
$beforeSize = (Get-ChildItem -Path . -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB

Write-Host "`n📊 Current project size: $("{0:N2}" -f $beforeSize) GB"

# Create backup if requested
if ($Backup -and -not $DryRun) {
    Write-Host "`n💾 Creating backup..." -ForegroundColor Yellow
    $backupFile = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
    Compress-Archive -Path . -DestinationPath $backupFile -Force
    Write-Host "✅ Backup created: $backupFile"
}

# Process deletions
Write-Host "`n🗑️  Items to delete:" -ForegroundColor Yellow

$totalToDelete = 0

foreach ($item in $itemsToDelete) {
    if (Test-Path $item) {
        $itemSize = (Get-Item $item -Force | Measure-Object -Property Length -Sum -Recurse).Sum / 1MB
        $totalToDelete += $itemSize
        
        if ($DryRun) {
            Write-Host "  [DRY RUN] Would delete: $item ($("{0:N2}" -f $itemSize) MB)" -ForegroundColor Magenta
        } else {
            Write-Host "  ❌ Deleting: $item ($("{0:N2}" -f $itemSize) MB)..." -NoNewline
            Remove-Item -Path $item -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host " Done ✅"
        }
    }
}

if (-not $DryRun) {
    # Calculate size after cleanup
    $afterSize = (Get-ChildItem -Path . -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
    $saved = $beforeSize - $afterSize
    $percentSaved = ($saved / $beforeSize) * 100
    
    Write-Host "`n📊 Size comparison:" -ForegroundColor Green
    Write-Host "  Before: $("{0:N2}" -f $beforeSize) GB"
    Write-Host "  After:  $("{0:N2}" -f $afterSize) GB"
    Write-Host "  Saved:  $("{0:N2}" -f $saved) GB ($("{0:N2}" -f $percentSaved)%)" -ForegroundColor Green
    
    Write-Host "`n📁 Remaining files in data/:" -ForegroundColor Green
    Get-ChildItem -Path "data" | Select-Object Name, @{Name="Size";Expression={"{0:N2} MB" -f ($_.Length/1MB)}}
}

Write-Host "`n✅ Cleanup complete!" -ForegroundColor Green
```

### Run the Cleanup Script

**Dry run (see what would be deleted):**
```powershell
.\cleanup_project.ps1 -DryRun
```

**Actual cleanup with backup:**
```powershell
.\cleanup_project.ps1 -Backup
```

**Cleanup without backup (faster):**
```powershell
.\cleanup_project.ps1 -Backup:$false
```

---

## Step 6: Verify Everything Works

### Test 1: App Runs
```powershell
# Install dependencies (if needed)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/main.py
```

**Expected**: App loads with sample heart failure data, model predictions work

### Test 2: Tests Pass
```powershell
# Run pytest
pytest -v

# Or specific test
pytest tests/test_streamlit_app.py -v
```

**Expected**: Tests pass (or show expected failures, not import errors)

### Test 3: Model Loads Correctly
```powershell
python -c "
import joblib
import os

model_path = 'models/best_heart_model.pkl'
model = joblib.load(model_path)
print(f'✅ Model loaded successfully: {model}')
"
```

**Expected**: Model loads without errors

### Test 4: Size Verification
```powershell
# Check project size
$size = (Get-ChildItem -Path . -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "📊 Final project size: $("{0:N2}" -f $size) GB"

# List top 10 largest directories
Write-Host "`n📁 Top directories by size:"
Get-ChildItem -Path . -Recurse -Directory | 
    ForEach-Object {
        $size = (Get-ChildItem -Path $_ -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        [PSCustomObject]@{
            Path = $_.FullName
            SizeMB = [Math]::Round($size, 2)
        }
    } | 
    Sort-Object -Property SizeMB -Descending | 
    Select-Object -First 10 | 
    Format-Table
```

---

## Option C: Using Git (If You're Using Version Control)

If your project is in Git, you can also clean history:

```powershell
# See large files in git history
git rev-list --all --objects | 
    ForEach-Object { 
        $obj = $_ -split ' ' 
        $size = (git cat-file -s $obj[0]) 
        [PSCustomObject]@{
            Size = $size
            File = $obj[1]
        }
    } | 
    Sort-Object -Property Size -Descending | 
    Select-Object -First 20

# After deleting files locally, remove from git history:
git add -A
git commit -m "Remove large files and artifacts for optimization"
```

---

## Cleanup Verification Checklist

After running cleanup, verify:

- [ ] `heart_failure_env/` deleted
- [ ] `heart_failure_env_new/` deleted
- [ ] `.venv/` deleted
- [ ] `htmlcov/` deleted
- [ ] `catboost_info/` deleted
- [ ] `data/combined_heart.csv` deleted
- [ ] `data/heart.csv` deleted
- [ ] `data/uci_heart.csv` deleted
- [ ] `data/train_original.csv` deleted
- [ ] `data/hospitals.csv` deleted
- [ ] `data/best_model.pkl` deleted
- [ ] `data/preprocessor.pkl` deleted
- [ ] `models/preprocessor_v2.pkl` deleted
- [ ] `models/best_heart_model.pkl` KEPT
- [ ] `models/preprocessor.pkl` KEPT
- [ ] `data/train_balanced.csv` KEPT (optional)
- [ ] `data/test.csv` KEPT
- [ ] `data/heart_hospitals.db` KEPT
- [ ] `data/users.db` KEPT
- [ ] App runs: `streamlit run app/main.py` ✅
- [ ] Tests pass: `pytest` ✅
- [ ] Model loads correctly ✅
- [ ] Project size < 500 MB ✅

---

## Expected Output After Cleanup

```
📊 Current project size: 3.61 GB

💾 Creating backup...
✅ Backup created: backup_20240426_143022.zip

🗑️  Items to delete:
  ❌ Deleting: heart_failure_env (817.53 MB)... Done ✅
  ❌ Deleting: heart_failure_env_new (612.35 MB)... Done ✅
  ❌ Deleting: .venv (445.23 MB)... Done ✅
  ❌ Deleting: htmlcov (154.67 MB)... Done ✅
  ❌ Deleting: catboost_info (45.89 MB)... Done ✅
  ❌ Deleting: models/preprocessor_v2.pkl (12.34 MB)... Done ✅
  ❌ Deleting: data/combined_heart.csv (198.76 MB)... Done ✅
  ❌ Deleting: data/heart.csv (523.45 MB)... Done ✅
  ❌ Deleting: data/uci_heart.csv (102.34 MB)... Done ✅
  ❌ Deleting: data/train_original.csv (156.78 MB)... Done ✅
  ❌ Deleting: data/hospitals.csv (8.90 MB)... Done ✅
  ❌ Deleting: data/best_model.pkl (34.56 MB)... Done ✅
  ❌ Deleting: data/preprocessor.pkl (9.87 MB)... Done ✅

📊 Size comparison:
  Before: 3.61 GB
  After:  0.42 GB
  Saved:  3.19 GB (88.37%) ✅

📁 Remaining files in data/:
Name                  Size (MB)
----                  --------
train_balanced.csv    5.23
test.csv              1.45
heart_hospitals.db    1.12
users.db              0.89
model_results.csv     0.15

✅ Cleanup complete!
```

---

## Troubleshooting

### Error: "File is in use"
**Solution**: Close all Python processes and Streamlit servers
```powershell
# Kill Python processes
Get-Process python | Stop-Process -Force
```

### Error: "Access Denied"
**Solution**: Run PowerShell as Administrator

### Want to restore?
```powershell
# Extract the backup
Expand-Archive -Path "backup_*.zip" -DestinationPath "." -Force
```

