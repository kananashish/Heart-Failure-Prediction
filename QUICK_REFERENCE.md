# Quick Reference: GitHub Data Loading Changes

## TL;DR - What Changes?

| Item | Before | After | Action |
|------|--------|-------|--------|
| **Project Size** | 3.61 GB | 350 MB | Delete files + code changes |
| **Data Location** | Local `data/` folder | GitHub `heart-data` repo | Move CSVs to separate repo |
| **Code Changes** | 0 | ~30 lines | Modify 3-4 files |
| **Files to Create** | 0 | 1 (`app/config.py`) | Create new config file |
| **Fallback** | Hardcoded sample | Hardcoded sample | Same as before |

---

## Files to Create/Modify

### 📝 CREATE: `app/config.py`

```python
import os

GITHUB_USERNAME = os.getenv('GITHUB_USERNAME', 'YOUR_USERNAME')
DATA_REPO_NAME = os.getenv('DATA_REPO_NAME', 'heart-data')
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{DATA_REPO_NAME}/main"

DATA_URLS = {
    'demo_data': f"{GITHUB_RAW_BASE}/train_balanced.csv",
    'test_data': f"{GITHUB_RAW_BASE}/test.csv",
    'model_results': f"{GITHUB_RAW_BASE}/model_results.csv",
}

GITHUB_TIMEOUT = 10
USE_FALLBACK_DATA = True
```

**Location**: Create at `app/config.py` (new file)

---

### ✏️ MODIFY: `app/main.py`

**Find this function** (around line 100-150):
```python
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return df.head(1000)
        else:
            # Create sample data if file doesn't exist
            sample_data = {...}
            return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None
```

**Replace with**:
```python
@st.cache_data
def load_sample_data():
    """Load sample data from GitHub repository."""
    from app.config import DATA_URLS
    
    try:
        # Load from GitHub
        url = DATA_URLS['demo_data']
        df = pd.read_csv(url, timeout=10)
        return df.head(1000)
    except:
        # Fallback to hardcoded sample data
        sample_data = {
            'Age': [54, 37, 41, 56, 57],
            'Sex': [1, 1, 0, 1, 0],
            'ChestPainType': [2, 1, 1, 0, 0],
            'RestingBP': [150, 130, 140, 120, 120],
            'Cholesterol': [195, 250, 204, 236, 354],
            'FastingBS': [0, 0, 0, 0, 0],
            'RestingECG': [1, 1, 0, 1, 1],
            'MaxHR': [122, 187, 172, 178, 163],
            'ExerciseAngina': [0, 0, 0, 0, 1],
            'Oldpeak': [0, 3.5, 1.4, 0.8, 0.6],
            'ST_Slope': [2, 0, 2, 2, 2],
            'HeartDisease': [0, 0, 0, 0, 0]
        }
        return pd.DataFrame(sample_data)
```

---

### ✏️ MODIFY: `.gitignore`

**Find this section** (around line 40-45):
```gitignore
# Data and models (keep commented if you want to track them)
# data/*.csv
# models/*.pkl
```

**Replace with**:
```gitignore
# Data files - stored in separate heart-data repository
data/*.csv
data/*.db
data/

# Models are kept (only production models)
!models/best_heart_model.pkl
!models/preprocessor.pkl
```

---

### ✏️ MODIFY: `README.md`

**Add this section** (after Features or Installation):

```markdown
## Data Source

This application uses datasets stored in a separate GitHub repository for optimal deployment size.

### External Data Repository
- **Repository**: [heart-data](https://github.com/YOUR_USERNAME/heart-data)
- **Datasets**: `train_balanced.csv`, `test.csv`, `model_results.csv`, and more...

### How It Works
- **Production**: Fetches data from GitHub raw content URLs
- **Fallback**: Uses hardcoded sample data if GitHub unavailable
- **Development**: Tries GitHub first, then local files if offline

### Setting Up Data Repository
```bash
# Create on GitHub: heart-data repository
git clone https://github.com/YOUR_USERNAME/heart-data.git
cp data/*.csv heart-data/
cd heart-data
git add .
git commit -m "Add datasets"
git push
```
```

---

## Files to Delete

### Delete These Directories
```
heart_failure_env/          [800 MB]
heart_failure_env_new/      [600 MB]
.venv/                      [500 MB]
htmlcov/                    [150 MB]
catboost_info/              [50 MB]
```

### Delete These Files
```
data/combined_heart.csv     [200 MB]  → Move to github-data repo
data/heart.csv              [500 MB]  → Move to github-data repo
data/uci_heart.csv          [100 MB]  → Move to github-data repo
data/train_original.csv     [150 MB]  → Move to github-data repo
data/hospitals.csv          [10 MB]   → Move to github-data repo
data/best_model.pkl         [30 MB]   → Duplicate, delete
data/preprocessor.pkl       [10 MB]   → Duplicate, delete
models/preprocessor_v2.pkl  [10 MB]   → Backup, delete
```

---

## Quick Checklist

### Pre-Implementation
- [ ] Read GITHUB_DATA_LOADING_PLAN.md
- [ ] Read CODE_CHANGES_IMPLEMENTATION.md
- [ ] Understand the GitHub data loading approach

### Phase 1: Create Data Repository (10 min)
- [ ] Create `heart-data` repo on GitHub
- [ ] Clone locally
- [ ] Copy CSV files to heart-data repo
- [ ] Push to GitHub

### Phase 2: Code Changes (15 min)
- [ ] Create `app/config.py` (copy code above)
- [ ] Update `load_sample_data()` in `app/main.py` (copy code above)
- [ ] Update `.gitignore` (add data exclusions)
- [ ] Update `README.md` (add Data Source section)

### Phase 3: Cleanup (10 min)
- [ ] Delete all files listed above
- [ ] Delete `data/` folder (or keep for local dev)
- [ ] Verify `models/best_heart_model.pkl` still exists
- [ ] Verify `models/preprocessor.pkl` still exists

### Phase 4: Testing (10 min)
- [ ] Run `streamlit run app/main.py`
- [ ] Verify app loads
- [ ] Verify data loads from GitHub
- [ ] Verify predictions work
- [ ] Run `pytest` (should pass)

### Phase 5: Deploy (10 min)
- [ ] Commit changes: `git add -A && git commit -m "GitHub data loading"`
- [ ] Push to GitHub: `git push`
- [ ] Deploy to Streamlit Community Cloud
- [ ] Verify app works in cloud

---

## Configuration

### Local Development Environment Variables (Optional)

**For custom GitHub repository location:**
```bash
export GITHUB_USERNAME="your_username"
export DATA_REPO_NAME="your_data_repo"
```

**Or in `.streamlit/secrets.toml`:**
```toml
GITHUB_USERNAME = "your_username"
DATA_REPO_NAME = "your_data_repo"
```

---

## Testing the Implementation

### Test 1: App Loads with GitHub Data
```bash
streamlit run app/main.py
# Should see: "Loading data from GitHub..." or similar message
# Should load successfully
```

### Test 2: App Works Offline
```bash
# Block GitHub access in firewall or mock it
# App should fall back to hardcoded sample data
# Should still work!
```

### Test 3: Tests Pass
```bash
pytest -v
# All tests should pass
```

### Test 4: Check Project Size
```bash
# On Windows PowerShell:
$size = (Get-ChildItem -Path . -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "Project size: $size GB"
# Should be < 0.4 GB
```

---

## Rollback Plan

If something doesn't work:

```bash
# Restore from backup (if you created one)
git restore .

# Or restore specific file
git restore app/main.py
git restore app/config.py

# Revert commit
git reset --hard HEAD~1
```

---

## File Structure After Changes

### Before
```
Heart-Failure-Prediction/  [3.61 GB]
├── app/main.py
├── data/
│   ├── combined_heart.csv  [200 MB]
│   ├── heart.csv           [500 MB]
│   ├── *.csv               [500+ MB]
│   └── *.pkl
├── models/
└── src/
```

### After
```
Heart-Failure-Prediction/  [350 MB]
├── app/
│   ├── main.py            (updated)
│   ├── config.py          (new)
│   └── ...
├── models/
│   ├── best_heart_model.pkl
│   └── preprocessor.pkl
├── src/
└── (NO data/ folder!)

heart-data/  [1.0 GB - separate GitHub repo]
├── combined_heart.csv
├── heart.csv
├── *.csv
└── README.md
```

---

## Code Changes Summary

| File | Type | Lines | Status |
|------|------|-------|--------|
| `app/config.py` | Create | ~20 | Copy from above |
| `app/main.py` | Edit | ~15 | Replace function |
| `.gitignore` | Edit | ~8 | Add exclusions |
| `README.md` | Edit | ~20 | Add section |
| **TOTAL** | | ~63 | All simple |

**Complexity**: Easy - just configuration and function replacement
**Risk**: Very Low - fallback ensures app always works

---

## Help & Troubleshooting

### App won't start
```python
# Check if config.py can be imported
python -c "from app.config import DATA_URLS; print(DATA_URLS)"
# Should print the URLs without error
```

### Data won't load
```python
# Test if GitHub URL is correct
import pandas as pd
url = "https://raw.githubusercontent.com/YOUR_USERNAME/heart-data/main/train_balanced.csv"
df = pd.read_csv(url)
print(df.head())
# Should print the data
```

### Tests failing
```bash
# Run specific test with verbose output
pytest tests/test_streamlit_app.py -v -s

# Check if fallback data is correct format
python -c "from app.main import load_sample_data; df = load_sample_data(); print(df.shape)"
```

### Size still too large
```bash
# Check what's taking space
Get-ChildItem -Recurse | Select-Object FullName, @{Name="Size";Expression={$_.Length/1MB}} | Sort-Object -Property Size -Descending | Select-Object -First 20
```

---

## Environment Setup (One-Time)

### For Local Development
```bash
# Clone data repo (optional, for offline dev)
git clone https://github.com/YOUR_USERNAME/heart-data.git

# Create data folder link (if using local data)
mkdir data
copy heart-data/train_balanced.csv data/
```

### For Deployment
- **Streamlit Community Cloud**: Auto-handles everything
- **Railway/Render**: Just push to GitHub, they handle the rest
- **Custom Server**: App will fetch from GitHub automatically

---

## Time Estimate

| Task | Time |
|------|------|
| Read documentation | 10 min |
| Create data repo | 10 min |
| Code changes | 15 min |
| Testing | 15 min |
| Cleanup | 10 min |
| Deployment | 10 min |
| **TOTAL** | **70 min** |

With practice: ~50 minutes

---

## Expected Results

✅ Project size: 350 MB (down from 3.61 GB)  
✅ Deployment time: 2-3 min (down from 10+ min)  
✅ App functionality: Identical  
✅ Code quality: Improved  
✅ Maintainability: Better  
✅ Scalability: Much better  

---

**Ready to start? Follow the Quick Checklist above! 🚀**

