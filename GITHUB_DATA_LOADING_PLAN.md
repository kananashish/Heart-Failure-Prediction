# Modified Optimization Plan: GitHub-Based Data Loading

## New Architecture

### Original Plan
```
Local Repo (3.61 GB)
├── app/
├── src/
├── data/  [1.5 GB - CSV files]
└── models/
```

### Modified Plan (BETTER! ✨)
```
Production Repo (300-350 MB) - Deployed
├── app/
├── src/
├── models/
└── requirements.txt

Data Repo (GitHub - ~1 GB) - NOT deployed
├── combined_heart.csv
├── heart.csv
├── uci_heart.csv
├── train_original.csv
├── train_balanced.csv
├── test.csv
├── hospitals.csv
└── README.md
```

---

## What Changes with GitHub-Based Data Loading

### 1. CODE CHANGES REQUIRED (Minimal ✅)

#### Change 1: app/main.py - load_sample_data() function

**BEFORE** (Load from local filesystem):
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
            # Fallback to hardcoded sample
            sample_data = {...}
            return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None
```

**AFTER** (Load from GitHub):
```python
@st.cache_data
def load_sample_data():
    """Load sample data from GitHub repository."""
    try:
        # GitHub raw content URL
        github_url = "https://raw.githubusercontent.com/YOUR_USERNAME/heart-data/main/train_balanced.csv"
        df = pd.read_csv(github_url)
        return df.head(1000)
    except Exception as e:
        # Fallback to hardcoded sample if GitHub is down
        st.warning("Could not load data from GitHub, using sample data")
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

#### Change 2: Add GitHub Config (Optional - Best Practice)

Create `app/config.py`:
```python
# Data source configuration
DATA_URLS = {
    'demo_data': 'https://raw.githubusercontent.com/YOUR_USERNAME/heart-data/main/train_balanced.csv',
    'test_data': 'https://raw.githubusercontent.com/YOUR_USERNAME/heart-data/main/test.csv',
    'model_results': 'https://raw.githubusercontent.com/YOUR_USERNAME/heart-data/main/model_results.csv',
}

# Optional: Use environment variable for flexibility
import os
GITHUB_DATA_REPO = os.getenv('GITHUB_DATA_REPO', 'YOUR_USERNAME/heart-data')
```

Then in `app/main.py`:
```python
from app.config import DATA_URLS

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv(DATA_URLS['demo_data'])
        return df.head(1000)
    except:
        # fallback...
```

#### Change 3: Local Development (Optional)

```python
@st.cache_data
def load_sample_data():
    """Load data from GitHub, fallback to local if dev mode."""
    # Option A: Try GitHub first
    try:
        df = pd.read_csv(DATA_URLS['demo_data'])
        st.info("📊 Data loaded from GitHub")
        return df.head(1000)
    except:
        pass
    
    # Option B: Fall back to local file if exists
    local_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_balanced.csv')
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        st.info("📊 Data loaded locally (development mode)")
        return df.head(1000)
    
    # Option C: Use hardcoded sample
    st.warning("Using sample data")
    return pd.DataFrame({...})
```

**Benefit**: Developers can work offline with local data, production uses GitHub

---

### 2. REPOSITORY STRUCTURE CHANGES

#### Production Repo (Current Location)
```
Heart-Failure-Prediction/  [350 MB total]
├── app/
│   ├── main.py  (UPDATED - uses GitHub URLs)
│   ├── config.py (NEW - stores data URLs)
│   └── __init__.py
├── src/
│   ├── auth.py
│   ├── preprocess.py
│   ├── recommendations.py
│   ├── train.py
│   └── ...
├── models/
│   ├── best_heart_model.pkl  [50 MB]
│   └── preprocessor.pkl  [10 MB]
├── tests/
├── k8s/
├── docs/
├── scripts/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore  (UPDATED)
└── README.md  (UPDATED with GitHub data repo link)

❌ REMOVED:
├── data/  (NO LONGER NEEDED IN PRODUCTION!)
└── ALL CSV FILES
```

#### New Separate Data Repo (Create on GitHub)
```
heart-data/  [1.0 GB - NOT deployed]
├── README.md
│   └── "Heart Disease Data Repository
│       └── Download links and descriptions"
├── combined_heart.csv  [200 MB]
├── heart.csv  [500 MB]
├── uci_heart.csv  [100 MB]
├── train_original.csv  [150 MB]
├── train_balanced.csv  [5 MB] ⭐ Used by app
├── test.csv  [1 MB]
├── hospitals.csv  [10 MB]
├── model_results.csv  [100 KB]
└── .gitignore
    ```
    # Large files - use Git LFS if you want to track history
    *.csv
    *.db
    ```
```

**Note**: You could use **Git LFS (Large File Storage)** if you want to track CSV history, but GitHub free tier has limits.

---

### 3. FILE DELETION IMPACT

#### Still Delete (Same as Before)
```
✅ heart_failure_env/         [800 MB]  → Delete
✅ heart_failure_env_new/     [600 MB]  → Delete
✅ .venv/                     [500 MB]  → Delete
✅ htmlcov/                   [150 MB]  → Delete
✅ catboost_info/             [50 MB]   → Delete
✅ models/preprocessor_v2.pkl [10 MB]   → Delete
✅ data/best_model.pkl        [30 MB]   → Delete (duplicate)
✅ data/preprocessor.pkl      [10 MB]   → Delete (duplicate)
```

#### Now ALSO Delete (Different from Original Plan)
```
✅ data/combined_heart.csv    [200 MB]  → Delete (move to github-data repo)
✅ data/heart.csv             [500 MB]  → Delete (move to github-data repo)
✅ data/uci_heart.csv         [100 MB]  → Delete (move to github-data repo)
✅ data/train_original.csv    [150 MB]  → Delete (move to github-data repo)
✅ data/hospitals.csv         [10 MB]   → Delete (move to github-data repo)

❓ data/train_balanced.csv    [5 MB]    → Optional: Keep or move to github-data
❓ data/test.csv              [1 MB]    → Optional: Keep or move to github-data
```

**Benefit**: Even more aggressive cleanup! Remove ALL data from production repo.

---

### 4. SIZE COMPARISON

#### Original Optimization Plan
```
Before:  3.61 GB (includes venv + data + artifacts)
After:   400 MB  (includes models + code)

Split:
├── Production deployed:  350 MB
└── Data (GitHub):        1.0 GB (not deployed)
```

#### With GitHub-Based Data Loading (BETTER!)
```
Before:  3.61 GB (all in one repo)
After:   300-350 MB (PRODUCTION ONLY!)

Split:
├── Production deployed:  300-350 MB ⭐ Even smaller!
└── Data repo (GitHub):   1.0 GB (can be private or archived)
```

**Improvement**: 50-100 MB additional savings!

---

### 5. NEW SETUP PROCESS

#### Step 1: Create Data Repository
```powershell
# Create new repo on GitHub: YOUR_USERNAME/heart-data
# Clone it locally
git clone https://github.com/YOUR_USERNAME/heart-data.git

# Add CSV files
copy data/combined_heart.csv .\heart-data\
copy data/heart.csv .\heart-data\
copy data/train_original.csv .\heart-data\
copy data/test.csv .\heart-data\
copy data/train_balanced.csv .\heart-data\
# etc...

# Commit and push
cd heart-data
git add .
git commit -m "Add heart disease datasets"
git push
```

#### Step 2: Update Production Repo
```powershell
# Update app/main.py with GitHub URLs
# Create app/config.py with data URLs
# Delete data/ folder from production repo
# Update .gitignore to exclude data/

git add -A
git commit -m "Switch to GitHub-based data loading"
git push
```

#### Step 3: Test Locally
```powershell
# Should load from GitHub
streamlit run app/main.py
```

---

### 6. CODE CHANGES SUMMARY

| File | Type | Details |
|------|------|---------|
| `app/main.py` | ✏️ Edit | Update `load_sample_data()` to use GitHub URLs |
| `app/config.py` | ✨ Create | Store data repository URLs |
| `requirements.txt` | ✅ No change | Same dependencies |
| `src/*.py` | ✅ No change | Model training code unchanged |
| `tests/*.py` | ⚠️ Minor | May need to update test data paths |
| `.gitignore` | ✏️ Edit | Add `data/` to exclude all CSVs |
| `README.md` | ✏️ Edit | Add "Data Source" section linking to data repo |

**Total Code Changes**: ~15-30 lines across 2-3 files

---

### 7. BENEFITS vs DRAWBACKS

#### ✅ Benefits
| Benefit | Impact |
|---------|--------|
| **Smallest production repo** | 300-350 MB (vs 400 MB original plan) |
| **Fully cloud-ready** | No local data needed |
| **Data versioning** | Separate repo for easy updates |
| **Scalability** | Add new datasets without repo bloat |
| **Offline fallback** | Hardcoded sample data works always |
| **GitHub as data store** | Free storage for datasets |
| **CI/CD friendly** | No large files in production pipeline |
| **License flexibility** | Data repo can have different license |
| **Easier sharing** | Share data repo separately from code |

#### ⚠️ Drawbacks
| Drawback | Mitigation |
|----------|-----------|
| **Requires internet** | Hardcoded fallback in code |
| **GitHub rate limits** | Very generous for direct downloads (1000/hour) |
| **Data loading slower** | Cached with `@st.cache_data` (loads once) |
| **Dependency on GitHub** | Use environment variables for URL flexibility |
| **Two repos to manage** | Worth the benefit! |

---

### 8. DEPLOYMENT FLOWCHART

#### Before (Original Plan)
```
Production Repo (400 MB)
├── Code + models + data
└── Deploy as-is
```

#### After (GitHub-Based)
```
Production Repo (300-350 MB)          Data Repo (1.0 GB)
├── Code + models + config ──────────→ GitHub raw content
├── Download train_balanced.csv
│   at app startup
└── Cache in memory
```

---

### 9. EXAMPLE CODE CHANGES

#### Minimal Version (3 lines changed)
```python
# app/main.py

@st.cache_data
def load_sample_data():
    try:
        # ⬇️ CHANGED: GitHub URL instead of local path
        url = "https://raw.githubusercontent.com/YOUR_USERNAME/heart-data/main/train_balanced.csv"
        df = pd.read_csv(url)
        return df.head(1000)
    except:
        # fallback to hardcoded sample data
        return pd.DataFrame({...})
```

#### Professional Version (Better maintainability)
```python
# app/config.py (NEW FILE)
import os

DATA_REPO = os.getenv('HEART_DATA_REPO', 'YOUR_USERNAME/heart-data')
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{DATA_REPO}/main"

DATA_URLS = {
    'demo': f"{GITHUB_RAW_URL}/train_balanced.csv",
    'test': f"{GITHUB_RAW_URL}/test.csv",
    'results': f"{GITHUB_RAW_URL}/model_results.csv",
}

# app/main.py
from app.config import DATA_URLS
import pandas as pd
import streamlit as st

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv(DATA_URLS['demo'])
        return df.head(1000)
    except Exception as e:
        st.warning(f"GitHub data unavailable: {e}. Using sample data.")
        return pd.DataFrame({
            'Age': [54, 37, 41, 56, 57],
            'Sex': [1, 1, 0, 1, 0],
            # ... rest of sample data
        })
```

---

### 10. .gitignore CHANGES

**Add these lines**:
```gitignore
# Exclude all data files from production repo
data/
!data/.gitkeep  # Keep folder structure if needed

# These go to separate heart-data repo:
*.csv
*.db

# Remove the CSV files from tracking if they exist
git rm --cached data/*.csv
```

---

### 11. README.md CHANGES

**Add section**:
```markdown
## Data Repository

This application uses data stored in a separate GitHub repository for optimal deployment size.

### Data Source
- **Repository**: [YOUR_USERNAME/heart-data](https://github.com/YOUR_USERNAME/heart-data)
- **Included datasets**:
  - `train_balanced.csv` - Preprocessed training data (used by app)
  - `test.csv` - Test dataset
  - `combined_heart.csv` - Combined dataset
  - And more...

### Local Development
The app will automatically download data from GitHub on first run.

For offline development, you can clone the data repo:
```bash
git clone https://github.com/YOUR_USERNAME/heart-data.git
mv heart-data/train_balanced.csv data/
```

### Data Loading
- **Production**: Downloads from GitHub (cached in memory)
- **Fallback**: Uses hardcoded sample data if GitHub unavailable
- **Development**: Can use local files if available
```

---

### 12. DEPLOYMENT CHANGES

#### Streamlit Community Cloud
```
Before:  Push code + 1.5 GB data = FAILS (1 GB limit)
After:   Push code only (300 MB) = ✅ WORKS!
         
         App fetches data from GitHub on startup
```

#### Environment Variables (Optional)
For flexibility, add to deployment:

**Streamlit Secrets** (`.streamlit/secrets.toml`):
```toml
HEART_DATA_REPO = "YOUR_USERNAME/heart-data"
```

Or for production:
```bash
export HEART_DATA_REPO="organization/heart-data-prod"
```

---

### 13. TESTING CHANGES

**tests/test_streamlit_app.py** - Minor updates needed:

```python
# OLD: Loads from local data/heart.csv
def test_load_sample_data():
    data = load_sample_data()
    assert data is not None
    assert len(data) > 0

# NEW: Verifies GitHub loading works
def test_load_sample_data():
    # Test GitHub URL works
    data = load_sample_data()
    assert data is not None
    assert len(data) > 0
    
    # Test fallback works if GitHub down
    with patch('pandas.read_csv') as mock_read:
        mock_read.side_effect = Exception("Network error")
        data = load_sample_data()
        assert data is not None  # Should use fallback
```

---

### 14. COMPLETE SETUP CHECKLIST

#### Create Data Repository
- [ ] Create `heart-data` repo on GitHub (can be private)
- [ ] Add CSV files to data repo
- [ ] Push to GitHub
- [ ] Note the repo URL: `github.com/YOUR_USERNAME/heart-data`

#### Update Production Repository
- [ ] Clone production repo locally
- [ ] Create `app/config.py` with GitHub URLs
- [ ] Update `app/main.py` to load from GitHub URLs
- [ ] Delete local `data/` folder
- [ ] Update `.gitignore` to exclude `data/`
- [ ] Update `README.md` with data source info
- [ ] Run `streamlit run app/main.py` to test
- [ ] Verify app loads data from GitHub
- [ ] Run `pytest` to verify tests work
- [ ] Commit changes: `git commit -m "Switch to GitHub-based data loading"`
- [ ] Push to GitHub

#### Verify
- [ ] App loads successfully
- [ ] Predictions work
- [ ] Tests pass
- [ ] Data loads from GitHub
- [ ] Fallback data works (optional: test by blocking GitHub)

#### Deploy
- [ ] Deploy to Streamlit Community Cloud
- [ ] Verify app works in cloud
- [ ] Test predictions

---

### 15. COMPARISON TABLE: Before → After

| Aspect | Original Plan | GitHub-Based Plan |
|--------|---------------|-------------------|
| **Production size** | 400 MB | 300-350 MB |
| **Data location** | Local files in repo | GitHub separate repo |
| **Data folder** | `/data/` with CSVs | Empty or removed |
| **Code changes** | None | ~15-30 lines |
| **New files** | 0 | 1 (`app/config.py`) |
| **CSV loading** | Local filesystem | GitHub raw URLs |
| **Network needed** | No | Yes (with fallback) |
| **Deployable to SCC** | ✅ Barely (1GB limit) | ✅ Comfortably (350MB) |
| **Maintenance** | Single repo | 2 repos (simple) |
| **Scalability** | Limited | Excellent |
| **Data versioning** | Hard (mixed with code) | Easy (separate repo) |

---

## Summary: GitHub-Based Data Loading

### Key Changes
1. ✅ **Code**: Update 2-3 files (~30 lines total)
2. ✅ **Size**: 400 MB → 300-350 MB (even better!)
3. ✅ **Repos**: 1 production + 1 data repo
4. ✅ **Testing**: Minor test updates for GitHub loading
5. ✅ **Deployment**: No changes (works automatically)

### Files to Change
- `app/main.py` - Add GitHub URL loading
- `app/config.py` - New file with data URLs
- `.gitignore` - Add `data/` folder
- `README.md` - Add "Data Source" section
- `requirements.txt` - No change

### Benefits
- ✨ Smallest production repo possible (300-350 MB)
- ✨ Scalable data management
- ✨ Easier to update datasets
- ✨ Clean separation of concerns
- ✨ Better for CI/CD pipelines

### Trade-offs
- ⚠️ Requires internet (but has fallback)
- ⚠️ Two repos to manage (simple!)
- ⚠️ Slight startup delay (mitigated by caching)

### Effort
- ⏱️ Implementation: 30-45 minutes
- ⏱️ Testing: 15 minutes
- ⏱️ Total: ~1 hour

**Recommendation**: 🟢 **This is the BEST approach!** Cleaner, more scalable, and production-ready.

