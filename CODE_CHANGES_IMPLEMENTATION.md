# Code Implementation Guide: GitHub Data Loading

## Files to Create/Modify

### FILE 1: Create `app/config.py` (NEW)

```python
"""
Configuration for data sources and URLs.
Centralizes all data repository references.
"""

import os

# GitHub data repository information
GITHUB_USERNAME = os.getenv('GITHUB_USERNAME', 'YOUR_USERNAME')
DATA_REPO_NAME = os.getenv('DATA_REPO_NAME', 'heart-data')

# Construct GitHub raw content URL
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{DATA_REPO_NAME}/main"

# Data source URLs
DATA_URLS = {
    'demo_data': f"{GITHUB_RAW_BASE}/train_balanced.csv",
    'test_data': f"{GITHUB_RAW_BASE}/test.csv",
    'model_results': f"{GITHUB_RAW_BASE}/model_results.csv",
}

# Optional: For future expansion
OPTIONAL_DATA_URLS = {
    'combined_heart': f"{GITHUB_RAW_BASE}/combined_heart.csv",
    'original_heart': f"{GITHUB_RAW_BASE}/heart.csv",
    'original_train': f"{GITHUB_RAW_BASE}/train_original.csv",
}

# Timeout for GitHub requests (seconds)
GITHUB_TIMEOUT = 10

# Enable/disable fallback to hardcoded data
USE_FALLBACK_DATA = True
```

---

### FILE 2: Modify `app/main.py` - Update load_sample_data()

**LOCATION**: Find the `load_sample_data()` function (around line 100-150)

**BEFORE**:
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
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None
```

**AFTER**:
```python
@st.cache_data
def load_sample_data():
    """Load sample data from GitHub repository with fallback."""
    from app.config import DATA_URLS, USE_FALLBACK_DATA, GITHUB_TIMEOUT
    
    # Try to load from GitHub first
    try:
        st.info("📊 Loading data from GitHub...")
        df = pd.read_csv(DATA_URLS['demo_data'], timeout=GITHUB_TIMEOUT)
        st.success("✅ Data loaded from GitHub")
        return df.head(1000)
    except Exception as e:
        st.warning(f"⚠️ Could not load from GitHub: {e}")
    
    # Fallback: Try to load from local file (for development)
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_balanced.csv')
        if os.path.exists(data_path):
            st.info("📁 Loading data from local file...")
            df = pd.read_csv(data_path)
            st.success("✅ Data loaded locally")
            return df.head(1000)
    except Exception as e:
        st.warning(f"⚠️ Could not load from local file: {e}")
    
    # Final fallback: Use hardcoded sample data
    if USE_FALLBACK_DATA:
        st.warning("📝 Using hardcoded sample data")
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
    else:
        st.error("❌ Could not load data and fallback disabled")
        return None
```

**Or Simplified Version** (if you prefer):
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

### FILE 3: Modify `.gitignore` - Add Data Exclusion

**LOCATION**: Find the comment section about data files (around line 40-45)

**BEFORE**:
```gitignore
# Data and models (keep commented if you want to track them)
# data/*.csv
# models/*.pkl
```

**AFTER**:
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

### FILE 4: Modify `README.md` - Add Data Source Section

**LOCATION**: Add new section after "Features" or "Installation"

**ADD THIS SECTION**:
```markdown
## Data Source

This application uses datasets stored in a separate GitHub repository to keep the main repository lean and deployment-ready.

### External Data Repository
- **Repository**: [heart-data](https://github.com/YOUR_USERNAME/heart-data)
- **Accessed via**: GitHub raw content URLs
- **Datasets included**:
  - `train_balanced.csv` - Balanced training dataset (used by app)
  - `test.csv` - Test dataset
  - `model_results.csv` - Model performance metrics
  - `combined_heart.csv` - Combined source data
  - `heart.csv` - Original UCI dataset
  - And more...

### How It Works

**Production Environment** (Streamlit Cloud, Railway, etc.):
1. App starts on cloud platform
2. Fetches `train_balanced.csv` from GitHub raw content URL
3. Caches data in memory with `@st.cache_data`
4. Uses cached data for all subsequent requests
5. Falls back to hardcoded sample data if GitHub is unavailable

**Local Development**:
1. App tries to load from GitHub first
2. If offline, tries to use local `data/train_balanced.csv` if it exists
3. Falls back to hardcoded sample data

### Setting Up Local Development

If you want to use local data files during development:

```bash
# Clone the data repository
git clone https://github.com/YOUR_USERNAME/heart-data.git
cp heart-data/train_balanced.csv data/
cp heart-data/test.csv data/
```

### Customizing Data Source

To use a different data repository, set environment variables:

```bash
export GITHUB_USERNAME="your_username"
export DATA_REPO_NAME="your_data_repo"
```

Or in `.streamlit/secrets.toml`:
```toml
GITHUB_USERNAME = "your_username"
DATA_REPO_NAME = "your_data_repo"
```
```

---

## Optional: Advanced Usage

### Add Function to Load Multiple Datasets

**IN**: `app/config.py` or `app/main.py`

```python
def get_data_from_github(dataset_name='demo_data'):
    """
    Load any dataset from GitHub data repository.
    
    Parameters:
    -----------
    dataset_name : str
        Key from DATA_URLS dict (demo_data, test_data, etc.)
    
    Returns:
    --------
    pd.DataFrame or None
    """
    from app.config import DATA_URLS, GITHUB_TIMEOUT
    import pandas as pd
    import streamlit as st
    
    if dataset_name not in DATA_URLS:
        st.error(f"Dataset '{dataset_name}' not found in DATA_URLS")
        return None
    
    try:
        url = DATA_URLS[dataset_name]
        df = pd.read_csv(url, timeout=GITHUB_TIMEOUT)
        return df
    except Exception as e:
        st.error(f"Error loading {dataset_name}: {e}")
        return None
```

---

## Requirements Changes

**NO CHANGES NEEDED** - Current `requirements.txt` already has `pandas`:

```txt
pandas>=1.5.0      # Already has URL reading capability
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
scipy>=1.10.0,<1.15.0
xgboost>=1.6.0
catboost>=1.0.0
imbalanced-learn>=0.9.0
shap>=0.41.0
streamlit>=1.25.0
plotly>=5.10.0
pytest>=7.0.0
fairlearn>=0.9.0
```

✅ `pandas.read_csv()` works with URLs out of the box!

---

## Testing Updates

**IN**: `tests/test_streamlit_app.py`

```python
# Add this import
from unittest.mock import patch, MagicMock

# Add/Update test
def test_load_sample_data_from_github():
    """Test that data loads from GitHub."""
    from app.main import load_sample_data
    
    # Mock successful GitHub response
    with patch('pandas.read_csv') as mock_read:
        mock_read.return_value = pd.DataFrame({
            'Age': [54, 37, 41],
            'Sex': [1, 1, 0],
            'HeartDisease': [1, 0, 1]
        })
        
        data = load_sample_data()
        assert data is not None
        assert len(data) >= 3
        mock_read.assert_called_once()

def test_load_sample_data_fallback():
    """Test that fallback data works when GitHub fails."""
    from app.main import load_sample_data
    
    # Mock GitHub failure
    with patch('pandas.read_csv') as mock_read:
        mock_read.side_effect = Exception("Network error")
        
        data = load_sample_data()
        assert data is not None
        assert len(data) > 0  # Should use fallback
```

---

## Step-by-Step Implementation

### Phase 1: Prepare Data Repository (5 minutes)
```bash
# 1. Create on GitHub: heart-data
# 2. Clone locally:
git clone https://github.com/YOUR_USERNAME/heart-data.git
cd heart-data

# 3. Copy CSV files
cp /path/to/Heart-Failure-Prediction/data/*.csv .

# 4. Create README
echo "# Heart Disease Datasets" > README.md

# 5. Push to GitHub
git add .
git commit -m "Add heart disease datasets"
git push
```

### Phase 2: Update Code (10 minutes)
```bash
cd /path/to/Heart-Failure-Prediction

# 1. Create config file
# (copy app/config.py code from above)

# 2. Update app/main.py
# (update load_sample_data function)

# 3. Update .gitignore
# (add data/ exclusions)

# 4. Update README.md
# (add Data Source section)

# 5. Test locally
streamlit run app/main.py
```

### Phase 3: Verify & Commit (10 minutes)
```bash
# 1. Test app loads data from GitHub
# (should see "Loading data from GitHub...")

# 2. Test with network down (optional)
# (should see "Using sample data")

# 3. Commit changes
git add -A
git commit -m "Implement GitHub-based data loading"
git push

# 4. Delete local data folder
rm -r data/
```

### Phase 4: Deploy (5 minutes)
- Push to GitHub
- Deploy to Streamlit Community Cloud
- Verify app works

**Total Time: ~30 minutes**

---

## Validation Checklist

- [ ] `app/config.py` created with GitHub URLs
- [ ] `app/main.py` updated to load from GitHub
- [ ] `load_sample_data()` has fallback logic
- [ ] Fallback data matches original format
- [ ] `.gitignore` updated to exclude `data/`
- [ ] `README.md` has Data Source section
- [ ] GitHub data repo created and has CSV files
- [ ] App runs locally: `streamlit run app/main.py`
- [ ] Data loads from GitHub (check console messages)
- [ ] App works with sample data (if GitHub down)
- [ ] Tests pass: `pytest -v`
- [ ] Repo size is now <400 MB
- [ ] Deployed to cloud and working

