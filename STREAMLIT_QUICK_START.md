# Quick Streamlit Deployment Steps

## 5-Minute Quick Start

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for Streamlit deployment"
git push origin main
```

### Step 2: Create Streamlit Cloud Account
- Go to https://share.streamlit.io
- Sign up with GitHub
- Authorize Streamlit

### Step 3: Deploy
1. Click "Create app"
2. Select your GitHub repo
3. Set main file to `streamlit_app.py`
4. Click "Deploy"

### Step 4: Add Secrets
1. Go to app settings (⚙️)
2. Click "Secrets"
3. Add:
```toml
GITHUB_USERNAME = "kananashish"
DATA_REPO_NAME = "heart-data"
```

### Step 5: Done! 🎉
Your app will be live at: `https://your-username-heart-failure-prediction.streamlit.app`

---

## What's Been Configured for You

✅ **Entry Point** - `streamlit_app.py` created at root level
✅ **Theme** - Custom red/white medical theme in `.streamlit/config.toml`
✅ **Dependencies** - `requirements.txt` updated with all needed packages
✅ **Documentation** - Full deployment guide in `STREAMLIT_DEPLOYMENT.md`
✅ **Secrets Template** - `.streamlit/secrets.toml.example` for reference

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Import errors | All imports are relative; project structure preserved |
| Data not loading | Check GitHub repo is public; verify secrets configured |
| Slow performance | Data is cached with `@st.cache_data` |
| Database issues | SQLite is file-based; consider PostgreSQL for production |

---

## Useful Links

- 📚 [Streamlit Documentation](https://docs.streamlit.io)
- ☁️ [Streamlit Cloud](https://streamlit.io/cloud)
- 🐙 [GitHub](https://github.com)
- 🏥 [Your Project Docs](./docs)

---

## Need Help?

See detailed guide: `STREAMLIT_DEPLOYMENT.md`
