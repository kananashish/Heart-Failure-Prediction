# Streamlit Cloud Deployment Guide

## Heart Failure Prediction System

This guide will help you deploy the Heart Failure Prediction System on Streamlit Cloud.

---

## Prerequisites

Before deploying, ensure you have:

1. **GitHub Account** - Required to connect your repository to Streamlit Cloud
2. **Streamlit Account** - Create a free account at [streamlit.io](https://streamlit.io)
3. **Git Repository** - Your project pushed to GitHub (public or private)

---

## Step 1: Prepare Your Repository

### 1.1 Ensure All Files Are Committed

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 1.2 Verify File Structure

The project should have these key files at the root level:
- ✅ `streamlit_app.py` - Main entry point (already created)
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `app/main.py` - Streamlit application code
- ✅ `src/` - Supporting modules (preprocess, recommendations, auth, etc.)

---

## Step 2: Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign up"** or **"Sign in"** with GitHub
3. Authorize Streamlit to access your GitHub repositories

---

## Step 3: Deploy Your App

### 3.1 Create New App

1. Click **"Create app"** button on Streamlit Cloud dashboard
2. Select your GitHub repository (kananashish/Heart-Failure-Prediction or your fork)
3. Specify the branch (usually `main`)
4. Set the main file path to `streamlit_app.py`
5. Click **"Deploy"**

### 3.2 Configure Secrets (Important!)

After deployment starts:

1. Go to your app's settings (⚙️ icon on the app page)
2. Navigate to **"Secrets"** tab
3. Add your secrets in the textarea:

```toml
GITHUB_USERNAME = "kananashish"
DATA_REPO_NAME = "heart-data"
```

**Note:** Secrets are NOT committed to GitHub. They're only stored in Streamlit Cloud.

---

## Step 4: Advanced Configuration (Optional)

### 4.1 Custom Requirements

If you need to install system dependencies or use specific versions:

Create a `packages.txt` file for system packages:

```
libpq-dev
libssl-dev
```

### 4.2 Update config.toml

The `.streamlit/config.toml` is already configured with:
- Custom theme (red/white/blue colors)
- CSRF protection enabled
- Max upload size: 200 MB
- Error details shown

Modify if needed for production.

### 4.3 Performance Optimization

For faster app loading, consider:

```toml
# .streamlit/config.toml additions
[client]
showErrorDetails = true

[logger]
level = "info"

[cache]
persistedDirSize = 1000  # MB
```

---

## Step 5: Environment Variables

### In Streamlit Cloud Dashboard:

1. Go to **App Settings** → **Secrets**
2. Add environment-specific configurations

### Available Env Variables:

- `GITHUB_USERNAME` - GitHub username for data repository
- `DATA_REPO_NAME` - Name of GitHub data repository
- `STREAMLIT_SERVER_MAXUPLOADSIZE` - Max upload size in MB

---

## Step 6: Verify Deployment

After deployment:

1. ✅ Check if app loads without errors
2. ✅ Test authentication system
3. ✅ Test prediction functionality
4. ✅ Verify data loads from GitHub
5. ✅ Test all interactive features

---

## Troubleshooting

### Issue: "streamlit_app.py not found"

**Solution:** Ensure `streamlit_app.py` exists at the root level (it should be created automatically).

### Issue: Import errors or module not found

**Solution:** 
- Verify all dependencies are in `requirements.txt`
- Check that `src/` directory structure is preserved in Git
- Ensure no local-only imports are used

### Issue: Data not loading from GitHub

**Solution:**
- Verify GitHub credentials in Secrets
- Check that data repository is public or has proper access
- Check GitHub rate limits (60 requests/hour for unauthenticated)

### Issue: Authentication not working

**Solution:**
- Ensure SQLite database path is writable
- For production, consider using an external database (PostgreSQL, etc.)
- Check that `auth.py` module is correctly imported

### Issue: Model not loading

**Solution:**
- Verify trained model files are in the repository or downloadable
- Check `models/` directory is properly committed
- Ensure model format matches training environment

---

## Scaling Considerations

### For Higher Traffic:

1. **Use External Database** - SQLite is file-based, not ideal for concurrent users
   - Migrate to PostgreSQL or MySQL
   - Update `database.py` with connection string from Secrets

2. **Cache Data** - Already implemented with `@st.cache_data`
   - Further optimize with TTL parameters

3. **Upgrade Streamlit Tier** - Free tier has limitations
   - Consider Streamlit's professional tier for dedicated resources

---

## Production Recommendations

### 1. Database Migration

```python
# Update database.py to use external database
import os
db_url = os.getenv('DATABASE_URL', 'sqlite:///heart_failure.db')
```

### 2. Add Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### 3. Error Monitoring

Consider integrating with:
- Sentry for error tracking
- LogRocket for user session replay
- Mixpanel for analytics

### 4. Security Hardening

- [ ] Enable HTTPS (automatic on Streamlit Cloud)
- [ ] Implement rate limiting
- [ ] Add CAPTCHA for repeated failed login attempts
- [ ] Regular security audits

---

## Monitoring and Maintenance

### Check App Health

1. Monitor Streamlit Cloud dashboard for:
   - Memory usage
   - CPU usage
   - Error logs

2. Set up alerts for:
   - Failed deployments
   - High memory usage
   - Error rate spikes

### Update Dependencies

Regularly update requirements.txt:

```bash
pip install --upgrade pip setuptools wheel
pip list --outdated
pip install --upgrade <package-name>
```

---

## Deployment Commands (Alternative - Manual)

If you prefer to deploy using Streamlit CLI:

```bash
# Install Streamlit CLI
pip install streamlit

# Run locally first to test
streamlit run streamlit_app.py

# Deploy to Streamlit Cloud
streamlit deploy
```

---

## Support & Resources

- **Streamlit Documentation:** https://docs.streamlit.io
- **Streamlit Cloud Docs:** https://docs.streamlit.io/streamlit-cloud
- **GitHub Issues:** https://github.com/streamlit/streamlit/issues
- **Project Documentation:** See `docs/` folder

---

## Quick Checklist

- [ ] All files committed to GitHub
- [ ] `streamlit_app.py` exists at root
- [ ] `requirements.txt` updated with all dependencies
- [ ] Streamlit Cloud account created
- [ ] GitHub repository connected to Streamlit Cloud
- [ ] Secrets configured in Streamlit Cloud dashboard
- [ ] App deployed and running
- [ ] Authentication system tested
- [ ] Predictions working correctly
- [ ] Data loading from GitHub verified

---

## Next Steps

After successful deployment:

1. Share your app URL with stakeholders
2. Collect feedback and usage metrics
3. Plan scaling improvements
4. Set up continuous monitoring

Happy deploying! 🚀❤️
