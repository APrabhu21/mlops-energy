# 🚀 Free Cloud Deployment Guide

This guide shows how to deploy the Energy Demand MLOps system to free cloud services.

## 📋 Prerequisites

1. GitHub account
2. Get your **EIA API key**: https://www.eia.gov/opendata/register.php (free)
3. Sign up for free cloud services (no credit card needed):
   - [Streamlit Community Cloud](https://streamlit.io/cloud)
   - [Render.com](https://render.com) (optional, for API)
   - [Prefect Cloud](https://prefect.io/cloud) (optional, for orchestration)

---

## 🎯 Deployment Option 1: Streamlit Cloud (Easiest)

**Best for:** Dashboard-only deployment with scheduled GitHub Actions

### Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for cloud deployment"
   git push origin main
   ```

2. **Deploy Dashboard**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repo: `your-username/mlops-energy`
   - Main file: `src/dashboard/app.py`
   - Click "Deploy"

3. **Add Secrets** (in Streamlit Cloud dashboard)
   - Click "⋮" → "Settings" → "Secrets"
   - Add:
   ```toml
   [general]
   EIA_API_KEY = "your_key_here"
   
   [mlflow]
   TRACKING_URI = "sqlite:///mlflow.db"
   ```

4. **Enable GitHub Actions** (for scheduled data updates)
   - In your GitHub repo: Settings → Secrets → Actions
   - Add secret: `EIA_API_KEY` = your key
   - GitHub Actions will run automatically every 6 hours

### ✅ Result:
- Dashboard live at: `https://your-app.streamlit.app`
- Data updates every 6 hours via GitHub Actions
- Drift checks daily at 2 AM UTC
- Auto-retraining when drift detected

---

## 🎯 Deployment Option 2: Streamlit + Render API

**Best for:** Full system with FastAPI + Dashboard

### Steps:

1. **Deploy API on Render**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn src.serving.app:app --host 0.0.0.0 --port $PORT`
     - **Environment Variables**:
       - `EIA_API_KEY` = your key
       - `MLFLOW_TRACKING_URI` = `sqlite:///mlflow.db`

2. **Deploy Dashboard on Streamlit Cloud**
   - Follow Option 1 steps above
   - Your API will be at: `https://your-app.onrender.com`

---

## 🎯 Deployment Option 3: Prefect Cloud (Advanced)

**Best for:** Enterprise-grade orchestration

### Steps:

1. **Sign up for Prefect Cloud**
   - Go to https://app.prefect.cloud
   - Create free account (20k task runs/month)

2. **Login locally**
   ```bash
   prefect cloud login
   ```

3. **Deploy flows**
   ```bash
   python src/scripts/deploy_flows.py
   ```

4. **Set schedules in Prefect UI**
   - Data ingestion: Every 6 hours
   - Drift check: Daily at 2 AM
   - Performance monitoring: Daily at 3 AM

---

## 📊 GitHub Actions (Included)

Two workflows are configured:

### 1. Data Ingestion (`.github/workflows/data_ingestion.yml`)
- Runs every 6 hours
- Fetches EIA + weather data
- Builds features
- Saves artifacts

### 2. Drift Detection (`.github/workflows/drift_retrain.yml`)
- Runs daily at 2 AM UTC
- Checks for data drift
- Auto-retrains if drift > 30%
- Promotes model if better than champion

**Enable them:**
1. GitHub repo → Settings → Actions → General
2. Enable "Allow all actions"
3. Add secret: `EIA_API_KEY`

---

## 🔑 Environment Variables

### Required:
- `EIA_API_KEY` - Get from https://www.eia.gov/opendata/register.php

### Optional:
- `MLFLOW_TRACKING_URI` - Default: `sqlite:///mlflow.db`
- `DRIFT_SHARE_THRESHOLD` - Default: `0.3` (30%)

---

## 📝 Post-Deployment Checklist

- [ ] Dashboard accessible at public URL
- [ ] GitHub Actions running successfully
- [ ] Secrets configured in Streamlit Cloud
- [ ] EIA API key working
- [ ] Data updates appearing in dashboard
- [ ] Drift detection generating reports

---

## 🐛 Troubleshooting

### Dashboard shows "No data available"
- Run data ingestion workflow manually in GitHub Actions
- Or commit initial data files to repo

### GitHub Actions failing
- Check EIA API key is set in repo secrets
- Verify workflows are enabled in repo settings

### Streamlit "Module not found" errors
- Clear cache: Settings → Clear cache
- Verify `requirements.txt` includes all dependencies

---

## 💰 Cost Breakdown

| Service | Free Tier | Cost |
|---------|-----------|------|
| Streamlit Cloud | 1 app | **$0** |
| Render.com | 750 hrs/month | **$0** |
| GitHub Actions | 2000 min/month | **$0** |
| Prefect Cloud | 20k runs/month | **$0** |
| EIA API | Unlimited | **$0** |
| Open-Meteo API | Unlimited | **$0** |
| **TOTAL** | | **$0/month** |

---

## 🔄 Alternative: DagsHub (MLflow Hosting)

Instead of SQLite, use free MLflow hosting:

1. Sign up at https://dagshub.com
2. Create new repo
3. Get MLflow tracking URL
4. Update `.streamlit/secrets.toml`:
   ```toml
   [mlflow]
   TRACKING_URI = "https://dagshub.com/your-username/your-repo.mlflow"
   ```

---

## 📚 Next Steps

After deployment:
1. Monitor dashboard daily
2. Check GitHub Actions logs
3. Review drift reports
4. Track model performance over time
5. Tune hyperparameters if needed

---

## 🆘 Support

- Streamlit docs: https://docs.streamlit.io
- Render docs: https://render.com/docs
- Prefect docs: https://docs.prefect.io
- GitHub Actions: https://docs.github.com/actions

---

**🎉 Congratulations!** Your MLOps system is now running in the cloud for free!
