# Energy Demand Forecasting — End-to-End MLOps System

<div align="center">

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://mlops-energy.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A production-ready MLOps system that forecasts US electricity demand with continuous learning, automated drift detection, and self-healing capabilities.**

[Live Demo](https://mlops-energy.streamlit.app) • [Documentation](DEPLOYMENT.md) • [Architecture](#-architecture)

</div>

---

## Project Overview

This is a complete end-to-end MLOps pipeline that demonstrates industry best practices for production machine learning systems:

### What It Does
- **Forecasts** electricity demand 24 hours ahead for the NY ISO grid region
- **Automatically ingests** hourly demand data (EIA API) and weather features (Open-Meteo)
- **Continuously learns** by detecting data drift and retraining models automatically
- **Tracks** all experiments and models with MLflow
- **Serves** predictions via FastAPI with Prometheus metrics
- **Monitors** model performance and data quality with Evidently AI
- **Orchestrates** all workflows with Prefect (scheduled and event-driven)
- **Visualizes** everything in an interactive Streamlit dashboard

### Why This Matters
Traditional ML projects stop at model training. This system handles the full production lifecycle:
- Real-world data ingestion from public APIs
- Feature engineering with lag variables and temporal features
- Automated drift detection (>30% = retrain trigger)
- Champion/Challenger model promotion
- RESTful API serving with monitoring
- Scheduled workflows running 24/7 in the cloud
- **100% free hosting** on GitHub Actions + Streamlit Cloud

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA INGESTION (Every 6h)                  │
│  EIA API (Demand) + Open-Meteo (Weather) → Feature Engineering  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DRIFT DETECTION (Daily 2 AM)                   │
│     Evidently AI: Compare recent vs reference data              │
│     Threshold: 30% drift → Trigger retraining                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ (if drift > 30%)
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING (On-Demand)                      │
│   LightGBM + MLflow → Champion/Challenger Promotion             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING & MONITORING                          │
│  FastAPI (Predictions) + Prometheus (Metrics) + Dashboard       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Orchestration**: Prefect flows with cron schedules
- **Storage**: Parquet files (features), SQLite (MLflow)
- **Monitoring**: Evidently (drift), Prometheus (API metrics)
- **Deployment**: GitHub Actions (workflows), Streamlit Cloud (dashboard)

---

## Live Demo

**Dashboard**: [https://mlops-energy.streamlit.app](https://mlops-energy.streamlit.app)

**What You'll See:**
- Real-time energy demand predictions vs actuals
- Drift detection gauge (currently 50% drift detected!)
- Error distribution histograms
- Top feature importance
- Recent predictions table
- Model performance metrics over time

---

## Quick Start

### Option 1: Use the Live System (No Setup!)
Just visit the [dashboard](https://mlops-energy.streamlit.app) — it's already running!

### Option 2: Run Locally

**Prerequisites:**
- Python 3.11+
- EIA API key ([Get free key](https://www.eia.gov/opendata/register.php))

**Setup:**

```bash
# 1. Clone and install dependencies
git clone https://github.com/APrabhu21/mlops-energy.git
cd mlops-energy
pip install -r requirements.txt

# 2. Set up API key
echo "EIA_API_KEY=your_key_here" > .env

# 3. Train initial model
python src/scripts/train_model.py

# 4. Start MLflow tracking server
python start_mlflow.bat  # Windows
# or: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 5. Launch dashboard
streamlit run src/dashboard/app.py

# 6. (Optional) Start FastAPI server
python start_api.bat  # Windows
# or: uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

**Dashboard will open at**: http://localhost:8501  
**API docs at**: http://localhost:8000/docs  
**MLflow at**: http://localhost:5000

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | LightGBM 4.0+ | Gradient boosting for demand forecasting |
| **Experiment Tracking** | MLflow 2.15+ | Model versioning & champion/challenger |
| **Drift Detection** | Evidently AI 0.7.20 | Statistical drift analysis |
| **Orchestration** | Prefect 3.0+ | Workflow scheduling & automation |
| **API Serving** | FastAPI + Uvicorn | RESTful prediction endpoints |
| **Monitoring** | Prometheus + Custom HTML | Metrics collection & visualization |
| **Dashboard** | Streamlit + Plotly | Interactive ML dashboard |
| **Data Storage** | Parquet + SQLite | Feature store & metadata |
| **CI/CD** | GitHub Actions | Automated workflows |
| **Hosting** | Streamlit Cloud (Free) | Zero-cost deployment |

---

## Project Structure

```
mlops-energy/
├── .github/workflows/       # GitHub Actions (data ingestion, drift checks)
├── src/
│   ├── data/               # EIA & weather API clients, feature engineering
│   ├── model/              # Training, evaluation, registry, predictions
│   ├── monitoring/         # Drift detection, performance tracking
│   ├── serving/            # FastAPI application & schemas
│   ├── orchestration/      # Prefect flows & tasks
│   ├── dashboard/          # Streamlit app
│   └── scripts/            # CLI tools (train, deploy, etc.)
├── data/
│   ├── raw/                # EIA demand & weather data (gitignored)
│   ├── processed/          # Engineered features (508 rows)
│   └── reference/          # Reference dataset for drift (4,732 rows)
├── reports/                # Drift HTML reports & JSON results
├── infra/                  # Docker configs (optional local setup)
├── requirements.txt        # Python dependencies
├── DEPLOYMENT.md          # Cloud deployment guide
└── README.md              # This file
```

---

## How It Works

### 1. Data Ingestion (Every 6 Hours)
```python
# Automated via GitHub Actions or Prefect
fetch_demand_data()     # EIA API → hourly demand (NY ISO)
fetch_weather_data()    # Open-Meteo → temperature, humidity, wind
build_features()        # Engineer 20+ features (lags, temporal, weather)
```

**Features Created** (20 total):
- **Temporal**: hour, day_of_week, month, is_weekend, is_holiday
- **Lag**: demand_lag_1h, demand_lag_24h, demand_lag_168h (weekly)
- **Rolling**: demand_rolling_24h_mean/std
- **Weather**: temperature, humidity, wind_speed, precipitation
- **Degree Days**: heating_degree_hours, cooling_degree_hours

### 2. Drift Detection (Daily at 2 AM UTC)
```python
# Compares recent 508 samples vs reference 4,732 samples
drift_share = run_drift_check(reference_df, current_df)
if drift_share > 0.30:  # 30% threshold
    trigger_retrain()
```

**Current Status**: 50% drift detected (10/20 features drifted)

### 3. Model Training (On Drift or Manual)
```python
model, metrics = train_model(features_df)
# Champion model: v1, MAE: 4,810 MWh, R²: 0.XX
promote_to_champion(run_id) if new_mae < champion_mae
```

### 4. Serving Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2026-02-08T12:00:00", "temperature": 15.5, ...}'
```

**API Endpoints:**
- `POST /predict` — Single prediction
- `POST /predict/batch` — Batch predictions
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics

---

## Results & Performance

**Model Metrics** (Champion v1):
- **MAE**: 4,810 MWh
- **RMSE**: ~6,000 MWh
- **R²**: High correlation with actuals
- **Inference**: <50ms per prediction

**System Stats**:
- **Uptime**: 24/7 via GitHub Actions
- **Data Points**: 508 recent + 4,732 reference
- **Drift Status**: 50% (retraining recommended)
- **Cost**: $0/month (free tier hosting)

---

## Cloud Deployment

This system runs **100% free** in the cloud! See [DEPLOYMENT.md](DEPLOYMENT.md) for full guide.

**Quick Deploy:**

1. **Fork this repo** on GitHub

2. **Deploy Dashboard** to Streamlit Cloud:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect repo: `APrabhu21/mlops-energy`
   - Main file: `src/dashboard/app.py`
   - Add secret: `EIA_API_KEY`

3. **Enable GitHub Actions**:
   - Repo → Settings → Actions → "Allow all actions"
   - Add secret: `EIA_API_KEY`
   - Workflows run automatically (data every 6h, drift daily)

**That's it!** Your MLOps system is live.

---

## Development

### Run Orchestration Flows

```bash
# Interactive menu
python src/scripts/run_flows.py

# Or run individually
python -c "from src.orchestration.flows import data_ingestion_flow; data_ingestion_flow()"
python -c "from src.orchestration.flows import drift_check_flow; drift_check_flow()"
```

### Test Predictions
```bash
python src/scripts/test_prediction.py
```

### Check Drift
```bash
python src/scripts/check_drift.py
# Generates HTML report in reports/ directory
```

### Run Tests
```bash
pytest tests/ -v  # (tests not included in this release)
```

---

## Learning Outcomes

This project demonstrates:
- **Production MLOps** patterns (not just notebooks)
- **Real-world data** from public APIs (EIA, Open-Meteo)
- **Automated pipelines** with scheduling and triggers
- **Model monitoring** with drift detection
- **Continuous learning** via champion/challenger
- **API design** for ML serving
- **Cloud deployment** on free tier
- **End-to-end ownership** (data → model → API → dashboard)

**Perfect for portfolio or interviews!**

---

## Screenshots

### Dashboard - Main View
![Dashboard Main](https://via.placeholder.com/800x400?text=Add+Screenshot)
*Real-time predictions, drift status, and performance metrics*

### Drift Detection Report
![Drift Report](https://via.placeholder.com/800x400?text=Add+Screenshot)
*50% drift detected across 10/20 features*

### MLflow Model Registry
![MLflow](https://via.placeholder.com/800x400?text=Add+Screenshot)
*Champion model tracking and versioning*

---

## Roadmap

Future enhancements:
- [ ] Multi-region support (CAISO, ERCOT, PJM)
- [ ] Ensemble models (LightGBM + XGBoost)
- [ ] A/B testing framework
- [ ] Real-time streaming with Apache Kafka
- [ ] Alert system (Slack/email notifications)
- [ ] Model explainability (SHAP values)
- [ ] Advanced feature engineering (external factors)
- [ ] PostgreSQL backend for production scale

---

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## License

MIT License - feel free to use this for learning or production!

---

## Acknowledgments

- **EIA (Energy Information Administration)** for electricity demand data
- **Open-Meteo** for weather API access
- **MLOps Community** for best practices and patterns
- **Evidently AI, MLflow, Prefect teams** for excellent open-source tools

---

## Contact

**Project by**: APrabhu21  
**GitHub**: [https://github.com/APrabhu21](https://github.com/APrabhu21)  
**Live Demo**: [https://mlops-energy.streamlit.app](https://mlops-energy.streamlit.app)

---

<div align="center">

**Star this repo if you found it helpful!**

[View Live Demo](https://mlops-energy.streamlit.app) • [Deploy Your Own](DEPLOYMENT.md) • [Report Issue](https://github.com/APrabhu21/mlops-energy/issues)

</div>
