# Energy Demand Forecasting — MLOps Pipeline

A production-grade MLOps system for forecasting US electricity demand with continuous learning, drift detection, and automated retraining.

## 🎯 Project Overview

This pipeline:
- **Ingests** hourly electricity demand (EIA API) and weather data (Open-Meteo)
- **Trains** a LightGBM model to forecast demand 24 hours ahead for the NY grid region
- **Serves** predictions via FastAPI with Prometheus metrics
- **Monitors** for data drift using Evidently AI
- **Automatically retrains** when drift is detected or performance degrades
- **Tracks** experiments and models in MLflow
- **Visualizes** metrics in Streamlit (ML) and Grafana (infrastructure)
- **Orchestrates** with Prefect flows

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- EIA API key (free): https://www.eia.gov/opendata/

### Setup

1. **Clone and setup environment:**
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your EIA_API_KEY
```

2. **Create data directories:**
```bash
mkdir -p data/raw data/processed data/reference reports
```

3. **Start infrastructure services:**
```bash
cd infra
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432)
- MLflow (port 5000)
- Prefect Server (port 4200)
- Prometheus (port 9090)
- Grafana (port 3000)

4. **Initialize database:**
```bash
# Run the init.sql script (automatically done by Docker Compose)
```

## 📊 Tech Stack

- **ML Framework**: LightGBM
- **Experiment Tracking**: MLflow
- **Drift Detection**: Evidently AI
- **Orchestration**: Prefect
- **Serving**: FastAPI + Uvicorn
- **Database**: PostgreSQL
- **Monitoring**: Prometheus + Grafana + Streamlit
- **Containerization**: Docker Compose

## 📁 Project Structure

```
energy-demand-mlops/
├── src/
│   ├── config.py              # Central configuration
│   ├── data/                  # Data ingestion & feature engineering
│   ├── model/                 # Training, evaluation, registry
│   ├── monitoring/            # Drift detection & metrics
│   ├── serving/               # FastAPI application
│   └── orchestration/         # Prefect flows
├── data/
│   ├── raw/                   # Raw EIA & weather data
│   ├── processed/             # Feature-engineered datasets
│   └── reference/             # Reference data for drift detection
├── infra/                     # Docker Compose & configs
├── tests/                     # Unit & integration tests
└── notebooks/                 # Exploratory analysis
```

## 🔄 Workflow

1. **Data Ingestion** (Prefect flow, every 6 hours)
   - Fetch demand from EIA API
   - Fetch weather from Open-Meteo
   - Engineer features
   - Store in PostgreSQL

2. **Drift Detection** (Prefect flow, daily)
   - Compare recent data vs reference
   - Calculate drift metrics
   - Trigger retrain if drift > threshold

3. **Model Training** (triggered by drift or manual)
   - Train LightGBM on all available data
   - Log to MLflow
   - Promote to production if better than champion

4. **Serving** (FastAPI, always on)
   - Load champion model from MLflow
   - Serve predictions
   - Log to PostgreSQL + Prometheus

## 📈 Dashboards

- **MLflow UI**: http://localhost:5000 - Experiment tracking & model registry
- **Prefect UI**: http://localhost:4200 - Flow orchestration
- **Grafana**: http://localhost:3000 - Infrastructure metrics (admin/admin)
- **Streamlit**: Run `streamlit run src/dashboard/app.py` - ML metrics & drift reports
- **API Docs**: http://localhost:8000/docs - FastAPI interactive docs

## 🧪 Development

### Test data ingestion (small sample):
```bash
python -c "from src.data.eia_client import EIAClient; \
client = EIAClient(); \
df = client.fetch_demand('2024-01-01T00', '2024-01-07T23'); \
print(df.head())"
```

### Run tests:
```bash
pytest tests/ -v
```

### Lint:
```bash
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

## 📝 License

Open source - see individual component licenses in the specification document.

## 🙏 Acknowledgments

- EIA for electricity demand data
- Open-Meteo for weather data
- MLOps community for best practices
