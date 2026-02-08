# Energy Demand Forecasting — Full MLOps Pipeline

## PROJECT SPECIFICATION & IMPLEMENTATION GUIDE

> **Purpose**: This document is a comprehensive, self-contained specification for building an end-to-end ML system that forecasts US electricity demand with continuous learning, drift detection, model observability, and automated retraining. Every component is fully open-source. Use this as a prompt/reference for GitHub Copilot or any AI coding assistant to scaffold and implement the entire project.

### IMPLEMENTATION DIRECTIVES (READ FIRST)

These are binding decisions that override any defaults in the spec below:

1. **Build from scratch** — empty workspace, no existing components.
2. **EIA API Key**: Use placeholder `EIA_API_KEY` env var. Code MUST handle a missing key gracefully with a clear error message (e.g., `raise ValueError("EIA_API_KEY not set. Register at https://www.eia.gov/opendata/")`).
3. **Deployment target**: Entirely local with Docker Compose. No cloud deployment.
4. **Implementation order**: Minimal viable pipeline first, then iterate. Order: data ingestion → feature engineering → training → serving → drift detection → orchestration → dashboards. Get a working end-to-end loop before polishing.
5. **Testing**: Core functionality first. Do NOT write tests until Phase 2 (model training) is complete. Then add tests iteratively.
6. **Region**: New York ISO (`NY`). Do not change.
7. **Development style**: Hybrid — local Python dev for data/model work, Docker Compose for infrastructure services (Postgres, MLflow, Prometheus, Grafana). The FastAPI service runs locally during dev, containerize later.
8. **DVC**: Local only. No remote storage configuration.
9. **Streamlit dashboard**: Add a Streamlit app for ML-specific visualizations (actual vs predicted plots, drift reports, model metrics, feature importance). Keep Grafana only for Prometheus infrastructure metrics. Streamlit is the primary ML dashboard.
10. **Timeline**: ~10 days of focused work.

---

## 1. PROJECT OVERVIEW

### 1.1 What We're Building

A production-grade MLOps pipeline that:

1. **Ingests** hourly US electricity demand data from the EIA (Energy Information Administration) API and weather data from Open-Meteo API
2. **Trains** a gradient boosting model (LightGBM) to forecast electricity demand 24 hours ahead for a specific US grid region
3. **Serves** predictions via a FastAPI REST endpoint inside a Docker container
4. **Monitors** for data drift and model performance degradation using Evidently AI
5. **Automatically retrains** the model when drift is detected or performance drops below a threshold
6. **Tracks** all experiments, metrics, and model versions in MLflow
7. **Visualizes** system health, drift scores, and prediction metrics in Grafana dashboards
8. **Orchestrates** everything with Prefect (DAGs for ingestion, drift checks, retraining)

### 1.2 Why Energy Demand

Energy demand naturally drifts due to:
- Seasonal temperature changes (summer AC, winter heating)
- Renewable energy penetration changing grid dynamics year-over-year
- Policy changes, EV adoption, industrial shifts
- Extreme weather events (heat waves, polar vortices)

This means drift detection and continuous learning are genuinely necessary, not simulated.

### 1.3 Tech Stack (All Open Source)

| Layer | Tool | Version | License |
|-------|------|---------|---------|
| Language | Python | 3.11+ | PSF |
| ML Model | LightGBM | latest | MIT |
| Experiment Tracking | MLflow | latest | Apache 2.0 |
| Model Registry | MLflow Model Registry | (included) | Apache 2.0 |
| Drift Detection | Evidently AI | latest | Apache 2.0 |
| Orchestration | Prefect | 3.x (OSS) | Apache 2.0 |
| Model Serving | FastAPI + Uvicorn | latest | MIT / BSD |
| Database | PostgreSQL | 16 | PostgreSQL License |
| Metrics Collection | Prometheus | latest | Apache 2.0 |
| Dashboards | Grafana | latest (OSS) | AGPL 3.0 |
| ML Dashboard | Streamlit | latest | Apache 2.0 |
| Containerization | Docker + Docker Compose | latest | Apache 2.0 |
| Data Versioning | DVC | latest | Apache 2.0 |
| CI/CD | GitHub Actions | free tier | Proprietary (free) |
| Log Aggregation | Loki (optional) | latest | AGPL 3.0 |

---

## 2. REPOSITORY STRUCTURE

```
energy-demand-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint, test, type-check on PR
│       └── retrain-deploy.yml        # Manual/scheduled retrain + deploy
├── data/
│   ├── raw/                          # DVC-tracked raw data
│   ├── processed/                    # DVC-tracked feature-engineered data
│   └── reference/                    # Reference dataset for drift detection
├── src/
│   ├── __init__.py
│   ├── config.py                     # All configuration (env vars, constants)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── eia_client.py             # EIA API v2 client
│   │   ├── weather_client.py         # Open-Meteo API client
│   │   ├── ingest.py                 # Orchestrated data ingestion logic
│   │   └── feature_engineering.py    # Feature building from raw data
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py                  # Training pipeline
│   │   ├── evaluate.py               # Evaluation metrics
│   │   ├── predict.py                # Inference logic
│   │   └── registry.py               # MLflow model registry helpers
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detection.py        # Evidently drift reports & checks
│   │   ├── performance_monitor.py    # Track prediction quality over time
│   │   └── prometheus_metrics.py     # Custom Prometheus metrics
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI application
│   │   ├── schemas.py                # Pydantic request/response models
│   │   └── middleware.py             # Logging, metrics middleware
│   └── orchestration/
│       ├── __init__.py
│       ├── flows.py                  # Prefect flows (ingest, drift, retrain)
│       └── deployments.py            # Prefect deployment configs
├── dashboard/
│   ├── app.py                        # Streamlit ML dashboard (primary)
│   ├── pages/
│   │   ├── 1_📊_Model_Performance.py  # Actual vs predicted, MAE over time
│   │   ├── 2_🔄_Drift_Monitor.py      # Evidently drift reports & history
│   │   ├── 3_🏋️_Training_History.py   # MLflow experiment comparison
│   │   └── 4_📈_Feature_Importance.py  # SHAP / LightGBM importance
│   └── utils.py                      # DB queries, MLflow helpers for dashboard
├── tests/
│   ├── test_data/
│   │   ├── test_eia_client.py
│   │   ├── test_weather_client.py
│   │   └── test_feature_engineering.py
│   ├── test_model/
│   │   ├── test_train.py
│   │   └── test_evaluate.py
│   ├── test_monitoring/
│   │   └── test_drift_detection.py
│   └── test_serving/
│       └── test_app.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   └── 04_drift_analysis.ipynb
├── infra/
│   ├── docker-compose.yml            # All services
│   ├── Dockerfile.api                # FastAPI serving image
│   ├── Dockerfile.worker             # Prefect worker image
│   ├── prometheus/
│   │   └── prometheus.yml            # Prometheus scrape config
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   ├── datasources/
│   │   │   │   └── datasources.yml
│   │   │   └── dashboards/
│   │   │       ├── dashboards.yml
│   │   │       └── energy-demand.json  # Pre-built dashboard
│   │   └── grafana.ini
│   └── postgres/
│       └── init.sql                  # DB schema initialization
├── .dvc/
│   └── config
├── .env.example                      # Template for environment variables
├── pyproject.toml                    # Project config + dependencies
├── requirements.txt                  # Pinned dependencies
├── Makefile                          # Common commands
├── README.md
└── dvc.yaml                          # DVC pipeline stages
```

---

## 3. DATA SOURCES — DETAILED API SPECIFICATIONS

### 3.1 EIA API v2 — Electricity Demand

**Registration**: Go to https://www.eia.gov/opendata/ → Register → Get free API key (no credit card).

**Base URL**: `https://api.eia.gov/v2/`

**Endpoint for Hourly Demand by Region**:
```
GET https://api.eia.gov/v2/electricity/rto/region-data/data/
```

**Parameters**:
```python
params = {
    "api_key": EIA_API_KEY,
    "frequency": "hourly",
    "data[0]": "value",                          # demand in MWh
    "facets[respondent][]": "NY",                 # New York ISO (pick one region)
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 5000,                               # max per request
    "start": "2023-01-01T00",                     # ISO format
    "end": "2025-12-31T23",
}
```

**Available Regions (respondent facet values)** — pick ONE to keep scope manageable:
- `NY` — New York ISO
- `CAL` — California ISO
- `TEX` — Texas (ERCOT)
- `MIDA` — Mid-Atlantic (PJM)
- `US48` — Lower 48 (aggregate)

**Recommended**: Use `NY` (New York ISO). Good seasonal variation, manageable size, well-documented.

**Response Structure**:
```json
{
  "response": {
    "total": 52000,
    "data": [
      {
        "period": "2025-01-15T14",
        "respondent": "NY",
        "respondent-name": "New York Independent System Operator",
        "value": 21345,
        "value-units": "megawatthours"
      }
    ]
  }
}
```

**Pagination**: The API returns max 5000 rows per request. Use `offset` parameter to paginate:
```python
# Pseudocode for full data pull
all_data = []
offset = 0
while True:
    params["offset"] = offset
    response = requests.get(url, params=params)
    data = response.json()["response"]["data"]
    if not data:
        break
    all_data.extend(data)
    offset += 5000
```

**Rate Limits**: Throttle to ~1 request/second. Key gets temporarily suspended if exceeded.

**Data Availability**: Hourly data available from ~2019 to present, updated with ~1-2 day lag.

### 3.2 Open-Meteo API — Weather Data

**No API key required** for non-commercial use (<10,000 calls/day).

**Historical Weather Endpoint**:
```
GET https://archive-api.open-meteo.com/v1/archive
```

**Forecast Weather Endpoint** (for live predictions):
```
GET https://api.open-meteo.com/v1/forecast
```

**Parameters for Historical Data**:
```python
params = {
    "latitude": 40.7128,        # NYC coordinates
    "longitude": -74.0060,
    "start_date": "2023-01-01",
    "end_date": "2025-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "dewpoint_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "surface_pressure",
        "shortwave_radiation",     # solar radiation — important for demand
    ],
    "timezone": "America/New_York",
}
```

**Response Structure**:
```json
{
  "hourly": {
    "time": ["2023-01-01T00:00", "2023-01-01T01:00", ...],
    "temperature_2m": [2.1, 1.8, ...],
    "relative_humidity_2m": [85, 87, ...],
    ...
  }
}
```

**For Live Forecasts** (used during serving/inference):
```python
forecast_params = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
               "apparent_temperature", "cloud_cover", "shortwave_radiation"],
    "forecast_days": 2,
    "timezone": "America/New_York",
}
# GET https://api.open-meteo.com/v1/forecast?{params}
```

---

## 4. DATABASE SCHEMA

Use PostgreSQL for all persistent storage.

### 4.1 init.sql

```sql
-- Raw demand data from EIA
CREATE TABLE IF NOT EXISTS raw_demand (
    id SERIAL PRIMARY KEY,
    period TIMESTAMP NOT NULL,
    respondent VARCHAR(10) NOT NULL,
    value FLOAT NOT NULL,                -- MWh
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(period, respondent)
);
CREATE INDEX idx_demand_period ON raw_demand(period);

-- Raw weather data from Open-Meteo
CREATE TABLE IF NOT EXISTS raw_weather (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    temperature_2m FLOAT,
    relative_humidity_2m FLOAT,
    dewpoint_2m FLOAT,
    apparent_temperature FLOAT,
    precipitation FLOAT,
    snowfall FLOAT,
    cloud_cover FLOAT,
    wind_speed_10m FLOAT,
    wind_gusts_10m FLOAT,
    surface_pressure FLOAT,
    shortwave_radiation FLOAT,
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(timestamp)
);
CREATE INDEX idx_weather_timestamp ON raw_weather(timestamp);

-- Feature-engineered dataset (joined demand + weather + time features)
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL UNIQUE,
    -- target
    demand_mwh FLOAT NOT NULL,
    -- weather features
    temperature FLOAT,
    apparent_temperature FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    cloud_cover FLOAT,
    solar_radiation FLOAT,
    precipitation FLOAT,
    -- time features
    hour_of_day INT,
    day_of_week INT,
    month INT,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    -- lag features
    demand_lag_1h FLOAT,
    demand_lag_24h FLOAT,
    demand_lag_168h FLOAT,           -- 1 week ago
    demand_rolling_mean_24h FLOAT,
    demand_rolling_std_24h FLOAT,
    -- temperature interaction
    temp_squared FLOAT,              -- captures nonlinear heating/cooling
    cooling_degree_hours FLOAT,      -- max(0, temp - 65)
    heating_degree_hours FLOAT,      -- max(0, 65 - temp)
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_features_timestamp ON features(timestamp);

-- Prediction log (for monitoring)
CREATE TABLE IF NOT EXISTS prediction_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,         -- prediction target time
    predicted_demand FLOAT NOT NULL,
    actual_demand FLOAT,                  -- filled in later when actual arrives
    model_version VARCHAR(50),
    prediction_made_at TIMESTAMP DEFAULT NOW(),
    features_json JSONB                   -- snapshot of input features
);
CREATE INDEX idx_predictions_timestamp ON prediction_log(timestamp);

-- Drift detection results
CREATE TABLE IF NOT EXISTS drift_log (
    id SERIAL PRIMARY KEY,
    check_timestamp TIMESTAMP DEFAULT NOW(),
    dataset_drift_detected BOOLEAN,
    drift_share FLOAT,                    -- % of features drifted
    n_drifted_features INT,
    n_total_features INT,
    drift_details JSONB,                  -- per-feature drift scores
    triggered_retrain BOOLEAN DEFAULT FALSE
);
```

---

## 5. IMPLEMENTATION DETAILS — MODULE BY MODULE

### 5.1 Configuration — `src/config.py`

```python
"""
Central configuration. All values come from environment variables
with sensible defaults for local development.
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

# EIA API
EIA_API_KEY = os.getenv("EIA_API_KEY")  # required
EIA_BASE_URL = "https://api.eia.gov/v2/"
EIA_REGION = os.getenv("EIA_REGION", "NY")  # default New York ISO

# Open-Meteo (no key needed)
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# NYC coordinates (adjust if using a different EIA region)
WEATHER_LATITUDE = float(os.getenv("WEATHER_LAT", "40.7128"))
WEATHER_LONGITUDE = float(os.getenv("WEATHER_LON", "-74.0060"))
WEATHER_TIMEZONE = os.getenv("WEATHER_TZ", "America/New_York")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://mlops:mlops@localhost:5432/energy_mlops")

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "energy-demand-forecast"

# Model
MODEL_NAME = "energy-demand-lgbm"  # name in MLflow registry
FORECAST_HORIZON_HOURS = 24
TARGET_COLUMN = "demand_mwh"
FEATURE_COLUMNS = [
    "temperature", "apparent_temperature", "humidity", "wind_speed",
    "cloud_cover", "solar_radiation", "precipitation",
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
    "demand_rolling_mean_24h", "demand_rolling_std_24h",
    "temp_squared", "cooling_degree_hours", "heating_degree_hours",
]

# Drift detection thresholds
DRIFT_SHARE_THRESHOLD = 0.3       # if >30% of features drift, trigger retrain
PERFORMANCE_MAE_THRESHOLD = 1500  # MAE in MWh above which we retrain

# Prometheus
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))

# Serving
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
```

### 5.2 Data Ingestion — `src/data/eia_client.py`

```python
"""
Client for EIA API v2 — fetches hourly electricity demand data.
Handles pagination (5000 row limit per request) and rate limiting.
"""
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
from src.config import EIA_API_KEY, EIA_BASE_URL, EIA_REGION

class EIAClient:
    ENDPOINT = "electricity/rto/region-data/data/"
    MAX_LENGTH = 5000
    RATE_LIMIT_DELAY = 1.0  # seconds between requests

    def __init__(self, api_key: str = EIA_API_KEY, region: str = EIA_REGION):
        self.api_key = api_key
        self.region = region
        self.base_url = EIA_BASE_URL + self.ENDPOINT

    def fetch_demand(
        self,
        start: str,   # format: "2023-01-01T00"
        end: str,      # format: "2025-12-31T23"
    ) -> pd.DataFrame:
        """
        Fetch all hourly demand data between start and end.
        Handles pagination automatically.
        Returns DataFrame with columns: [period, respondent, value]
        """
        all_records = []
        offset = 0

        while True:
            params = {
                "api_key": self.api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": self.region,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": self.MAX_LENGTH,
                "start": start,
                "end": end,
            }

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()["response"]["data"]

            if not data:
                break

            all_records.extend(data)
            offset += self.MAX_LENGTH

            if len(data) < self.MAX_LENGTH:
                break

            time.sleep(self.RATE_LIMIT_DELAY)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df["period"] = pd.to_datetime(df["period"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.rename(columns={"value": "demand_mwh"})
        return df

    def fetch_incremental(self, last_timestamp: datetime) -> pd.DataFrame:
        """Fetch only new data since last_timestamp."""
        start = last_timestamp.strftime("%Y-%m-%dT%H")
        end = datetime.utcnow().strftime("%Y-%m-%dT%H")
        return self.fetch_demand(start=start, end=end)
```

### 5.3 Data Ingestion — `src/data/weather_client.py`

```python
"""
Client for Open-Meteo API — fetches historical and forecast weather data.
No API key required for non-commercial use.
"""
import requests
import pandas as pd
from typing import List, Optional
from src.config import (
    OPEN_METEO_HISTORICAL_URL, OPEN_METEO_FORECAST_URL,
    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_TIMEZONE,
)

HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "surface_pressure", "shortwave_radiation",
]

class WeatherClient:
    def __init__(
        self,
        latitude: float = WEATHER_LATITUDE,
        longitude: float = WEATHER_LONGITUDE,
        timezone: str = WEATHER_TIMEZONE,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone

    def fetch_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical hourly weather data.
        start_date/end_date format: "2023-01-01"
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": self.timezone,
        }
        response = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()["hourly"]

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"time": "timestamp"})
        return df

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch current weather forecast for the next N days."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": self.timezone,
        }
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()["hourly"]

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"time": "timestamp"})
        return df
```

### 5.4 Feature Engineering — `src/data/feature_engineering.py`

```python
"""
Feature engineering pipeline.
Joins demand + weather data, creates time features, lag features,
and temperature interaction terms.
"""
import pandas as pd
import numpy as np
from typing import Optional

# US federal holidays — use the 'holidays' library
import holidays

US_HOLIDAYS = holidays.US()

def build_features(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join demand and weather data, then engineer features.

    demand_df must have columns: [period, demand_mwh]
    weather_df must have columns: [timestamp, temperature_2m, ...]

    Returns a DataFrame ready for model training with all feature columns.
    """
    # Standardize column names
    demand = demand_df[["period", "demand_mwh"]].copy()
    demand = demand.rename(columns={"period": "timestamp"})
    demand["timestamp"] = pd.to_datetime(demand["timestamp"]).dt.floor("h")

    weather = weather_df.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"]).dt.floor("h")

    # Merge on timestamp
    df = pd.merge(demand, weather, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rename weather columns to cleaner names
    rename_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "apparent_temperature": "apparent_temperature",
        "wind_speed_10m": "wind_speed",
        "cloud_cover": "cloud_cover",
        "shortwave_radiation": "solar_radiation",
        "precipitation": "precipitation",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ---- Time features ----
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek        # 0=Monday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_holiday"] = df["timestamp"].dt.date.apply(lambda d: d in US_HOLIDAYS)

    # ---- Lag features ----
    df["demand_lag_1h"] = df["demand_mwh"].shift(1)
    df["demand_lag_24h"] = df["demand_mwh"].shift(24)
    df["demand_lag_168h"] = df["demand_mwh"].shift(168)       # 1 week

    # ---- Rolling features ----
    df["demand_rolling_mean_24h"] = df["demand_mwh"].rolling(24).mean()
    df["demand_rolling_std_24h"] = df["demand_mwh"].rolling(24).std()

    # ---- Temperature interaction features ----
    # These capture the nonlinear relationship between temperature and demand
    # (both very hot and very cold increase demand)
    df["temp_squared"] = df["temperature"] ** 2
    # Cooling degree hours: how much above 65°F (~18.3°C)
    df["cooling_degree_hours"] = np.maximum(0, df["temperature"] - 18.3)
    # Heating degree hours: how much below 65°F (~18.3°C)
    df["heating_degree_hours"] = np.maximum(0, 18.3 - df["temperature"])

    # Drop rows with NaN from lag/rolling features
    df = df.dropna().reset_index(drop=True)

    return df
```

### 5.5 Model Training — `src/model/train.py`

```python
"""
Training pipeline for LightGBM demand forecast model.
Logs everything to MLflow: params, metrics, model artifact, feature importance.
"""
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_NAME,
    TARGET_COLUMN, FEATURE_COLUMNS,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# LightGBM hyperparameters
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}


def train_model(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    test_size: int = 168 * 4,  # last 4 weeks as test
) -> Tuple[lgb.LGBMRegressor, Dict[str, float]]:
    """
    Train a LightGBM model on the provided feature dataframe.

    Uses time-based split (no shuffling — respects temporal order).
    Logs everything to MLflow.

    Returns:
        (trained_model, metrics_dict)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Time-based train/test split (no shuffle!)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", FEATURE_COLUMNS)
        mlflow.log_param("train_start", str(df["timestamp"].iloc[0]))
        mlflow.log_param("train_end", str(df["timestamp"].iloc[-test_size - 1]))
        mlflow.log_param("test_start", str(df["timestamp"].iloc[-test_size]))
        mlflow.log_param("test_end", str(df["timestamp"].iloc[-1]))

        # Train
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        # Predict & evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log feature importance
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        mlflow.log_text(importance_df.to_csv(index=False), "feature_importance.csv")

        # Log model to MLflow
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")

    return model, metrics
```

### 5.6 Model Registry Helpers — `src/model/registry.py`

```python
"""
Helpers for MLflow Model Registry — champion/challenger pattern.
"""
import mlflow
from mlflow.tracking import MlflowClient
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def get_champion_model():
    """Load the current Production (champion) model."""
    model_uri = f"models:/{MODEL_NAME}@champion"
    try:
        return mlflow.lightgbm.load_model(model_uri)
    except Exception:
        # Fallback: get latest version
        model_uri = f"models:/{MODEL_NAME}/latest"
        return mlflow.lightgbm.load_model(model_uri)


def promote_to_champion(run_id: str, current_mae: float) -> bool:
    """
    Compare the new model (from run_id) against the current champion.
    Promote if the new model has lower MAE.

    Returns True if promoted.
    """
    # Get the new model's MAE
    new_run = client.get_run(run_id)
    new_mae = float(new_run.data.metrics.get("mae", float("inf")))

    # Get the current champion's MAE
    try:
        champion_versions = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run_id = champion_versions.run_id
        champion_run = client.get_run(champion_run_id)
        champion_mae = float(champion_run.data.metrics.get("mae", float("inf")))
    except Exception:
        # No champion exists yet — promote automatically
        champion_mae = float("inf")

    if new_mae < champion_mae:
        # Get the latest version number for this run
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            version = versions[0].version
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            print(f"Promoted version {version} to champion (MAE: {new_mae:.1f} < {champion_mae:.1f})")
            return True

    print(f"New model NOT promoted (MAE: {new_mae:.1f} >= {champion_mae:.1f})")
    return False
```

### 5.7 Drift Detection — `src/monitoring/drift_detection.py`

```python
"""
Drift detection using Evidently AI.
Compares current production data against the reference (training) dataset.
Generates reports and extracts metrics for Prometheus/Grafana.
"""
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataQualityPreset
from typing import Dict, Tuple
from src.config import FEATURE_COLUMNS, DRIFT_SHARE_THRESHOLD, REFERENCE_DIR


def run_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list = FEATURE_COLUMNS,
) -> Tuple[bool, float, Dict]:
    """
    Run Evidently data drift detection.

    Args:
        reference_df: Historical/training data (baseline)
        current_df: Recent production data to check for drift

    Returns:
        (drift_detected: bool, drift_share: float, details: dict)
    """
    # Use only feature columns for drift analysis
    ref = reference_df[feature_columns]
    cur = current_df[feature_columns]

    # Create and run drift report
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    # Extract results as dict
    result_dict = result.as_dict()

    # Parse drift results
    # The structure may vary by Evidently version — adapt accordingly
    drift_results = result_dict.get("metrics", [])
    dataset_drift = False
    drift_share = 0.0
    per_feature_drift = {}

    for metric in drift_results:
        metric_id = metric.get("metric", "")
        if "DatasetDrift" in metric_id:
            dataset_drift = metric["result"].get("dataset_drift", False)
            drift_share = metric["result"].get("drift_share", 0.0)
        if "ColumnDrift" in metric_id:
            col_name = metric["result"].get("column_name", "")
            col_drift = metric["result"].get("drift_detected", False)
            col_score = metric["result"].get("drift_score", 0.0)
            per_feature_drift[col_name] = {
                "drifted": col_drift,
                "score": col_score,
            }

    # Decision: retrain if drift_share exceeds threshold
    should_retrain = drift_share > DRIFT_SHARE_THRESHOLD

    return should_retrain, drift_share, {
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "per_feature": per_feature_drift,
    }


def generate_drift_html_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
) -> str:
    """Generate and save an HTML drift report for visual inspection."""
    report = Report([DataDriftPreset(), DataQualityPreset()])
    result = report.run(reference_data=reference_df, current_data=current_df)
    result.save_html(output_path)
    return output_path
```

### 5.8 Prometheus Metrics — `src/monitoring/prometheus_metrics.py`

```python
"""
Custom Prometheus metrics for model observability.
These are scraped by Prometheus and visualized in Grafana.
"""
from prometheus_client import (
    Counter, Histogram, Gauge, start_http_server,
)
from src.config import PROMETHEUS_PORT

# --- Prediction metrics ---
PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
PREDICTION_VALUE = Histogram(
    "predicted_demand_mwh",
    "Distribution of predicted demand values",
    buckets=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
)

# --- Model quality metrics (updated by monitoring jobs) ---
MODEL_MAE = Gauge(
    "model_mae_mwh",
    "Current model MAE on recent actuals",
)
MODEL_MAPE = Gauge(
    "model_mape_percent",
    "Current model MAPE on recent actuals",
)

# --- Drift metrics (updated by drift check jobs) ---
DRIFT_SHARE = Gauge(
    "drift_share",
    "Fraction of features with detected drift",
)
DRIFT_DETECTED = Gauge(
    "drift_detected",
    "1 if dataset drift detected, 0 otherwise",
)

# --- Data freshness ---
DATA_FRESHNESS_HOURS = Gauge(
    "data_freshness_hours",
    "Hours since the most recent data point was ingested",
)

# --- Retrain tracking ---
RETRAIN_COUNTER = Counter(
    "retrain_triggered_total",
    "Total number of automatic retraining events",
)


def start_metrics_server(port: int = PROMETHEUS_PORT):
    """Start the Prometheus metrics HTTP server."""
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
```

### 5.9 FastAPI Serving — `src/serving/app.py`

```python
"""
FastAPI prediction service.
Loads the champion model from MLflow registry.
Logs every prediction to PostgreSQL and updates Prometheus metrics.
"""
import time
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import psycopg2

from src.model.registry import get_champion_model
from src.monitoring.prometheus_metrics import (
    PREDICTION_COUNTER, PREDICTION_LATENCY, PREDICTION_VALUE,
    start_metrics_server,
)
from src.config import DATABASE_URL, FEATURE_COLUMNS

# Global model reference
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, start metrics server."""
    global model
    model = get_champion_model()
    start_metrics_server()
    print("Model loaded, metrics server started.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Energy Demand Forecast API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Input features for prediction. All fields correspond to feature columns."""
    temperature: float
    apparent_temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float
    solar_radiation: float
    precipitation: float
    hour_of_day: int
    day_of_week: int
    month: int
    is_weekend: bool
    is_holiday: bool
    demand_lag_1h: float
    demand_lag_24h: float
    demand_lag_168h: float
    demand_rolling_mean_24h: float
    demand_rolling_std_24h: float
    temp_squared: float
    cooling_degree_hours: float
    heating_degree_hours: float


class PredictionResponse(BaseModel):
    predicted_demand_mwh: float
    model_version: str
    timestamp: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate a demand forecast from input features."""
    start_time = time.time()

    try:
        # Build feature DataFrame
        features = pd.DataFrame([request.model_dump()])
        features = features[FEATURE_COLUMNS]

        # Predict
        prediction = float(model.predict(features)[0])

        # Update Prometheus metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_VALUE.observe(prediction)

        response = PredictionResponse(
            predicted_demand_mwh=round(prediction, 2),
            model_version="champion",
            timestamp=datetime.utcnow().isoformat(),
        )

        # Log to database (async in production, sync here for simplicity)
        _log_prediction(request, prediction)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/reload-model")
async def reload_model():
    """Hot-reload the champion model without restarting the service."""
    global model
    model = get_champion_model()
    return {"status": "model reloaded"}


def _log_prediction(request: PredictionRequest, prediction: float):
    """Log prediction to PostgreSQL for monitoring."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO prediction_log
            (timestamp, predicted_demand, model_version, features_json)
            VALUES (%s, %s, %s, %s)""",
            (datetime.utcnow(), prediction, "champion", json.dumps(request.model_dump())),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Warning: failed to log prediction: {e}")
```

### 5.10 Orchestration — `src/orchestration/flows.py`

```python
"""
Prefect flows for automated pipeline orchestration.
Three main flows:
1. data_ingestion_flow — fetch new data from EIA + weather APIs
2. drift_check_flow — run Evidently drift detection
3. retrain_flow — retrain model if drift detected or performance dropped
"""
from prefect import flow, task
from prefect.logging import get_run_logger
from datetime import datetime, timedelta
import pandas as pd

from src.data.eia_client import EIAClient
from src.data.weather_client import WeatherClient
from src.data.feature_engineering import build_features
from src.model.train import train_model
from src.model.registry import promote_to_champion
from src.monitoring.drift_detection import run_drift_check, generate_drift_html_report
from src.monitoring.prometheus_metrics import (
    DRIFT_SHARE, DRIFT_DETECTED, RETRAIN_COUNTER, DATA_FRESHNESS_HOURS,
)
from src.config import DATABASE_URL, REFERENCE_DIR


# ---- Tasks ----

@task(retries=3, retry_delay_seconds=60)
def fetch_eia_data(start: str, end: str) -> pd.DataFrame:
    client = EIAClient()
    return client.fetch_demand(start=start, end=end)


@task(retries=3, retry_delay_seconds=60)
def fetch_weather_data(start_date: str, end_date: str) -> pd.DataFrame:
    client = WeatherClient()
    return client.fetch_historical(start_date=start_date, end_date=end_date)


@task
def engineer_features(demand_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    return build_features(demand_df, weather_df)


@task
def save_to_db(df: pd.DataFrame, table_name: str):
    """Save DataFrame to PostgreSQL table."""
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL)
    df.to_sql(table_name, engine, if_exists="append", index=False)


@task
def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used for drift comparison."""
    return pd.read_parquet(REFERENCE_DIR / "reference_features.parquet")


@task
def load_recent_data(days: int = 7) -> pd.DataFrame:
    """Load the most recent N days of feature data from DB."""
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL)
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    query = f"SELECT * FROM features WHERE timestamp >= '{cutoff}' ORDER BY timestamp"
    return pd.read_sql(query, engine)


@task
def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    should_retrain, drift_share, details = run_drift_check(reference_df, current_df)
    # Update Prometheus gauges
    DRIFT_SHARE.set(drift_share)
    DRIFT_DETECTED.set(1.0 if should_retrain else 0.0)
    return should_retrain, drift_share, details


@task
def run_training(df: pd.DataFrame):
    model, metrics = train_model(df)
    return model, metrics


# ---- Flows ----

@flow(name="data-ingestion", log_prints=True)
def data_ingestion_flow(
    start: str = None,
    end: str = None,
    days_back: int = 7,
):
    """
    Ingest new demand + weather data, engineer features, save to DB.
    By default fetches the last 7 days.
    """
    logger = get_run_logger()

    if start is None:
        start_dt = datetime.utcnow() - timedelta(days=days_back)
        start = start_dt.strftime("%Y-%m-%dT%H")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%dT%H")

    start_date = start[:10]  # "YYYY-MM-DD"
    end_date = end[:10]

    logger.info(f"Ingesting data from {start} to {end}")

    demand_df = fetch_eia_data(start=start, end=end)
    weather_df = fetch_weather_data(start_date=start_date, end_date=end_date)

    if demand_df.empty or weather_df.empty:
        logger.warning("No data returned from APIs. Skipping.")
        return

    features_df = engineer_features(demand_df, weather_df)
    save_to_db(features_df, "features")

    logger.info(f"Ingested {len(features_df)} feature rows")


@flow(name="drift-check", log_prints=True)
def drift_check_flow():
    """
    Run drift detection: compare recent 7 days against reference data.
    If drift exceeds threshold, trigger retrain_flow.
    """
    logger = get_run_logger()

    reference_df = load_reference_data()
    current_df = load_recent_data(days=7)

    if current_df.empty:
        logger.warning("No recent data for drift check. Skipping.")
        return

    should_retrain, drift_share, details = check_drift(reference_df, current_df)

    logger.info(f"Drift share: {drift_share:.2%}, Retrain needed: {should_retrain}")

    # Save drift report
    generate_drift_html_report(reference_df, current_df, "reports/latest_drift.html")

    # Log to drift_log table
    save_to_db(
        pd.DataFrame([{
            "dataset_drift_detected": details["dataset_drift"],
            "drift_share": drift_share,
            "n_drifted_features": sum(
                1 for v in details["per_feature"].values() if v["drifted"]
            ),
            "n_total_features": len(details["per_feature"]),
            "drift_details": str(details),
            "triggered_retrain": should_retrain,
        }]),
        "drift_log",
    )

    if should_retrain:
        logger.info("Drift threshold exceeded — triggering retrain flow")
        RETRAIN_COUNTER.inc()
        retrain_flow()


@flow(name="retrain", log_prints=True)
def retrain_flow():
    """
    Retrain the model on all available feature data.
    Compare against champion — promote if better.
    """
    logger = get_run_logger()

    # Load all feature data
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM features ORDER BY timestamp", engine)

    if len(df) < 1000:
        logger.warning("Not enough data to retrain. Skipping.")
        return

    logger.info(f"Retraining on {len(df)} rows")
    model, metrics = run_training(df)

    # Get the run_id from the most recent MLflow run
    import mlflow
    runs = mlflow.search_runs(
        experiment_names=["energy-demand-forecast"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs.empty:
        run_id = runs.iloc[0]["run_id"]
        promoted = promote_to_champion(run_id, metrics["mae"])
        logger.info(f"Champion promotion: {promoted}")

    # Update reference data with latest training data
    reference_df = df[:-168*4]  # exclude test portion
    reference_df.to_parquet(REFERENCE_DIR / "reference_features.parquet", index=False)
    logger.info("Reference dataset updated")
```

---

## 6. INFRASTRUCTURE — Docker Compose & Config Files

### 6.1 `infra/docker-compose.yml`

```yaml
version: "3.8"

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: mlops
      POSTGRES_PASSWORD: mlops
      POSTGRES_DB: energy_mlops
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlops"]
      interval: 5s
      timeout: 5s
      retries: 5

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.18.0
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://mlops:mlops@postgres:5432/energy_mlops
      --default-artifact-root /mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      postgres:
        condition: service_healthy

  prefect-server:
    image: prefecthq/prefect:3-latest
    command: prefect server start --host 0.0.0.0
    ports:
      - "4200:4200"
    environment:
      PREFECT_API_DATABASE_CONNECTION_URL: postgresql+asyncpg://mlops:mlops@postgres:5432/energy_mlops

  prefect-worker:
    build:
      context: ..
      dockerfile: infra/Dockerfile.worker
    environment:
      PREFECT_API_URL: http://prefect-server:4200/api
      EIA_API_KEY: ${EIA_API_KEY}
      DATABASE_URL: postgresql://mlops:mlops@postgres:5432/energy_mlops
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - prefect-server
      - mlflow
      - postgres

  api:
    build:
      context: ..
      dockerfile: infra/Dockerfile.api
    ports:
      - "8000:8000"
      - "8001:8001"    # Prometheus metrics
    environment:
      DATABASE_URL: postgresql://mlops:mlops@postgres:5432/energy_mlops
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - mlflow
      - postgres

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana-oss:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_SECURITY_ADMIN_USER: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus

  streamlit:
    build:
      context: ..
      dockerfile: infra/Dockerfile.api   # reuse same image, different CMD
    ports:
      - "8501:8501"
    environment:
      DATABASE_URL: postgresql://mlops:mlops@postgres:5432/energy_mlops
      MLFLOW_TRACKING_URI: http://mlflow:5000
    command: ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
    depends_on:
      - mlflow
      - postgres

volumes:
  postgres_data:
  mlflow_artifacts:
  prometheus_data:
  grafana_data:
```

### 6.2 `infra/Dockerfile.api`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

EXPOSE 8000 8001

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.3 `infra/Dockerfile.worker`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

CMD ["prefect", "worker", "start", "--pool", "default"]
```

### 6.4 `infra/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "energy-demand-api"
    static_configs:
      - targets: ["api:8001"]
    metrics_path: /metrics

  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
```

### 6.5 `infra/grafana/provisioning/datasources/datasources.yml`

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: energy_mlops
    user: mlops
    secureJsonData:
      password: mlops
    jsonData:
      sslmode: disable
    editable: true
```

---

## 7. REQUIREMENTS

### 7.1 `requirements.txt`

```
# Core
python-dotenv>=1.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3

# ML
lightgbm>=4.0

# Experiment tracking
mlflow>=2.15

# Drift detection
evidently>=0.6

# API serving
fastapi>=0.110
uvicorn>=0.27
pydantic>=2.0

# Database
psycopg2-binary>=2.9
sqlalchemy>=2.0

# Monitoring
prometheus-client>=0.20

# Dashboard
streamlit>=1.35
plotly>=5.20

# Orchestration
prefect>=3.0

# Data versioning
dvc>=3.0

# Utilities
requests>=2.31
holidays>=0.40

# Testing
pytest>=8.0
httpx>=0.27     # for testing FastAPI
```

---

## 8. GRAFANA DASHBOARD PANELS

Create a Grafana dashboard (`energy-demand.json`) with these panels:

### Row 1: Model Health Overview
1. **Stat Panel**: Current MAE (from `model_mae_mwh` gauge)
2. **Stat Panel**: Current MAPE (from `model_mape_percent` gauge)
3. **Stat Panel**: Drift Share % (from `drift_share` gauge) — red if >0.3
4. **Stat Panel**: Data Freshness Hours (from `data_freshness_hours` gauge)

### Row 2: Predictions
5. **Time Series**: Prediction count rate (`rate(prediction_requests_total[5m])`)
6. **Histogram**: Prediction latency distribution (`prediction_latency_seconds`)
7. **Time Series**: Predicted demand distribution over time (`predicted_demand_mwh`)

### Row 3: Drift Monitoring
8. **Time Series**: Drift share over time (query PostgreSQL `drift_log` table)
9. **Table**: Per-feature drift scores (query `drift_log.drift_details`)
10. **Stat Panel**: Total retrain count (`retrain_triggered_total`)

### Row 4: Data Pipeline
11. **Time Series**: Actual vs Predicted demand (query `prediction_log`, overlay `actual_demand` and `predicted_demand`)
12. **Stat Panel**: Total data points ingested (query `SELECT count(*) FROM features`)

### Alert Rules (configured in Grafana):
- **Drift alert**: Fire when `drift_share > 0.3` for >5 minutes
- **Performance alert**: Fire when `model_mae_mwh > 1500` for >1 hour
- **Freshness alert**: Fire when `data_freshness_hours > 48`

---

## 8.5 STREAMLIT ML DASHBOARD (PRIMARY ML VISUALIZATION)

Grafana handles Prometheus infrastructure metrics. The **Streamlit app is the primary ML dashboard** for model performance, drift analysis, and experiment history.

### `dashboard/app.py` — Main Entry

```python
"""
Streamlit ML Dashboard — primary interface for model observability.
Run with: streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Energy Demand MLOps Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Demand Forecasting — MLOps Dashboard")
st.markdown("""
Monitor model performance, data drift, training history, and feature importance.
Use the sidebar to navigate between pages.
""")

# Overview metrics (pulled from DB)
import psycopg2
import pandas as pd
from src.config import DATABASE_URL

@st.cache_data(ttl=60)
def get_overview_stats():
    conn = psycopg2.connect(DATABASE_URL)
    stats = {}
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM features")
        stats["total_features"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM prediction_log")
        stats["total_predictions"] = cur.fetchone()[0]
        cur.execute("SELECT MAX(timestamp) FROM features")
        stats["latest_data"] = cur.fetchone()[0]
        cur.execute("""
            SELECT drift_share, check_timestamp
            FROM drift_log ORDER BY check_timestamp DESC LIMIT 1
        """)
        row = cur.fetchone()
        stats["latest_drift_share"] = row[0] if row else 0.0
        stats["latest_drift_check"] = row[1] if row else None
    conn.close()
    return stats

stats = get_overview_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data Points", f"{stats['total_features']:,}")
col2.metric("Total Predictions", f"{stats['total_predictions']:,}")
col3.metric("Latest Data", str(stats["latest_data"])[:16] if stats["latest_data"] else "N/A")
col4.metric("Drift Share", f"{stats['latest_drift_share']:.1%}",
            delta_color="inverse")
```

### `dashboard/pages/1_📊_Model_Performance.py`

```python
"""
Page 1: Actual vs Predicted demand over time, MAE trend, error distribution.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from src.config import DATABASE_URL

st.header("📊 Model Performance")

engine = create_engine(DATABASE_URL)

# Load prediction log where actuals are available
@st.cache_data(ttl=120)
def load_predictions():
    query = """
        SELECT timestamp, predicted_demand, actual_demand, model_version, prediction_made_at
        FROM prediction_log
        WHERE actual_demand IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 5000
    """
    return pd.read_sql(query, engine)

df = load_predictions()

if df.empty:
    st.info("No predictions with actuals yet. Run the pipeline to generate data.")
    st.stop()

# Actual vs Predicted time series
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["actual_demand"],
                         mode="lines", name="Actual", line=dict(color="#2196F3")))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["predicted_demand"],
                         mode="lines", name="Predicted", line=dict(color="#FF9800", dash="dot")))
fig.update_layout(title="Actual vs Predicted Demand (MWh)", xaxis_title="Time", yaxis_title="MWh")
st.plotly_chart(fig, use_container_width=True)

# Error distribution
df["error"] = df["actual_demand"] - df["predicted_demand"]
df["abs_error"] = df["error"].abs()

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{df['abs_error'].mean():,.0f} MWh")
col2.metric("RMSE", f"{(df['error']**2).mean()**0.5:,.0f} MWh")
col3.metric("MAPE", f"{(df['abs_error'] / df['actual_demand']).mean() * 100:.1f}%")

fig_err = px.histogram(df, x="error", nbins=50, title="Prediction Error Distribution",
                       labels={"error": "Error (MWh)"})
st.plotly_chart(fig_err, use_container_width=True)

# Rolling MAE over time
df_sorted = df.sort_values("timestamp")
df_sorted["rolling_mae"] = df_sorted["abs_error"].rolling(168).mean()  # 1 week
fig_mae = px.line(df_sorted, x="timestamp", y="rolling_mae",
                  title="Rolling MAE (7-day window)")
st.plotly_chart(fig_mae, use_container_width=True)
```

### `dashboard/pages/2_🔄_Drift_Monitor.py`

```python
"""
Page 2: Drift detection history, per-feature drift scores, Evidently HTML reports.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from sqlalchemy import create_engine
from pathlib import Path
from src.config import DATABASE_URL

st.header("🔄 Drift Monitor")

engine = create_engine(DATABASE_URL)

@st.cache_data(ttl=120)
def load_drift_history():
    return pd.read_sql(
        "SELECT * FROM drift_log ORDER BY check_timestamp DESC LIMIT 100",
        engine,
    )

drift_df = load_drift_history()

if drift_df.empty:
    st.info("No drift checks recorded yet. Run the drift check flow.")
    st.stop()

# Drift share over time
fig = px.line(drift_df.sort_values("check_timestamp"),
              x="check_timestamp", y="drift_share",
              title="Drift Share Over Time")
fig.add_hline(y=0.3, line_dash="dash", line_color="red",
              annotation_text="Retrain Threshold (30%)")
st.plotly_chart(fig, use_container_width=True)

# Latest drift details
st.subheader("Latest Drift Report")
latest = drift_df.iloc[0]
col1, col2, col3 = st.columns(3)
col1.metric("Dataset Drift", "YES" if latest["dataset_drift_detected"] else "No")
col2.metric("Drift Share", f"{latest['drift_share']:.1%}")
col3.metric("Drifted Features", f"{latest['n_drifted_features']}/{latest['n_total_features']}")

# Per-feature drift table
try:
    details = json.loads(latest["drift_details"]) if isinstance(latest["drift_details"], str) else latest["drift_details"]
    if "per_feature" in details:
        feat_df = pd.DataFrame([
            {"Feature": k, "Drifted": v["drifted"], "Score": v["score"]}
            for k, v in details["per_feature"].items()
        ]).sort_values("Score", ascending=False)
        st.dataframe(feat_df, use_container_width=True)
except Exception:
    st.write("Could not parse drift details.")

# Show Evidently HTML report if available
report_path = Path("reports/latest_drift.html")
if report_path.exists():
    with st.expander("📄 Full Evidently Drift Report (HTML)"):
        st.components.v1.html(report_path.read_text(), height=800, scrolling=True)
```

### `dashboard/pages/3_🏋️_Training_History.py`

```python
"""
Page 3: MLflow experiment tracking — compare runs, view metrics trends.
"""
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

st.header("🏋️ Training History")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@st.cache_data(ttl=120)
def load_runs():
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        order_by=["start_time DESC"],
        max_results=50,
    )
    return runs

runs = load_runs()

if runs.empty:
    st.info("No training runs yet. Train the model first.")
    st.stop()

# Metrics over time
cols_to_show = [c for c in runs.columns if c.startswith("metrics.")]
display_df = runs[["start_time", "run_id"] + cols_to_show].copy()
display_df.columns = [c.replace("metrics.", "") for c in display_df.columns]

fig = px.scatter(display_df, x="start_time", y="mae", size="rmse",
                 hover_data=["run_id", "r2"],
                 title="Training Runs: MAE Over Time")
st.plotly_chart(fig, use_container_width=True)

# Run comparison table
st.subheader("Run Comparison")
st.dataframe(display_df.head(20), use_container_width=True)

# Champion model info
st.subheader("Current Champion")
try:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    from src.config import MODEL_NAME
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    st.json({
        "Model": MODEL_NAME,
        "Version": champion.version,
        "Run ID": champion.run_id,
        "Created": str(champion.creation_timestamp),
    })
except Exception as e:
    st.warning(f"No champion model set yet: {e}")
```

### `dashboard/pages/4_📈_Feature_Importance.py`

```python
"""
Page 4: Feature importance from the champion model.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

st.header("📈 Feature Importance")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@champion")

    importance_df = pd.DataFrame({
        "Feature": model.feature_name_,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                 title="LightGBM Feature Importance (Split-based)")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Top 5 summary
    st.subheader("Top 5 Most Important Features")
    top5 = importance_df.nlargest(5, "Importance")
    for _, row in top5.iterrows():
        st.write(f"**{row['Feature']}**: {row['Importance']}")

except Exception as e:
    st.warning(f"Could not load champion model: {e}")
    st.info("Train a model and promote it to champion first.")
```

---

## 9. MAKEFILE — COMMON COMMANDS

```makefile
.PHONY: setup up down logs train drift ingest test lint

# Start all services
up:
	cd infra && docker-compose up -d

# Stop all services
down:
	cd infra && docker-compose down

# View logs
logs:
	cd infra && docker-compose logs -f

# Initial setup: create dirs, install deps, init DB
setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/reference reports
	dvc init

# Run data ingestion (full historical backfill)
backfill:
	python -c "from src.orchestration.flows import data_ingestion_flow; \
	data_ingestion_flow(start='2023-01-01T00', end='2025-12-31T23')"

# Run incremental ingestion (last 7 days)
ingest:
	python -c "from src.orchestration.flows import data_ingestion_flow; \
	data_ingestion_flow(days_back=7)"

# Train model
train:
	python -c "from src.orchestration.flows import retrain_flow; retrain_flow()"

# Run drift check
drift:
	python -c "from src.orchestration.flows import drift_check_flow; drift_check_flow()"

# Run all tests
test:
	pytest tests/ -v

# Lint
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Run Streamlit ML dashboard
dashboard:
	streamlit run dashboard/app.py --server.port 8501
```

---

## 10. STEP-BY-STEP IMPLEMENTATION ORDER

> **Strategy: MVP-first.** Get a working end-to-end loop before polishing. Each phase should produce something runnable. Do NOT write tests until Phase 2 is complete.

### Phase 1: Infrastructure + Data Foundation (Day 1-2)
1. Create repo structure as defined in Section 2
2. Write `docker-compose.yml` with Postgres, MLflow only (start lean — add Prometheus/Grafana later)
3. Write `init.sql` and verify DB schema: `docker-compose up postgres`
4. Set up `.env.example` and `.env` with placeholder `EIA_API_KEY`
5. Implement `src/config.py` — with graceful error if EIA_API_KEY missing
6. Implement `src/data/eia_client.py` — verify with a small date range (1 week)
7. Implement `src/data/weather_client.py` — verify with same date range
8. Implement `src/data/feature_engineering.py`
9. Run a historical backfill: fetch 2023-01-01 to present, save to Postgres + parquet
10. Save the training portion as reference dataset (`data/reference/reference_features.parquet`)

**Checkpoint**: You can run `python -c "from src.data.eia_client import EIAClient; print(EIAClient().fetch_demand('2024-12-01T00', '2024-12-07T23').shape)"` and get data back.

### Phase 2: Model Training + Registry (Day 3-4)
11. Start MLflow via docker-compose: `docker-compose up mlflow`
12. Implement `src/model/train.py` — train first model, verify it appears in MLflow UI at localhost:5000
13. Implement `src/model/registry.py` — champion/challenger promotion logic
14. Implement `src/model/evaluate.py`
15. Train the first model and promote it to champion
16. Experiment in notebooks if desired (optional)
17. **NOW write tests**: `test_feature_engineering.py`, `test_train.py`, `test_evaluate.py`

**Checkpoint**: `mlflow.lightgbm.load_model("models:/energy-demand-lgbm@champion")` loads successfully.

### Phase 3: Serving API (Day 4-5)
18. Implement `src/serving/schemas.py` (Pydantic models)
19. Implement `src/serving/app.py` — FastAPI with `/predict`, `/health`, `/reload-model`
20. Implement `src/monitoring/prometheus_metrics.py`
21. Run locally: `uvicorn src.serving.app:app --reload` and test with curl/httpx
22. Write `test_app.py` with httpx TestClient
23. Create `Dockerfile.api` (containerize later, not now)

**Checkpoint**: `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"temperature": 15.0, ...}'` returns a prediction.

### Phase 4: Drift Detection (Day 5-6)
24. Implement `src/monitoring/drift_detection.py` — Evidently drift reports
25. Test with synthetic drift: take reference data, shift temperature by +10°C, verify drift is detected
26. Implement `src/monitoring/performance_monitor.py` — backfill actuals and compute rolling metrics
27. Write `test_drift_detection.py`

**Checkpoint**: `run_drift_check(reference_df, shifted_df)` returns `(True, 0.6, {...})`.

### Phase 5: Orchestration (Day 6-7)
28. Implement `src/orchestration/flows.py` — all three Prefect flows
29. Test flows locally without Prefect server first (just call `data_ingestion_flow()` directly)
30. Add Prefect server to docker-compose
31. Create Prefect deployments with schedules:
    - `data_ingestion_flow`: every 6 hours
    - `drift_check_flow`: daily at midnight
    - `retrain_flow`: triggered by drift or manual

**Checkpoint**: Full loop works: ingest → drift check detects shift → retrain triggers → new champion promoted.

### Phase 6: Dashboards (Day 7-8)
32. Add Prometheus + Grafana to docker-compose
33. Configure Prometheus scraping (prometheus.yml)
34. Configure Grafana datasources provisioning
35. Build Grafana panels for infrastructure metrics (prediction latency, request count)
36. Implement Streamlit dashboard: `dashboard/app.py` + all 4 pages
37. Test full dashboard: `streamlit run dashboard/app.py`

**Checkpoint**: `make dashboard` shows actual vs predicted plots, drift history, feature importance.

### Phase 7: DVC + CI/CD (Day 8-9)
38. Initialize DVC: `dvc init`, track `data/raw/` and `data/reference/`
39. Write GitHub Actions CI workflow (`ci.yml`)
40. Write `README.md` with architecture diagram (Mermaid)

### Phase 8: Polish & End-to-End Test (Day 9-10)
41. Run full end-to-end: `docker-compose up` → backfill → train → serve → drift check → retrain
42. Add Grafana alert rules
43. Verify Streamlit dashboard populates correctly
44. Clean up code, add docstrings, finalize README
45. Final `make test && make lint`

---

## 11. KEY DESIGN DECISIONS & RATIONALE

### Why LightGBM (not deep learning)?
Gradient boosting is the right tool for tabular time-series forecasting with <100 features. It's fast to train, easy to interpret, and doesn't require GPU. The MLOps infrastructure is the star of this project, not the model.

### Why Prefect over Airflow?
Prefect 3.x has a simpler Python-native API (decorators, not DAG definitions), lighter resource footprint, and easier local development. Airflow is more enterprise but overkill for this scope.

### Why time-based split (not random)?
Energy demand is a time series. Random splits would leak future information into training data. Always split chronologically.

### Why store predictions in PostgreSQL?
To compute model performance metrics when ground truth arrives (demand actuals come with ~1-2 day lag from EIA). The prediction log enables retroactive evaluation.

### Champion/Challenger pattern
Never auto-deploy a worse model. Every retrain compares against the current champion on the holdout set. Only promote if MAE improves.

### Reference dataset updates
After each successful retrain, the training data (minus the test set) becomes the new reference dataset for drift detection. This prevents "reference staleness" where old baselines trigger false drift alarms.

### Why Streamlit + Grafana (not just Grafana)?
Grafana excels at time-series infrastructure metrics from Prometheus (request latency, throughput). But for ML-specific visualizations — actual vs predicted plots, Evidently HTML drift reports, MLflow experiment comparison, feature importance charts — Streamlit is far easier to iterate on. It's Python-native, requires no JSON provisioning, and can query both PostgreSQL and MLflow directly. The split: Grafana for infra metrics + alerts, Streamlit for ML observability.

---

## 12. ENVIRONMENT VARIABLES

```env
# .env.example — copy to .env and fill in values

# EIA API (required — get from https://www.eia.gov/opendata/)
EIA_API_KEY=your_eia_api_key_here
EIA_REGION=NY

# Weather location (NYC for NY grid region)
WEATHER_LAT=40.7128
WEATHER_LON=-74.0060
WEATHER_TZ=America/New_York

# PostgreSQL
DATABASE_URL=postgresql://mlops:mlops@localhost:5432/energy_mlops

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_HOST=0.0.0.0
API_PORT=8000
PROMETHEUS_PORT=8001
```

---

## 13. TESTING STRATEGY

### Unit Tests
- `test_eia_client.py`: Mock API responses, test pagination, error handling
- `test_weather_client.py`: Mock API responses, test date parsing
- `test_feature_engineering.py`: Test feature computation correctness with known inputs
- `test_drift_detection.py`: Test with identical data (no drift) and synthetically shifted data (drift detected)

### Integration Tests
- `test_app.py`: Use httpx TestClient to hit FastAPI endpoints
- Test full pipeline: ingest → feature engineering → train → predict

### Example test structure:

```python
# tests/test_data/test_feature_engineering.py

import pandas as pd
import numpy as np
from src.data.feature_engineering import build_features

def test_build_features_basic():
    """Test that feature engineering produces expected columns."""
    demand = pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=200, freq="h"),
        "demand_mwh": np.random.uniform(15000, 25000, 200),
    })
    weather = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
        "temperature_2m": np.random.uniform(-5, 35, 200),
        "relative_humidity_2m": np.random.uniform(30, 95, 200),
        "apparent_temperature": np.random.uniform(-10, 40, 200),
        "wind_speed_10m": np.random.uniform(0, 20, 200),
        "cloud_cover": np.random.uniform(0, 100, 200),
        "shortwave_radiation": np.random.uniform(0, 800, 200),
        "precipitation": np.random.uniform(0, 5, 200),
    })

    result = build_features(demand, weather)

    # Check all expected columns exist
    assert "demand_mwh" in result.columns
    assert "hour_of_day" in result.columns
    assert "demand_lag_24h" in result.columns
    assert "temp_squared" in result.columns
    assert "cooling_degree_hours" in result.columns

    # Check no NaN values (they should be dropped)
    assert result.isna().sum().sum() == 0

    # Check temp_squared is correct
    assert np.allclose(result["temp_squared"], result["temperature"] ** 2)
```

---

## 14. GITHUB ACTIONS CI WORKFLOW

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: mlops
          POSTGRES_PASSWORD: mlops
          POSTGRES_DB: energy_mlops
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff mypy

      - name: Lint
        run: ruff check src/ tests/

      - name: Type check
        run: mypy src/ --ignore-missing-imports

      - name: Run tests
        env:
          DATABASE_URL: postgresql://mlops:mlops@localhost:5432/energy_mlops
          EIA_API_KEY: test_key
        run: pytest tests/ -v --tb=short
```

---

## 15. README TEMPLATE

The README should include:
1. Project title + one-line description
2. Architecture diagram (use Mermaid or ASCII)
3. Quick start: `cp .env.example .env && make up`
4. Tech stack table
5. API documentation link (FastAPI auto-generates at `/docs`)
6. Screenshots: Grafana dashboard, MLflow experiment tracking, Evidently drift report
7. How continuous learning works (explain the three flows)
8. How to contribute

---

This document contains everything needed to build the complete project. Start with Phase 1 and work through each phase sequentially. Every code snippet is production-ready and can be directly used or adapted. The architecture is designed to run entirely on a single machine with `docker-compose up`.
