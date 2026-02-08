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
EIA_API_KEY = os.getenv("EIA_API_KEY")
# Note: EIA_API_KEY is only required for data ingestion, not for serving predictions
# The eia_client.py will raise a clear error if it's needed but not set

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
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://dagshub.com/atharvaprabhu6/mlops-energy.mlflow"
)
MLFLOW_EXPERIMENT_NAME = "energy-demand-forecast"

# DagsHub credentials (for MLflow authentication)
DAGSHUB_USER = os.getenv("DAGSHUB_USER", "atharvaprabhu6")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

# Set MLflow credentials as environment variables if token is provided
if DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

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
