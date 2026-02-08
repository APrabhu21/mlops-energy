"""
Custom Prometheus metrics for model observability.
These are scraped by Prometheus and visualized in Grafana.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
from functools import wraps
import time

# --- Prediction metrics ---
PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status"],  # success or error
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_VALUE = Histogram(
    "predicted_demand_mwh",
    "Distribution of predicted demand values",
    buckets=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000],
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

MODEL_RMSE = Gauge(
    "model_rmse_mwh",
    "Current model RMSE on recent actuals",
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
    ["reason"],  # drift, performance, manual
)

# --- API health ---
API_UP = Gauge(
    "api_up",
    "API service availability (1 = up, 0 = down)",
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Model loaded status (1 = loaded, 0 = not loaded)",
)

# --- Request errors ---
REQUEST_ERRORS = Counter(
    "request_errors_total",
    "Total number of request errors",
    ["error_type"],
)


def track_prediction_time(func):
    """Decorator to track prediction latency."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            PREDICTION_COUNTER.labels(status="success").inc()
            return result
        except Exception as e:
            PREDICTION_COUNTER.labels(status="error").inc()
            REQUEST_ERRORS.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            latency = time.time() - start_time
            PREDICTION_LATENCY.observe(latency)
    return wrapper


def start_metrics_server(port: int = 8001):
    """
    Start the Prometheus metrics HTTP server.
    This exposes metrics at http://localhost:8001/metrics
    """
    try:
        start_http_server(port)
        API_UP.set(1)
        print(f"✓ Prometheus metrics server started on port {port}")
        print(f"  Metrics available at http://localhost:{port}/metrics")
    except OSError as e:
        if "already in use" in str(e).lower():
            print(f"⚠️  Port {port} already in use, metrics server may already be running")
        else:
            raise


def shutdown_metrics_server():
    """Clean shutdown of metrics server."""
    API_UP.set(0)
    MODEL_LOADED.set(0)
