"""
Reusable tasks for Prefect flows.
Each task represents an atomic operation that can be composed into workflows.
"""
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from prefect import task
from prefect.logging import get_run_logger

from src.data.eia_client import EIAClient
from src.data.weather_client import WeatherClient
from src.data.feature_engineering import build_features
from src.model.train import train_model
from src.model.registry import get_champion_model, promote_to_champion
from src.model.predict import predict_single, predict_batch
from src.monitoring.drift_detection import run_drift_check, load_reference_data, generate_drift_html_report
from src.monitoring.performance_monitor import calculate_recent_performance
from src.monitoring.prometheus_metrics import DRIFT_SHARE, DRIFT_DETECTED, RETRAIN_COUNTER
from src.config import PROCESSED_DIR, RAW_DIR


@task(name="fetch-demand-data", retries=2, retry_delay_seconds=60)
def fetch_demand_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch electricity demand data from EIA API."""
    logger = get_run_logger()
    logger.info(f"Fetching demand data from {start_date} to {end_date}")
    
    client = EIAClient()
    df = client.fetch_demand(start_date, end_date)
    
    logger.info(f"Fetched {len(df)} demand records")
    return df


@task(name="fetch-weather-data", retries=2, retry_delay_seconds=60)
def fetch_weather_data(start_date: str, end_date: str, forecast: bool = False) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo API."""
    logger = get_run_logger()
    logger.info(f"Fetching {'forecast' if forecast else 'historical'} weather data")
    
    client = WeatherClient()
    if forecast:
        df = client.fetch_forecast()
    else:
        df = client.fetch_historical(start_date, end_date)
    
    logger.info(f"Fetched {len(df)} weather records")
    return df


@task(name="build-features")
def build_features_task(demand_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from raw data."""
    logger = get_run_logger()
    logger.info("Building features")
    
    features_df = build_features(demand_df, weather_df)
    
    logger.info(f"Built {len(features_df)} feature rows with {len(features_df.columns)} columns")
    return features_df


@task(name="save-data")
def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to parquet file."""
    logger = get_run_logger()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df)} rows to {output_path}")


@task(name="load-data")
def load_data(input_path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    logger = get_run_logger()
    
    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    return df


@task(name="check-drift")
def check_drift_task(current_df: pd.DataFrame, reference_df: pd.DataFrame) -> dict:
    """Run drift detection and return results."""
    from src.monitoring.drift_detection import save_drift_results
    logger = get_run_logger()
    logger.info(f"Checking drift: {len(current_df)} current vs {len(reference_df)} reference samples")
    
    should_retrain, drift_share, details = run_drift_check(reference_df, current_df)
    
    # Save drift results to JSON for dashboard
    save_drift_results(details, "reports/drift_results.json")
    
    # Update Prometheus metrics
    DRIFT_SHARE.set(drift_share)
    DRIFT_DETECTED.set(1.0 if should_retrain else 0.0)
    
    logger.info(f"Drift check complete: drift_share={drift_share:.2%}, should_retrain={should_retrain}")
    
    return {
        "should_retrain": should_retrain,
        "drift_share": drift_share,
        "details": details
    }


@task(name="generate-drift-report")
def generate_drift_report_task(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str) -> str:
    """Generate HTML drift report."""
    logger = get_run_logger()
    
    report_path = generate_drift_html_report(reference_df, current_df, output_path)
    logger.info(f"Generated drift report: {report_path}")
    
    return report_path


@task(name="train-model")
def train_model_task(features_df: pd.DataFrame) -> dict:
    """Train a new model and return metrics."""
    logger = get_run_logger()
    logger.info(f"Training model on {len(features_df)} samples")
    
    model, metrics, run_id = train_model(features_df)
    
    logger.info(f"Training complete: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    logger.info(f"MLflow run_id: {run_id}")
    
    return {
        "run_id": run_id,
        "metrics": metrics
    }


@task(name="evaluate-and-promote")
def evaluate_and_promote_task(run_id: str) -> bool:
    """Evaluate new model and promote if better than champion."""
    logger = get_run_logger()
    logger.info(f"Evaluating model {run_id} for promotion")
    
    promoted = promote_to_champion(run_id)
    
    if promoted:
        logger.info(f"✓ Model {run_id} promoted to champion")
        RETRAIN_COUNTER.labels(reason="drift").inc()
    else:
        logger.warning(f"✗ Model {run_id} not promoted (worse than champion)")
    
    return promoted


@task(name="generate-predictions")
def generate_predictions_task(features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using champion model."""
    from src.config import PROCESSED_DIR
    logger = get_run_logger()
    logger.info(f"Generating predictions for {len(features_df)} samples")
    
    model = get_champion_model()
    predictions = predict_batch(model, features_df)
    
    # Add predictions to dataframe
    result_df = features_df.copy()
    result_df['prediction'] = predictions
    result_df['prediction_timestamp'] = datetime.utcnow()
    
    # Save predictions back to features file for dashboard
    output_path = PROCESSED_DIR / "features.parquet"
    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    return result_df


@task(name="calculate-performance")
def calculate_performance_task(predictions_df: pd.DataFrame) -> dict:
    """Calculate model performance metrics if actuals are available."""
    logger = get_run_logger()
    
    if 'demand_mwh' not in predictions_df.columns or 'prediction' not in predictions_df.columns:
        logger.warning("Cannot calculate performance: missing demand_mwh or prediction columns")
        return {}
    
    # Rename columns to match performance_monitor expectations
    performance_df = predictions_df.copy()
    performance_df['actual_demand'] = performance_df['demand_mwh']
    performance_df['predicted_demand'] = performance_df['prediction']
    
    # Filter to rows with both actual and predicted values
    valid_df = performance_df.dropna(subset=['actual_demand', 'predicted_demand'])
    
    if len(valid_df) == 0:
        logger.warning("No valid prediction-actual pairs found")
        return {}
    
    performance = calculate_recent_performance(
        predictions_df=valid_df,
        days_back=7
    )
    
    if performance:
        logger.info(f"Performance: MAE={performance.get('mae', 0):.2f}, RMSE={performance.get('rmse', 0):.2f}")
    
    return performance or {}
