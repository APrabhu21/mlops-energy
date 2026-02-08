"""
Prefect flows for MLOps pipeline orchestration.
Implements automated data ingestion, drift detection, and model retraining.
"""
from datetime import datetime, timedelta
from pathlib import Path
from prefect import flow
from prefect.logging import get_run_logger

from src.orchestration.tasks import (
    fetch_demand_data,
    fetch_weather_data,
    build_features_task,
    save_data,
    load_data,
    check_drift_task,
    generate_drift_report_task,
    train_model_task,
    evaluate_and_promote_task,
    generate_predictions_task,
    calculate_performance_task,
)
from src.config import RAW_DIR, PROCESSED_DIR, REFERENCE_DIR


@flow(name="data-ingestion", log_prints=True)
def data_ingestion_flow(days_back: int = 7, save_to_disk: bool = True):
    """
    Fetch and process new data from APIs.
    
    Args:
        days_back: Number of days of historical data to fetch
        save_to_disk: Whether to save processed data to disk
        
    Returns:
        DataFrame with processed features
    """
    logger = get_run_logger()
    logger.info(f"Starting data ingestion flow for last {days_back} days")
    
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Fetch data from APIs
    demand_df = fetch_demand_data(start_str, end_str)
    weather_df = fetch_weather_data(start_str, end_str, forecast=False)
    
    # Build features
    features_df = build_features_task(demand_df, weather_df)
    
    # Save to disk if requested
    if save_to_disk:
        save_data(demand_df, RAW_DIR / "demand_data.parquet")
        save_data(weather_df, RAW_DIR / "weather_data.parquet")
        save_data(features_df, PROCESSED_DIR / "features.parquet")
    
    logger.info(f"✓ Data ingestion complete: {len(features_df)} feature rows")
    
    return features_df


@flow(name="drift-check", log_prints=True)
def drift_check_flow(
    days_back: int = 7,
    generate_report: bool = True,
    trigger_retrain: bool = True
):
    """
    Check for data drift and optionally trigger retraining.
    
    Args:
        days_back: Number of days of recent data to check for drift
        generate_report: Whether to generate HTML drift report
        trigger_retrain: Whether to automatically trigger retraining if drift detected
        
    Returns:
        Dictionary with drift check results
    """
    logger = get_run_logger()
    logger.info("Starting drift check flow")
    
    # Load reference dataset
    reference_path = REFERENCE_DIR / "reference_features.parquet"
    if not reference_path.exists():
        logger.error(f"Reference dataset not found at {reference_path}. Run training first.")
        return {"error": "No reference dataset"}
    
    reference_df = load_data(reference_path)
    
    # Load recent data
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        logger.error(f"No feature data found at {features_path}. Run data ingestion first.")
        return {"error": "No feature data"}
    
    all_features = load_data(features_path)
    
    # Get recent data
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    current_df = all_features[all_features['timestamp'] >= cutoff].copy()
    
    if len(current_df) == 0:
        logger.warning(f"No data found in the last {days_back} days")
        return {"error": "No recent data"}
    
    logger.info(f"Comparing {len(current_df)} current samples vs {len(reference_df)} reference")
    
    # Check for drift
    drift_result = check_drift_task(current_df, reference_df)
    
    # Generate HTML report
    if generate_report:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/drift_report_{timestamp}.html"
        generate_drift_report_task(reference_df, current_df, report_path)
    
    # Trigger retraining if drift detected
    if trigger_retrain and drift_result["should_retrain"]:
        logger.warning(f"⚠️  Drift detected ({drift_result['drift_share']:.1%})! Triggering retrain flow...")
        retrain_result = retrain_flow(use_cached_data=True)
        drift_result["retrain_result"] = retrain_result
    
    logger.info(f"✓ Drift check complete: drift_share={drift_result['drift_share']:.2%}")
    
    return drift_result


@flow(name="retrain", log_prints=True)
def retrain_flow(
    days_back: int = 60,
    use_cached_data: bool = False,
    auto_promote: bool = True
):
    """
    Retrain model on recent data and optionally promote to champion.
    
    Args:
        days_back: Number of days of data to use for training
        use_cached_data: Whether to use cached data or fetch fresh data
        auto_promote: Whether to automatically promote if better than champion
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info(f"Starting retrain flow with {days_back} days of data")
    
    # Get training data
    if use_cached_data:
        logger.info("Using cached feature data")
        features_path = PROCESSED_DIR / "features.parquet"
        
        if not features_path.exists():
            logger.error("No cached data found. Running data ingestion first...")
            data_ingestion_flow(days_back=days_back, save_to_disk=True)
        
        features_df = load_data(features_path)
    else:
        logger.info("Fetching fresh data")
        features_df = data_ingestion_flow(days_back=days_back, save_to_disk=True)
    
    # Train new model
    train_result = train_model_task(features_df)
    run_id = train_result["run_id"]
    metrics = train_result["metrics"]
    
    # Evaluate and promote
    promoted = False
    if auto_promote:
        promoted = evaluate_and_promote_task(run_id)
    
    result = {
        "run_id": run_id,
        "metrics": metrics,
        "promoted": promoted,
        "training_samples": len(features_df)
    }
    
    logger.info(f"✓ Retrain complete: run_id={run_id}, promoted={promoted}")
    
    return result


@flow(name="prediction", log_prints=True)
def prediction_flow(
    forecast_hours: int = 24,
    save_predictions: bool = True
):
    """
    Generate predictions for future time periods.
    
    Args:
        forecast_hours: Number of hours to forecast ahead
        save_predictions: Whether to save predictions to disk
        
    Returns:
        DataFrame with predictions
    """
    logger = get_run_logger()
    logger.info(f"Starting prediction flow for {forecast_hours} hours ahead")
    
    # Fetch forecast weather data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(hours=168)  # Need history for lag features
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Get historical demand for lag features
    demand_df = fetch_demand_data(start_str, end_str)
    
    # Get forecast weather
    weather_forecast_df = fetch_weather_data(start_str, end_str, forecast=True)
    
    # Build features (will use latest demand for lags)
    features_df = build_features_task(demand_df, weather_forecast_df)
    
    # Keep only future timestamps
    future_df = features_df[features_df['timestamp'] > datetime.utcnow()].head(forecast_hours)
    
    if len(future_df) == 0:
        logger.warning("No future data available for prediction")
        return None
    
    # Generate predictions
    predictions_df = generate_predictions_task(future_df)
    
    # Save predictions
    if save_predictions:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = PROCESSED_DIR / f"predictions_{timestamp}.parquet"
        save_data(predictions_df, output_path)
    
    logger.info(f"✓ Generated {len(predictions_df)} predictions")
    
    return predictions_df


@flow(name="performance-monitoring", log_prints=True)
def performance_monitoring_flow(days_back: int = 7):
    """
    Monitor model performance on recent predictions vs actuals.
    
    Args:
        days_back: Number of days to evaluate
        
    Returns:
        Dictionary with performance metrics
    """
    logger = get_run_logger()
    logger.info(f"Starting performance monitoring for last {days_back} days")
    
    # Load features with actuals
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        logger.error("No feature data found")
        return {"error": "No data"}
    
    features_df = load_data(features_path)
    
    # Filter to recent data
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    recent_df = features_df[features_df['timestamp'] >= cutoff].copy()
    
    if len(recent_df) == 0:
        logger.warning("No recent data found")
        return {"error": "No recent data"}
    
    # Generate predictions
    predictions_df = generate_predictions_task(recent_df)
    
    # Calculate performance
    performance = calculate_performance_task(predictions_df)
    
    logger.info(f"✓ Performance monitoring complete")
    
    return performance


@flow(name="end-to-end-pipeline", log_prints=True)
def end_to_end_pipeline(
    days_back: int = 60,
    check_drift: bool = True,
    retrain_on_drift: bool = True
):
    """
    Complete MLOps pipeline: ingest data, check drift, retrain if needed, generate predictions.
    
    Args:
        days_back: Number of days of data to process
        check_drift: Whether to run drift detection
        retrain_on_drift: Whether to retrain if drift detected
        
    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    logger.info("=" * 70)
    logger.info("STARTING END-TO-END MLOPS PIPELINE")
    logger.info("=" * 70)
    
    results = {}
    
    # 1. Data Ingestion
    logger.info("\n[1/4] Data Ingestion...")
    features_df = data_ingestion_flow(days_back=days_back, save_to_disk=True)
    results["data_ingestion"] = {"samples": len(features_df)}
    
    # 2. Drift Check
    if check_drift:
        logger.info("\n[2/4] Drift Detection...")
        drift_result = drift_check_flow(
            days_back=7,
            generate_report=True,
            trigger_retrain=retrain_on_drift
        )
        results["drift_check"] = drift_result
    
    # 3. Generate Predictions
    logger.info("\n[3/4] Generating Predictions...")
    predictions_df = prediction_flow(forecast_hours=24, save_predictions=True)
    results["predictions"] = {"count": len(predictions_df) if predictions_df is not None else 0}
    
    # 4. Performance Monitoring
    logger.info("\n[4/4] Performance Monitoring...")
    performance = performance_monitoring_flow(days_back=7)
    results["performance"] = performance
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ END-TO-END PIPELINE COMPLETE")
    logger.info("=" * 70)
    
    return results
