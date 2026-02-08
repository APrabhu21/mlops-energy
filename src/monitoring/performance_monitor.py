"""
Performance monitoring module.
Tracks model prediction quality over time by comparing predictions to actuals.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

from src.model.evaluate import evaluate_predictions
from src.monitoring.prometheus_metrics import MODEL_MAE, MODEL_MAPE, MODEL_RMSE


def calculate_recent_performance(
    predictions_df: pd.DataFrame,
    days_back: int = 7,
) -> Optional[Dict[str, float]]:
    """
    Calculate model performance on recent predictions where actuals are available.
    
    Args:
        predictions_df: DataFrame with columns: timestamp, predicted_demand, actual_demand
        days_back: Number of days to look back
        
    Returns:
        Dictionary of performance metrics or None if insufficient data
    """
    # Filter to recent period
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    recent = predictions_df[predictions_df['timestamp'] >= cutoff].copy()
    
    # Filter to only rows where we have actuals
    with_actuals = recent.dropna(subset=['actual_demand'])
    
    if len(with_actuals) < 10:
        print(f"⚠️  Insufficient data: only {len(with_actuals)} predictions with actuals")
        return None
    
    # Calculate metrics
    y_true = with_actuals['actual_demand'].values
    y_pred = with_actuals['predicted_demand'].values
    
    metrics = evaluate_predictions(y_true, y_pred)
    
    return metrics


def update_prometheus_metrics(metrics: Dict[str, float]) -> None:
    """Update Prometheus gauges with current performance metrics."""
    MODEL_MAE.set(metrics['mae'])
    MODEL_MAPE.set(metrics['mape'])
    MODEL_RMSE.set(metrics['rmse'])
    
    print("✓ Updated Prometheus metrics")


def check_performance_threshold(
    metrics: Dict[str, float],
    mae_threshold: float = 1500.0,
) -> bool:
    """
    Check if model performance has degraded below acceptable threshold.
    
    Args:
        metrics: Performance metrics dictionary
        mae_threshold: Maximum acceptable MAE in MWh
        
    Returns:
        True if retraining is recommended
    """
    if metrics['mae'] > mae_threshold:
        print(f"⚠️  Performance degradation detected: MAE {metrics['mae']:.1f} > {mae_threshold}")
        return True
    
    return False


def generate_performance_report(
    predictions_df: pd.DataFrame,
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Generate a performance report showing metrics over time.
    
    Args:
        predictions_df: Full predictions history
        days_back: Number of days to include in report
        
    Returns:
        DataFrame with daily performance metrics
    """
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    recent = predictions_df[predictions_df['timestamp'] >= cutoff].copy()
    
    # Group by day
    recent['date'] = pd.to_datetime(recent['timestamp']).dt.date
    
    daily_metrics = []
    
    for date, group in recent.groupby('date'):
        with_actuals = group.dropna(subset=['actual_demand'])
        
        if len(with_actuals) < 5:
            continue
        
        y_true = with_actuals['actual_demand'].values
        y_pred = with_actuals['predicted_demand'].values
        
        metrics = evaluate_predictions(y_true, y_pred)
        metrics['date'] = date
        metrics['n_predictions'] = len(with_actuals)
        
        daily_metrics.append(metrics)
    
    if not daily_metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(daily_metrics)


def print_performance_summary(metrics: Dict[str, float], days_back: int = 7) -> None:
    """Print formatted performance summary."""
    print("\n" + "=" * 70)
    print(f"MODEL PERFORMANCE (Last {days_back} days)")
    print("=" * 70)
    print(f"MAE:   {metrics['mae']:>10.2f} MWh")
    print(f"RMSE:  {metrics['rmse']:>10.2f} MWh")
    print(f"MAPE:  {metrics['mape']:>10.2f} %")
    print(f"R²:    {metrics['r2']:>10.4f}")
    print("=" * 70 + "\n")
