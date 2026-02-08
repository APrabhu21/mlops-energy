"""
Performance monitoring - track model metrics over time.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.config import PROCESSED_DIR


def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape)
    }


def log_performance(df: pd.DataFrame, log_path: Path):
    """
    Calculate and log current model performance.
    Appends to performance log file.
    """
    # Calculate metrics on predictions
    y_true = df['demand_mwh']
    y_pred = df['prediction']
    
    # Remove invalid values
    valid_mask = (y_true > 0) & (~y_true.isna()) & (~y_pred.isna())
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) < 10:
        print("⚠ Not enough valid data points for metrics")
        return
    
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create log entry
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(y_true),
        "data_start": df['timestamp'].min().isoformat(),
        "data_end": df['timestamp'].max().isoformat(),
        **metrics
    }
    
    # Load existing log or create new
    if log_path.exists():
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = {"entries": []}
    
    # Append new entry
    log["entries"].append(entry)
    
    # Keep only last 100 entries
    log["entries"] = log["entries"][-100:]
    
    # Save
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"✓ Performance logged:")
    print(f"  MAE: {metrics['mae']:,.0f} MWh")
    print(f"  RMSE: {metrics['rmse']:,.0f} MWh")
    print(f"  R²: {metrics['r2']:.3f}")
    print(f"  MAPE: {metrics['mape']:.1f}%")
    print(f"  Samples: {len(y_true)}")
    
    # Alert if performance is poor
    if metrics['mape'] > 20:
        print(f"⚠ WARNING: High MAPE ({metrics['mape']:.1f}%) - model may be degrading")
    if metrics['r2'] < 0.5:
        print(f"⚠ WARNING: Low R² ({metrics['r2']:.3f}) - poor predictions")


def main():
    features_path = PROCESSED_DIR / 'features.parquet'
    log_path = Path('reports/performance_log.json')
    
    print("Loading features with predictions...")
    df = pd.read_parquet(features_path)
    
    if 'prediction' not in df.columns:
        print("✗ No predictions found - run predictions first")
        return
    
    print(f"Calculating performance on {len(df)} samples...")
    log_performance(df, log_path)


if __name__ == "__main__":
    main()
