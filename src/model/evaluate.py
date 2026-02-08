"""
Evaluation metrics and utilities for model assessment.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from typing import Dict, Optional


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_dataframe: bool = False,
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        return_dataframe: If True, return metrics as a DataFrame
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "r2": r2_score(y_true, y_pred),
        "mean_error": np.mean(y_pred - y_true),  # bias
        "std_error": np.std(y_pred - y_true),
        "max_error": np.max(np.abs(y_pred - y_true)),
    }
    
    if return_dataframe:
        return pd.DataFrame([metrics])
    
    return metrics


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    Handles zero values gracefully.
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error):       {metrics['mae']:>10.2f} MWh")
    print(f"RMSE (Root Mean Squared Error):  {metrics['rmse']:>10.2f} MWh")
    print(f"MAPE (Mean Absolute % Error):    {metrics['mape']:>10.2f} %")
    print(f"R² Score:                        {metrics['r2']:>10.4f}")
    print(f"Mean Error (Bias):               {metrics['mean_error']:>10.2f} MWh")
    print(f"Std Error:                       {metrics['std_error']:>10.2f} MWh")
    print(f"Max Absolute Error:              {metrics['max_error']:>10.2f} MWh")
    print("=" * 60 + "\n")
