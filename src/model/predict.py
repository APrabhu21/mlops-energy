"""
Inference logic for making predictions with trained models.
"""
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from src.config import FEATURE_COLUMNS


def predict_single(model, features: Union[pd.DataFrame, Dict]) -> float:
    """
    Make a single prediction.
    
    Args:
        model: Trained model (LightGBM or loaded from MLflow)
        features: Dictionary or DataFrame with feature values
        
    Returns:
        Predicted demand in MWh
    """
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    # Ensure correct column order
    features = features[FEATURE_COLUMNS]
    
    prediction = model.predict(features)[0]
    return float(prediction)


def predict_batch(model, features_df: pd.DataFrame) -> np.ndarray:
    """
    Make batch predictions.
    
    Args:
        model: Trained model
        features_df: DataFrame with feature columns
        
    Returns:
        Array of predictions
    """
    # Ensure correct column order
    features = features_df[FEATURE_COLUMNS]
    
    predictions = model.predict(features)
    return predictions


def predict_with_intervals(
    model,
    features_df: pd.DataFrame,
    n_iterations: int = 100,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Make predictions with confidence intervals using quantile regression.
    
    Note: This is a simple implementation. For production, consider using
    LightGBM's built-in quantile regression or a separate uncertainty model.
    
    Args:
        model: Trained model
        features_df: DataFrame with features
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level (default 95%)
        
    Returns:
        DataFrame with columns: prediction, lower_bound, upper_bound
    """
    base_predictions = predict_batch(model, features_df)
    
    # For now, use a simple heuristic based on model residuals
    # In production, this should be based on validation set residuals
    # or proper uncertainty quantification
    alpha = 1 - confidence
    std_multiplier = 1.96  # for 95% confidence
    
    # Estimate prediction std (this is a placeholder - improve in production)
    prediction_std = base_predictions * 0.05  # Assume 5% uncertainty
    
    lower_bound = base_predictions - std_multiplier * prediction_std
    upper_bound = base_predictions + std_multiplier * prediction_std
    
    return pd.DataFrame({
        "prediction": base_predictions,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    })


def format_prediction_result(
    prediction: float,
    timestamp: str = None,
    model_version: str = "unknown",
) -> Dict:
    """
    Format a prediction result for API response or logging.
    
    Args:
        prediction: Predicted demand value
        timestamp: Timestamp for the prediction
        model_version: Model version identifier
        
    Returns:
        Dictionary with formatted prediction result
    """
    result = {
        "predicted_demand_mwh": round(prediction, 2),
        "model_version": model_version,
    }
    
    if timestamp:
        result["timestamp"] = timestamp
    
    return result
