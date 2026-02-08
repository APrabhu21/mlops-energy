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
) -> Tuple[lgb.LGBMRegressor, Dict[str, float], str]:
    """
    Train a LightGBM model on the provided feature dataframe.

    Uses time-based split (no shuffling — respects temporal order).
    Logs everything to MLflow.

    Returns:
        (trained_model, metrics_dict, run_id)
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
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("train_start", str(df["timestamp"].iloc[0]))
        mlflow.log_param("train_end", str(df["timestamp"].iloc[-test_size - 1]))
        mlflow.log_param("test_start", str(df["timestamp"].iloc[-test_size]))
        mlflow.log_param("test_end", str(df["timestamp"].iloc[-1]))

        # Train
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
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

        print(f"\n{'='*60}")
        print(f"Run ID: {run.info.run_id}")
        print(f"{'='*60}")
        print(f"Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name.upper()}: {metric_value:.2f}")
        print(f"{'='*60}")

        return model, metrics, run.info.run_id
