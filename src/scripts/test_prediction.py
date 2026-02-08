"""
Quick test prediction script.
Loads the champion model and makes a test prediction.
"""
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.model.registry import get_champion_model
from src.model.predict import predict_single, format_prediction_result
from src.config import PROCESSED_DIR


def test_prediction():
    """Load champion model and make a test prediction."""
    print("\n" + "=" * 70)
    print("TESTING MODEL PREDICTION")
    print("=" * 70)
    
    # Load the champion model
    print("\nLoading champion model from MLflow...")
    try:
        model = get_champion_model()
        print("✓ Champion model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you've:")
        print("  1. Started MLflow server: mlflow server --host 0.0.0.0 --port 5000")
        print("  2. Trained at least one model: python src/scripts/train_model.py")
        return None
    
    # Load sample features
    features_path = PROCESSED_DIR / "features.parquet"
    
    if not features_path.exists():
        print(f"\n❌ No features found at {features_path}")
        print("Run the training script first to generate features.")
        return None
    
    print(f"\nLoading features from {features_path}...")
    features_df = pd.read_parquet(features_path)
    
    # Take the last row as a test example
    test_row = features_df.iloc[-1]
    test_timestamp = test_row["timestamp"]
    actual_demand = test_row["demand_mwh"]
    
    # Prepare features (exclude target and timestamp)
    test_features = test_row.drop(["timestamp", "demand_mwh"]).to_dict()
    
    print("\n" + "-" * 70)
    print("TEST PREDICTION")
    print("-" * 70)
    print(f"Timestamp: {test_timestamp}")
    print(f"Actual Demand: {actual_demand:.2f} MWh")
    
    # Make prediction
    prediction = predict_single(model, test_features)
    
    print(f"Predicted Demand: {prediction:.2f} MWh")
    
    error = abs(prediction - actual_demand)
    pct_error = (error / actual_demand) * 100
    
    print(f"\nAbsolute Error: {error:.2f} MWh ({pct_error:.2f}%)")
    
    # Format result
    result = format_prediction_result(
        prediction=prediction,
        timestamp=str(test_timestamp),
        model_version="champion"
    )
    
    print("\nFormatted API Response:")
    print(result)
    
    print("\n" + "=" * 70)
    print("PREDICTION TEST COMPLETE")
    print("=" * 70 + "\n")
    
    return result


if __name__ == "__main__":
    try:
        test_prediction()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
