"""
Generate predictions for features data using the champion model.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import pandas as pd

from src.model.registry import get_champion_model
from src.config import PROCESSED_DIR, FEATURE_COLUMNS


def main():
    # Load features
    features_path = PROCESSED_DIR / 'features.parquet'
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"✓ Loaded {len(df)} rows")
    
    # Load champion model
    print("Loading champion model...")
    model = get_champion_model()
    print(f"✓ Model loaded")
    
    # Generate predictions
    print("Generating predictions...")
    X = df[FEATURE_COLUMNS]
    predictions = model.predict(X)
    df['prediction'] = predictions
    print(f"✓ Generated {len(predictions)} predictions")
    
    # Save updated features with predictions
    df.to_parquet(features_path, index=False)
    print(f"✓ Saved predictions to {features_path}")


if __name__ == "__main__":
    main()
