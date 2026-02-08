"""
Run model retraining for GitHub Actions or manual execution.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.model.train import train_model
from src.model.registry import promote_to_champion
from src.config import PROCESSED_DIR


def main():
    # Load features
    features_path = PROCESSED_DIR / 'features.parquet'
    print(f"Loading features from {features_path}...")
    features = pd.read_parquet(features_path)
    print(f"✓ Loaded {len(features)} rows, {len(features.columns)} columns")
    
    # Train model
    print("Training model...")
    run_id, metrics = train_model(
        features,
        target_col='demand_mwh',
        experiment_name='energy-demand-auto-retrain'
    )
    
    print(f"✓ Model trained: {run_id}")
    print(f"  Metrics: {metrics}")
    
    # Try to promote to champion
    print("Evaluating for champion promotion...")
    promoted = promote_to_champion(run_id)
    
    if promoted:
        print("✓ New champion model promoted!")
    else:
        print("○ Model not better than current champion")


if __name__ == "__main__":
    main()
