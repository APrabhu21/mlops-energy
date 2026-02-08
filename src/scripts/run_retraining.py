"""
Run model retraining for GitHub Actions or manual execution.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import pandas as pd

from src.model.train import train_model
from src.model.registry import promote_to_champion
from src.config import PROCESSED_DIR
import shutil


def main():
    # Load features
    features_path = PROCESSED_DIR / 'features.parquet'
    print(f"Loading features from {features_path}...")
    features = pd.read_parquet(features_path)
    print(f"✓ Loaded {len(features)} rows, {len(features.columns)} columns")
    
    # Calculate appropriate test size (use last 20% or min 24 hours)
    test_size = max(24, int(len(features) * 0.2))
    print(f"Using test_size: {test_size} rows ({test_size/len(features):.1%})")
    
    # Train model
    print("Training model...")
    model, metrics, run_id = train_model(features, test_size=test_size)
    
    print(f"✓ Model trained: {run_id}")
    print(f"  Metrics: {metrics}")
    
    # Try to promote to champion
    print("Evaluating for champion promotion...")
    promoted = promote_to_champion(run_id)
    
    if promoted:
        print("✓ New champion model promoted!")
        
        # Update reference data with current training data
        reference_dir = project_root / 'data' / 'reference'
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_path = reference_dir / 'reference_features.parquet'
        
        print(f"Updating reference data at {reference_path}...")
        shutil.copy2(features_path, reference_path)
        print("✓ Reference data updated with new training data")
    else:
        print("○ Model not better than current champion - keeping existing reference data")


if __name__ == "__main__":
    main()
