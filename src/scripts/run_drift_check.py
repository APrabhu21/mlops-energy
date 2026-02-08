"""
Run drift detection for GitHub Actions or manual execution.
Outputs should_retrain status for workflow decision.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import pandas as pd

from src.monitoring.drift_detection import run_drift_check, save_drift_results
from src.config import PROCESSED_DIR


def main():
    # Load reference and current data
    ref_path = Path('data/reference/reference_features.parquet')
    cur_path = PROCESSED_DIR / 'features.parquet'
    
    print(f"Loading reference data from {ref_path}...")
    ref = pd.read_parquet(ref_path)
    print(f"✓ Reference: {len(ref)} rows")
    
    print(f"Loading current data from {cur_path}...")
    cur = pd.read_parquet(cur_path)
    print(f"✓ Current: {len(cur)} rows")
    
    # Run drift check
    print("Running drift detection...")
    should_retrain, drift_share, details = run_drift_check(ref, cur)
    
    # Save results
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    results_path = reports_dir / 'drift_results.json'
    save_drift_results(details, results_path)
    
    print(f"✓ Drift check complete")
    print(f"  Drift share: {drift_share:.1%}")
    print(f"  Should retrain: {should_retrain}")
    print(f"  Results saved to {results_path}")
    
    # Set GitHub Actions output
    github_output = os.getenv('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"should_retrain={str(should_retrain).lower()}\n")
        print(f"✓ Set GitHub output: should_retrain={should_retrain}")


if __name__ == "__main__":
    main()
