"""
Drift detection using Evidently AI.
Compares current production data against the reference (training) dataset.
Generates reports and extracts metrics for Prometheus/Grafana.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.config import FEATURE_COLUMNS, DRIFT_SHARE_THRESHOLD, REFERENCE_DIR
from src.monitoring.html_templates import generate_drift_html


def run_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list = None,
) -> Tuple[bool, float, Dict]:
    """
    Run Evidently data drift detection.

    Args:
        reference_df: Historical/training data (baseline)
        current_df: Recent production data to check for drift
        feature_columns: List of feature columns to check (default: FEATURE_COLUMNS)

    Returns:
        (should_retrain: bool, drift_share: float, details: dict)
    """
    if feature_columns is None:
        feature_columns = [col for col in FEATURE_COLUMNS if col in reference_df.columns]
    
    # Use only feature columns for drift analysis (exclude target and timestamp)
    ref_cols = [col for col in feature_columns if col in reference_df.columns]
    cur_cols = [col for col in feature_columns if col in current_df.columns]
    
    # Find common columns
    common_cols = list(set(ref_cols) & set(cur_cols))
    
    if not common_cols:
        raise ValueError("No common feature columns between reference and current data")
    
    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()
    
    print(f"Running drift detection on {len(common_cols)} features...")
    print(f"Reference data: {len(ref)} rows")
    print(f"Current data: {len(cur)} rows")
    
    # Create and run drift report using preset (v0.7.20 API)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    
    # Extract drift_share from the preset (v0.7.20 API)
    # After run(), the preset in metrics list has drift_share attribute
    drift_share = 0.0
    if report.metrics:
        preset = report.metrics[0]
        drift_share = preset.drift_share if hasattr(preset, 'drift_share') else 0.0
    
    dataset_drift = drift_share > DRIFT_SHARE_THRESHOLD
    per_feature_drift = {}
    n_drifted = int(drift_share * len(common_cols)) if drift_share > 0 else 0
    
    # Determine dataset-level drift
    dataset_drift = drift_share > DRIFT_SHARE_THRESHOLD
    
    # Decision: retrain if drift_share exceeds threshold
    should_retrain = drift_share > DRIFT_SHARE_THRESHOLD
    
    details = {
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "n_drifted_features": n_drifted,
        "n_total_features": len(common_cols),
        "per_feature": per_feature_drift,
        "threshold": DRIFT_SHARE_THRESHOLD,
        "check_timestamp": datetime.utcnow().isoformat(),
    }
    
    return should_retrain, drift_share, details


def generate_drift_html_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
    feature_columns: list = None,
) -> str:
    """
    Generate and save an HTML drift report for visual inspection.
    
    Args:
        reference_df: Reference dataset
        current_df: Current dataset
        output_path: Path to save HTML report
        feature_columns: Features to analyze
        
    Returns:
        Path to saved report
    """
    if feature_columns is None:
        feature_columns = [col for col in FEATURE_COLUMNS if col in reference_df.columns]
    
    # Use only feature columns
    ref_cols = [col for col in feature_columns if col in reference_df.columns]
    cur_cols = [col for col in feature_columns if col in current_df.columns]
    common_cols = list(set(ref_cols) & set(cur_cols))
    
    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()
    
    # Run drift detection
    should_retrain, drift_share, details = run_drift_check(reference_df, current_df, feature_columns)
    
    # Calculate basic statistics for visualization
    ref_stats = ref.describe().to_dict()
    cur_stats = cur.describe().to_dict()
    
    # Generate custom HTML report
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    html_content = generate_drift_html(
        drift_share=drift_share,
        details=details,
        ref_stats=ref_stats,
        cur_stats=cur_stats,
        ref_size=len(ref),
        cur_size=len(cur),
        feature_columns=common_cols
    )
    
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Drift report saved to: {output_path}")
    
    return str(output_path_obj)


def load_reference_data(reference_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the reference dataset for drift comparison."""
    if reference_path is None:
        reference_path = REFERENCE_DIR / "reference_features.parquet"
    
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {reference_path}. "
            "Train a model first to generate the reference dataset."
        )
    
    return pd.read_parquet(reference_path)


def save_drift_results(
    drift_details: Dict,
    output_path: str = "reports/drift_results.json",
) -> None:
    """Save drift check results to JSON file."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        json.dump(drift_details, f, indent=2)
    
    print(f"✓ Drift results saved to: {output_path}")


def print_drift_summary(should_retrain: bool, drift_share: float, details: Dict) -> None:
    """Print a formatted drift detection summary."""
    print("\n" + "=" * 70)
    print("DRIFT DETECTION SUMMARY")
    print("=" * 70)
    print(f"Dataset Drift Detected:  {'YES' if details['dataset_drift'] else 'NO'}")
    print(f"Drift Share:             {drift_share:.2%}")
    print(f"Drifted Features:        {details['n_drifted_features']} / {details['n_total_features']}")
    print(f"Threshold:               {details['threshold']:.2%}")
    print(f"Retrain Recommended:     {'YES' if should_retrain else 'NO'}")
    print("=" * 70)
    
    if details['n_drifted_features'] > 0:
        print("\nDrifted Features:")
        for feature, info in details['per_feature'].items():
            if info['drifted']:
                print(f"  - {feature}: score={info['score']:.4f}, method={info['method']}")
    
    print()
