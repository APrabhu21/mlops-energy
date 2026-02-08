"""
Script to run drift detection check.
Compares recent data against the reference dataset and generates reports.
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.monitoring.drift_detection import (
    run_drift_check,
    generate_drift_html_report,
    load_reference_data,
    save_drift_results,
    print_drift_summary,
)
from src.monitoring.prometheus_metrics import DRIFT_SHARE, DRIFT_DETECTED, RETRAIN_COUNTER
from src.config import PROCESSED_DIR


def run_drift_check_pipeline(
    days_back: int = 7,
    generate_report: bool = True,
) -> dict:
    """
    Run complete drift detection pipeline.
    
    Args:
        days_back: Number of days of recent data to check
        generate_report: Whether to generate HTML report
        
    Returns:
        Dictionary with drift check results
    """
    print("\n" + "=" * 70)
    print("DRIFT DETECTION PIPELINE")
    print("=" * 70)
    
    # Load reference data
    print(f"\n[1/4] Loading reference dataset...")
    try:
        reference_df = load_reference_data()
        print(f"✓ Loaded {len(reference_df)} reference samples")
        print(f"  Date range: {reference_df['timestamp'].min()} to {reference_df['timestamp'].max()}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    # Load recent production data
    print(f"\n[2/4] Loading recent production data ({days_back} days)...")
    features_path = PROCESSED_DIR / "features.parquet"
    
    if not features_path.exists():
        print(f"❌ No production data found at {features_path}")
        print("Run data ingestion first to generate features.")
        return None
    
    all_features = pd.read_parquet(features_path)
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    current_df = all_features[all_features['timestamp'] >= cutoff].copy()
    
    if len(current_df) == 0:
        print(f"❌ No data found in the last {days_back} days")
        return None
    
    print(f"✓ Loaded {len(current_df)} current samples")
    print(f"  Date range: {current_df['timestamp'].min()} to {current_df['timestamp'].max()}")
    
    # Run drift detection
    print(f"\n[3/4] Running drift detection...")
    should_retrain, drift_share, details = run_drift_check(reference_df, current_df)
    
    print_drift_summary(should_retrain, drift_share, details)
    
    # Update Prometheus metrics
    DRIFT_SHARE.set(drift_share)
    DRIFT_DETECTED.set(1.0 if should_retrain else 0.0)
    
    if should_retrain:
        RETRAIN_COUNTER.labels(reason="drift").inc()
    
    # Save results
    save_drift_results(details, "reports/drift_results.json")
    
    # Generate HTML report
    if generate_report:
        print(f"\n[4/4] Generating HTML report...")
        report_path = generate_drift_html_report(
            reference_df,
            current_df,
            "reports/drift_report.html"
        )
        print(f"✓ View report at: {report_path}")
    
    # Final recommendation
    print("\n" + "=" * 70)
    if should_retrain:
        print("⚠️  RECOMMENDATION: RETRAIN MODEL")
        print(f"Drift share ({drift_share:.1%}) exceeds threshold ({details['threshold']:.1%})")
        print("\nTo retrain:")
        print("  python src/scripts/train_model.py")
    else:
        print("✓ NO ACTION NEEDED")
        print(f"Drift share ({drift_share:.1%}) is within acceptable limits")
    print("=" * 70 + "\n")
    
    return {
        "should_retrain": should_retrain,
        "drift_share": drift_share,
        "details": details,
        "report_path": report_path if generate_report else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift detection on recent production data")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of recent data to check (default: 7)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_drift_check_pipeline(
            days_back=args.days,
            generate_report=not args.no_report,
        )
        
        if result and result["should_retrain"]:
            sys.exit(1)  # Exit with error code if retrain is needed
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
