"""
Data quality checks for ingested data.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import pandas as pd
from src.config import PROCESSED_DIR


def check_data_quality(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check data quality and return (is_valid, issues_list).
    """
    issues = []
    
    # Check for minimum rows
    if len(df) < 24:
        issues.append(f"CRITICAL: Insufficient data - only {len(df)} rows (need at least 24)")
    
    # Check for missing critical columns
    required_cols = ['timestamp', 'demand_mwh', 'temperature']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"CRITICAL: Missing columns: {missing}")
    
    # Check for excessive nulls (over 80% is critical)
    null_pct = df.isnull().sum() / len(df)
    high_null_cols = null_pct[null_pct > 0.8].index.tolist()
    if high_null_cols:
        issues.append(f"CRITICAL: Columns with >80% nulls: {high_null_cols}")
    
    # Check for invalid demand values (should be cleaned by ingestion)
    if 'demand_mwh' in df.columns:
        invalid_demand = (df['demand_mwh'] <= 0) | (df['demand_mwh'] >= 100000)
        invalid_count = invalid_demand.sum()
        if invalid_count > 0:
            pct = (invalid_count / len(df)) * 100
            if pct > 5:  # Only critical if >5% are invalid
                issues.append(f"CRITICAL: {invalid_count} invalid demand values ({pct:.1f}%)")
            else:
                issues.append(f"WARNING: {invalid_count} invalid demand values ({pct:.1f}%)")
    
    # Check for duplicate timestamps (should be cleaned by ingestion)
    if 'timestamp' in df.columns:
        dupes = df['timestamp'].duplicated().sum()
        if dupes > 0:
            pct = (dupes / len(df)) * 100
            if pct > 5:  # Only critical if >5% are duplicates
                issues.append(f"CRITICAL: {dupes} duplicate timestamps ({pct:.1f}%)")
            else:
                issues.append(f"WARNING: {dupes} duplicate timestamps ({pct:.1f}%)")
    
    # Only fail on CRITICAL issues
    critical_issues = [i for i in issues if i.startswith('CRITICAL')]
    is_valid = len(critical_issues) == 0
    return is_valid, issues


def main():
    features_path = PROCESSED_DIR / 'features.parquet'
    print(f"Checking data quality: {features_path}")
    
    df = pd.read_parquet(features_path)
    is_valid, issues = check_data_quality(df)
    
    if is_valid:
        print("✓ Data quality check passed")
        print(f"  {len(df)} rows, {len(df.columns)} columns")
        sys.exit(0)
    else:
        print("✗ Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
