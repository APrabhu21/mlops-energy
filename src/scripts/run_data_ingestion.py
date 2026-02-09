"""
Run data ingestion pipeline for GitHub Actions or manual execution.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Debug: Print paths
print(f"Script location: {__file__}")
print(f"Project root: {project_root}")
print(f"sys.path[0]: {sys.path[0]}")
print(f"src exists: {(project_root / 'src').exists()}")

from datetime import datetime, timedelta
import pandas as pd

from src.data.eia_client import EIAClient
from src.data.weather_client import WeatherClient
from src.data.feature_engineering import build_features
from src.config import RAW_DIR, PROCESSED_DIR


def main():
    eia_client = EIAClient()
    weather_client = WeatherClient()
    output_path = PROCESSED_DIR / 'features.parquet'
    
    # Check if we need to backfill historical data
    needs_backfill = False
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        
        # Check data span
        min_ts = existing_df['timestamp'].min()
        max_ts = existing_df['timestamp'].max()
        data_span_days = (max_ts - min_ts).days
        
        print(f"Existing data: {len(existing_df)} rows, spanning {data_span_days} days")
        print(f"Date range: {min_ts} to {max_ts}")
        
        # If we have less than 1 year of data, backfill to 2 years
        if data_span_days < 365:
            needs_backfill = True
            print(f"⚠️ Less than 1 year of data - backfilling to 2 years...")
    else:
        needs_backfill = True
        print("No existing data - fetching 2 years of historical data...")
    
    # Determine fetch period
    end = datetime.utcnow()
    if needs_backfill:
        # Fetch full 2 years
        start = end - timedelta(days=730)
        print(f"Backfilling: Fetching 2 years from {start} to {end}...")
    else:
        # Normal operation: fetch last 30 days for updates
        start = end - timedelta(days=30)
        print(f"Regular update: Fetching last 30 days from {start} to {end}...")
    
    # Fetch demand data
    demand_df = eia_client.fetch_demand(
        start.strftime('%Y-%m-%dT%H'),
        end.strftime('%Y-%m-%dT%H')
    )
    print(f"✓ Fetched {len(demand_df)} demand records")
    
    # Fetch weather data
    weather_df = weather_client.fetch_historical(
        start.strftime('%Y-%m-%d'),
        end.strftime('%Y-%m-%d')
    )
    print(f"✓ Fetched {len(weather_df)} weather records")
    
    # Build features
    features_df = build_features(demand_df, weather_df)
    print(f"✓ Built {len(features_df)} feature rows")
    
    # If existing features exist, merge with new data (keep last 730 days total)
    if output_path.exists():
        print("Merging with existing data...")
        existing_df = pd.read_parquet(output_path)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        
        # Combine and deduplicate
        features_df = pd.concat([existing_df, features_df], ignore_index=True)
        features_df = features_df.drop_duplicates(subset=['timestamp'], keep='last')
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        
        # Keep last 730 days (2 years) for seasonal patterns and year-over-year trends
        cutoff = features_df['timestamp'].max() - timedelta(days=730)
        features_df = features_df[features_df['timestamp'] >= cutoff]
        print(f"✓ Merged to {len(features_df)} total rows (up to 2 years)")
    
    # Clean data
    print("Cleaning data...")
    initial_rows = len(features_df)
    
    # Remove rows with invalid demand values
    features_df = features_df[
        (features_df['demand_mwh'] > 0) & 
        (features_df['demand_mwh'] < 100000)
    ]
    
    # Remove duplicate timestamps (keep first occurrence)
    features_df = features_df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Sort by timestamp
    features_df = features_df.sort_values('timestamp').reset_index(drop=True)
    
    removed_rows = initial_rows - len(features_df)
    if removed_rows > 0:
        print(f"  Removed {removed_rows} invalid/duplicate rows")
    
    print(f"✓ Cleaned to {len(features_df)} valid rows")
    
    # Save features
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / 'features.parquet'
    features_df.to_parquet(output_path, index=False)
    print(f"✓ Saved features to {output_path}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Columns: {list(features_df.columns)}")


if __name__ == "__main__":
    main()
