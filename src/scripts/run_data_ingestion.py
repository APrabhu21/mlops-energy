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
    # Fetch last 7 days of data
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    
    print(f"Fetching data from {start} to {end}...")
    
    # Fetch demand data
    eia_client = EIAClient()
    demand_df = eia_client.fetch_demand(
        start.strftime('%Y-%m-%dT%H'),
        end.strftime('%Y-%m-%dT%H')
    )
    print(f"✓ Fetched {len(demand_df)} demand records")
    
    # Fetch weather data
    weather_client = WeatherClient()
    weather_df = weather_client.fetch_historical(
        start.strftime('%Y-%m-%d'),
        end.strftime('%Y-%m-%d')
    )
    print(f"✓ Fetched {len(weather_df)} weather records")
    
    # Build features
    features_df = build_features(demand_df, weather_df)
    print(f"✓ Built {len(features_df)} feature rows")
    
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
