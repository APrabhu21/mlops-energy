"""
Run data ingestion pipeline for GitHub Actions or manual execution.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
import pandas as pd

from src.data.eia_api import fetch_demand_data
from src.data.weather_api import fetch_weather_data
from src.data.features import build_features
from src.config import RAW_DIR, PROCESSED_DIR


def main():
    # Fetch last 7 days of data
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    
    print(f"Fetching data from {start} to {end}...")
    
    # Fetch demand data
    demand_df = fetch_demand_data(
        start.strftime('%Y-%m-%dT%H'),
        end.strftime('%Y-%m-%dT%H')
    )
    print(f"✓ Fetched {len(demand_df)} demand records")
    
    # Fetch weather data
    weather_df = fetch_weather_data(
        start.strftime('%Y-%m-%d'),
        end.strftime('%Y-%m-%d')
    )
    print(f"✓ Fetched {len(weather_df)} weather records")
    
    # Build features
    features_df = build_features(demand_df, weather_df)
    print(f"✓ Built {len(features_df)} feature rows")
    
    # Save features
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / 'features.parquet'
    features_df.to_parquet(output_path, index=False)
    print(f"✓ Saved features to {output_path}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Columns: {list(features_df.columns)}")


if __name__ == "__main__":
    main()
