"""
Complete end-to-end training script.
Fetches data, engineers features, trains model, and registers it in MLflow.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.data.eia_client import EIAClient
from src.data.weather_client import WeatherClient
from src.data.feature_engineering import build_features
from src.model.train import train_model
from src.model.registry import promote_to_champion
from src.config import RAW_DIR, PROCESSED_DIR, REFERENCE_DIR


def run_training_pipeline(
    start_date: str = "2024-01-01T00",
    end_date: str = "2024-12-31T23",
    use_cached: bool = True,
):
    """
    Run the complete training pipeline.
    
    Args:
        start_date: Start date for data ingestion (format: YYYY-MM-DDT00)
        end_date: End date for data ingestion (format: YYYY-MM-DDT23)
        use_cached: If True, use cached data if available
    """
    print("\n" + "=" * 70)
    print("ENERGY DEMAND FORECASTING - TRAINING PIPELINE")
    print("=" * 70)
    
    # Convert to date strings for weather API
    start_date_str = start_date[:10]  # "YYYY-MM-DD"
    end_date_str = end_date[:10]
    
    # File paths for caching
    raw_demand_path = RAW_DIR / "demand_data.parquet"
    raw_weather_path = RAW_DIR / "weather_data.parquet"
    features_path = PROCESSED_DIR / "features.parquet"
    
    # Step 1: Data Ingestion
    print("\n[1/4] DATA INGESTION")
    print("-" * 70)
    
    if use_cached and raw_demand_path.exists() and raw_weather_path.exists():
        print(f"Loading cached demand data from {raw_demand_path}")
        demand_df = pd.read_parquet(raw_demand_path)
        print(f"Loading cached weather data from {raw_weather_path}")
        weather_df = pd.read_parquet(raw_weather_path)
    else:
        print(f"Fetching demand data from EIA API ({start_date} to {end_date})...")
        eia_client = EIAClient()
        demand_df = eia_client.fetch_demand(start=start_date, end=end_date)
        
        print(f"✓ Fetched {len(demand_df)} demand records")
        
        print(f"Fetching weather data from Open-Meteo ({start_date_str} to {end_date_str})...")
        weather_client = WeatherClient()
        weather_df = weather_client.fetch_historical(
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        print(f"✓ Fetched {len(weather_df)} weather records")
        
        # Cache the raw data
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        demand_df.to_parquet(raw_demand_path, index=False)
        weather_df.to_parquet(raw_weather_path, index=False)
        print(f"✓ Cached raw data to {RAW_DIR}")
    
    # Step 2: Feature Engineering
    print("\n[2/4] FEATURE ENGINEERING")
    print("-" * 70)
    
    if use_cached and features_path.exists():
        print(f"Loading cached features from {features_path}")
        features_df = pd.read_parquet(features_path)
    else:
        print("Building features...")
        features_df = build_features(demand_df, weather_df)
        
        # Save processed features
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(features_path, index=False)
        print(f"✓ Saved features to {features_path}")
    
    print(f"✓ Total feature rows: {len(features_df)}")
    print(f"✓ Date range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
    print(f"✓ Feature columns: {len(features_df.columns)}")
    
    # Step 3: Model Training
    print("\n[3/4] MODEL TRAINING")
    print("-" * 70)
    
    if len(features_df) < 1000:
        print(f"❌ Not enough data to train ({len(features_df)} rows). Need at least 1000.")
        print("Try fetching more data by adjusting the date range.")
        return None
    
    model, metrics, run_id = train_model(features_df)
    
    # Step 4: Model Registration & Promotion
    print("\n[4/4] MODEL REGISTRATION")
    print("-" * 70)
    
    promoted = promote_to_champion(run_id)
    
    # Save reference dataset for drift detection (exclude test set)
    test_size = 168 * 4  # 4 weeks
    reference_df = features_df.iloc[:-test_size]
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    reference_path = REFERENCE_DIR / "reference_features.parquet"
    reference_df.to_parquet(reference_path, index=False)
    print(f"✓ Saved reference dataset ({len(reference_df)} rows) to {reference_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Champion Model: {'YES' if promoted else 'NO'}")
    print(f"MAE: {metrics['mae']:.2f} MWh")
    print(f"RMSE: {metrics['rmse']:.2f} MWh")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"\nView experiment details at: http://localhost:5000")
    print("=" * 70 + "\n")
    
    return {
        "model": model,
        "metrics": metrics,
        "run_id": run_id,
        "promoted": promoted,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the energy demand forecasting model")
    parser.add_argument(
        "--start",
        default="2024-01-01T00",
        help="Start date (YYYY-MM-DDTHH)"
    )
    parser.add_argument(
        "--end",
        default="2024-12-31T23",
        help="End date (YYYY-MM-DDTHH)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-fetch data even if cached"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_training_pipeline(
            start_date=args.start,
            end_date=args.end,
            use_cached=not args.no_cache,
        )
        
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
