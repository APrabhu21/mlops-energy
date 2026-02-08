"""
Feature engineering pipeline.
Joins demand + weather data, creates time features, lag features,
and temperature interaction terms.
"""
import pandas as pd
import numpy as np
from typing import Optional

# US federal holidays — use the 'holidays' library
import holidays

US_HOLIDAYS = holidays.US()


def build_features(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join demand and weather data, then engineer features.

    demand_df must have columns: [period, demand_mwh]
    weather_df must have columns: [timestamp, temperature_2m, ...]

    Returns a DataFrame ready for model training with all feature columns.
    """
    # Standardize column names
    demand = demand_df[["period", "demand_mwh"]].copy()
    demand = demand.rename(columns={"period": "timestamp"})
    demand["timestamp"] = pd.to_datetime(demand["timestamp"]).dt.floor("h")

    weather = weather_df.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"]).dt.floor("h")

    # Merge on timestamp
    df = pd.merge(demand, weather, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rename weather columns to cleaner names
    rename_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "apparent_temperature": "apparent_temperature",
        "wind_speed_10m": "wind_speed",
        "cloud_cover": "cloud_cover",
        "shortwave_radiation": "solar_radiation",
        "precipitation": "precipitation",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ---- Time features ----
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek        # 0=Monday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_holiday"] = df["timestamp"].dt.date.apply(lambda d: d in US_HOLIDAYS)

    # ---- Lag features ----
    df["demand_lag_1h"] = df["demand_mwh"].shift(1)
    df["demand_lag_24h"] = df["demand_mwh"].shift(24)
    df["demand_lag_168h"] = df["demand_mwh"].shift(168)       # 1 week

    # ---- Rolling features ----
    df["demand_rolling_mean_24h"] = df["demand_mwh"].rolling(24).mean()
    df["demand_rolling_std_24h"] = df["demand_mwh"].rolling(24).std()

    # ---- Temperature interaction features ----
    # These capture the nonlinear relationship between temperature and demand
    # (both very hot and very cold increase demand)
    df["temp_squared"] = df["temperature"] ** 2
    # Cooling degree hours: how much above 65°F (~18.3°C)
    df["cooling_degree_hours"] = np.maximum(0, df["temperature"] - 18.3)
    # Heating degree hours: how much below 65°F (~18.3°C)
    df["heating_degree_hours"] = np.maximum(0, 18.3 - df["temperature"])

    # Drop rows with NaN from lag/rolling features
    df = df.dropna().reset_index(drop=True)

    return df
