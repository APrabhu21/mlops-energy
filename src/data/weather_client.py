"""
Client for Open-Meteo API — fetches historical and forecast weather data.
No API key required for non-commercial use.
"""
import requests
import pandas as pd
from typing import List, Optional
from src.config import (
    OPEN_METEO_HISTORICAL_URL, OPEN_METEO_FORECAST_URL,
    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_TIMEZONE,
)

HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "surface_pressure", "shortwave_radiation",
]


class WeatherClient:
    def __init__(
        self,
        latitude: float = WEATHER_LATITUDE,
        longitude: float = WEATHER_LONGITUDE,
        timezone: str = WEATHER_TIMEZONE,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone

    def fetch_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical hourly weather data.
        start_date/end_date format: "2023-01-01"
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": self.timezone,
        }
        
        try:
            response = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()["hourly"]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected weather API response format: {e}")
            print(f"Response: {response.text[:500]}")
            raise

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"time": "timestamp"})
        return df

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch current weather forecast for the next N days."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": self.timezone,
        }
        
        try:
            response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()["hourly"]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather forecast: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected forecast API response format: {e}")
            raise

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"time": "timestamp"})
        return df
