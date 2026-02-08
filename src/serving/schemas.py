"""
Pydantic schemas for FastAPI request/response validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Input features for prediction. All fields correspond to feature columns."""
    temperature: float = Field(..., description="Temperature in °C")
    apparent_temperature: float = Field(..., description="Feels-like temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover (%)")
    solar_radiation: float = Field(..., ge=0, description="Solar radiation in W/m²")
    precipitation: float = Field(..., ge=0, description="Precipitation in mm")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_weekend: bool = Field(..., description="Is it a weekend?")
    is_holiday: bool = Field(..., description="Is it a US federal holiday?")
    demand_lag_1h: float = Field(..., description="Demand 1 hour ago (MWh)")
    demand_lag_24h: float = Field(..., description="Demand 24 hours ago (MWh)")
    demand_lag_168h: float = Field(..., description="Demand 168 hours (1 week) ago (MWh)")
    demand_rolling_mean_24h: float = Field(..., description="24-hour rolling mean demand (MWh)")
    demand_rolling_std_24h: float = Field(..., ge=0, description="24-hour rolling std demand (MWh)")
    temp_squared: float = Field(..., description="Temperature squared")
    cooling_degree_hours: float = Field(..., ge=0, description="Cooling degree hours")
    heating_degree_hours: float = Field(..., ge=0, description="Heating degree hours")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "temperature": 5.2,
                "apparent_temperature": 2.8,
                "humidity": 75.0,
                "wind_speed": 4.5,
                "cloud_cover": 80.0,
                "solar_radiation": 150.0,
                "precipitation": 0.0,
                "hour_of_day": 14,
                "day_of_week": 2,
                "month": 2,
                "is_weekend": False,
                "is_holiday": False,
                "demand_lag_1h": 18500.0,
                "demand_lag_24h": 19200.0,
                "demand_lag_168h": 18800.0,
                "demand_rolling_mean_24h": 18900.0,
                "demand_rolling_std_24h": 1200.0,
                "temp_squared": 27.04,
                "cooling_degree_hours": 0.0,
                "heating_degree_hours": 13.1,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response containing the prediction result."""
    predicted_demand_mwh: float = Field(..., description="Predicted electricity demand in MWh")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Timestamp when prediction was made")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_demand_mwh": 18750.25,
                "model_version": "champion",
                "timestamp": "2024-02-08T14:30:00",
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    mlflow_uri: str = Field(..., description="MLflow tracking URI")
    timestamp: str = Field(..., description="Current timestamp")


class ModelReloadResponse(BaseModel):
    """Response for model reload operation."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Timestamp of reload")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Timestamp of error")
