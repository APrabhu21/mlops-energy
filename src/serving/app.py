"""
FastAPI prediction service.
Loads the champion model from MLflow registry and serves predictions.
Exposes metrics to Prometheus.
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.model.registry import get_champion_model
from src.model.predict import predict_single
from src.monitoring.prometheus_metrics import (
    PREDICTION_VALUE,
    MODEL_LOADED,
    REQUEST_ERRORS,
    start_metrics_server,
    shutdown_metrics_server,
    track_prediction_time,
)
from src.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelReloadResponse,
    ErrorResponse,
)
from src.config import MLFLOW_TRACKING_URI, PROMETHEUS_PORT

# Global model reference
model = None
model_load_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads model on startup, starts metrics server.
    """
    global model, model_load_time
    
    print("\n" + "=" * 70)
    print("Starting Energy Demand Forecast API")
    print("=" * 70)
    
    # Start Prometheus metrics server
    try:
        start_metrics_server(PROMETHEUS_PORT)
    except Exception as e:
        print(f"⚠️  Could not start metrics server: {e}")
    
    # Load the champion model
    try:
        print(f"\nLoading champion model from MLflow ({MLFLOW_TRACKING_URI})...")
        model = get_champion_model()
        model_load_time = datetime.utcnow()
        MODEL_LOADED.set(1)
        print("✓ Champion model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("The API will start, but predictions will fail until a model is loaded.")
        MODEL_LOADED.set(0)
    
    print("\n" + "=" * 70)
    print("API is ready!")
    print(f"  API docs: http://localhost:8000/docs")
    print(f"  Metrics:  http://localhost:{PROMETHEUS_PORT}/metrics")
    print("=" * 70 + "\n")
    
    yield
    
    # Shutdown
    print("\nShutting down API...")
    shutdown_metrics_server()
    print("✓ Shutdown complete\n")


app = FastAPI(
    title="Energy Demand Forecast API",
    description="Production ML API for forecasting electricity demand in the NY ISO region",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    REQUEST_ERRORS.labels(error_type=type(exc).__name__).inc()
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump(),
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Energy Demand Forecast API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": f"http://localhost:{PROMETHEUS_PORT}/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        mlflow_uri=MLFLOW_TRACKING_URI,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
@track_prediction_time
async def predict(request: PredictionRequest):
    """
    Generate a demand forecast from input features.
    
    The input should contain all 20 feature values required by the model.
    Returns the predicted electricity demand in MWh.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check the service health and logs.",
        )
    
    try:
        # Convert request to dict for prediction
        features = request.model_dump()
        
        # Make prediction
        prediction = predict_single(model, features)
        
        # Update Prometheus metrics
        PREDICTION_VALUE.observe(prediction)
        
        return PredictionResponse(
            predicted_demand_mwh=round(prediction, 2),
            model_version="champion",
            timestamp=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        REQUEST_ERRORS.labels(error_type="prediction_error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/reload-model", response_model=ModelReloadResponse, tags=["Admin"])
async def reload_model():
    """
    Hot-reload the champion model from MLflow without restarting the service.
    Useful after retraining and promoting a new champion.
    """
    global model, model_load_time
    
    try:
        print("\nReloading model from MLflow...")
        model = get_champion_model()
        model_load_time = datetime.utcnow()
        MODEL_LOADED.set(1)
        print("✓ Model reloaded successfully")
        
        return ModelReloadResponse(
            status="success",
            message="Champion model reloaded successfully",
            timestamp=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        MODEL_LOADED.set(0)
        REQUEST_ERRORS.labels(error_type="model_reload_error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}",
        )


@app.get("/model-info", tags=["Admin"])
async def model_info():
    """Get information about the currently loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded",
        )
    
    return {
        "model_type": type(model).__name__,
        "model_version": "champion",
        "loaded_at": model_load_time.isoformat() if model_load_time else None,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    
    uvicorn.run(
        "src.serving.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,  # Enable auto-reload during development
    )
