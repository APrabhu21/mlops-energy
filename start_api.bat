@echo off
REM Start the FastAPI server
echo Starting Energy Demand Forecast API...
echo.
echo API will be available at:
echo   - API docs: http://localhost:8000/docs
echo   - Health:   http://localhost:8000/health
echo   - Metrics:  http://localhost:8001/metrics
echo.
echo Press Ctrl+C to stop
echo.

python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
