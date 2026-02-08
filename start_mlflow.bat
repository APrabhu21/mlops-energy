@echo off
REM Start MLflow tracking server locally
echo Starting MLflow server on http://localhost:5000
echo Press Ctrl+C to stop

mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
