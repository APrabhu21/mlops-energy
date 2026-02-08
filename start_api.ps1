# PowerShell script to start the FastAPI server
Write-Host "Starting Energy Demand Forecast API..." -ForegroundColor Green
Write-Host ""
Write-Host "The API will be available at:" -ForegroundColor Cyan
Write-Host "  - API docs:  http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  - Health:    http://localhost:8000/health" -ForegroundColor Yellow
Write-Host "  - Metrics:   http://localhost:8001/metrics" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Red
Write-Host ""

python -m uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --reload
