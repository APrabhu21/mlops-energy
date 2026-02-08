.PHONY: setup up down logs train drift ingest test lint

# Install dependencies
setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/reference reports
	dvc init

# Start all services (Docker Compose)
up:
	cd infra && docker-compose up -d

# Stop all services
down:
	cd infra && docker-compose down

# View logs
logs:
	cd infra && docker-compose logs -f

# Run incremental ingestion (last 7 days)
ingest:
	python -m src.scripts.run_ingestion

# Train model
train:
	python -m src.scripts.run_training

# Run drift check
drift:
	python -m src.scripts.run_drift_check

# Run all tests
test:
	pytest tests/ -v

# Lint
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Create .env from template
env:
	cp .env.example .env
	@echo "Created .env file. Please edit it and add your EIA_API_KEY"
