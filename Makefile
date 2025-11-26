# Makefile for Flaschen Depot Project

.PHONY: install clean test lint format run-api run-mlflow docker-build docker-run help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist .pytest_cache .coverage htmlcov

test:  ## Run tests
	pytest tests/ -v --cov=src/flaschen_depot --cov-report=html --cov-report=term-missing

test-fast:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linters
	flake8 src/flaschen_depot
	black --check src/flaschen_depot
	isort --check-only src/flaschen_depot

format:  ## Format code
	black src/flaschen_depot
	isort src/flaschen_depot

train:  ## Run training pipeline
	python scripts/train.py

run-api:  ## Run FastAPI server
	uvicorn flaschen_depot.api:app --host 0.0.0.0 --port 8000 --reload

run-mlflow:  ## Run MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

docker-build:  ## Build Docker image
	docker build -t flaschen_depot:latest .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 flaschen_depot:latest

docker-compose-up:  ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down:  ## Stop all services
	docker-compose down

setup-dvc:  ## Initialize DVC
	dvc init
	dvc remote add -d local ./dvc-storage

all: clean install lint test  ## Clean, install, lint, and test
