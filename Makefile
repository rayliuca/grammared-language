# Makefile for common development tasks using uv

.PHONY: help install install-dev test test-api test-clients test-classifier test-triton test-utils test-non-hermetic test-coverage lint format clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install              - Install package dependencies with uv"
	@echo "  make install-dev          - Install package with dev dependencies"
	@echo "  make test                 - Run all tests in tests/"
	@echo "  make test-api             - Run API tests"
	@echo "  make test-clients         - Run client tests"
	@echo "  make test-classifier      - Run classifier tests"
	@echo "  make test-triton          - Run Triton tests"
	@echo "  make test-utils           - Run utils tests"
	@echo "  make test-non-hermetic    - Run non-hermetic tests (requires running services)"
	@echo "  make test-coverage        - Run tests with coverage report"
	@echo "  make lint                 - Run linters"
	@echo "  make format               - Format code"
	@echo "  make clean                - Clean build artifacts"
	@echo "  make docker-build         - Build Docker images"
	@echo "  make docker-up            - Start Docker services"
	@echo "  make docker-down          - Stop Docker services"

install:
	uv sync

install-dev:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

test-api:
	uv run pytest tests/api/ -v

test-clients:
	uv run pytest tests/clients/ -v

test-classifier:
	uv run pytest tests/grammared_classifier/ -v

test-triton:
	uv run pytest tests/triton/ -v

test-utils:
	uv run pytest tests/utils/ -v

test-non-hermetic:
	@echo "Running non-hermetic tests (requires running services)..."
	RUN_NON_HERMETIC=true uv run pytest tests/ -v -m "not skip"

test-coverage:
	uv run pytest tests/ -v --cov=grammared_language --cov-report=html --cov-report=term
	@echo "Coverage report generated at htmlcov/index.html"

lint:
	uv run ruff check grammared_language/ tests/ api/
	uv run mypy grammared_language/

format:
	uv run black grammared_language/ tests/ api/
	uv run isort grammared_language/ tests/ api/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart