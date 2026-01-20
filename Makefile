# Makefile for common development tasks

.PHONY: help install install-dev test test-unit test-integration test-functional test-coverage lint format clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install           - Install package dependencies"
	@echo "  make install-dev       - Install package with dev dependencies"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make test-functional   - Run functional tests only"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo "  make lint              - Run linters"
	@echo "  make format            - Format code"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make docker-build      - Build Docker images"
	@echo "  make docker-up         - Start Docker services"
	@echo "  make docker-down       - Stop Docker services"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	@echo "Starting integration tests (requires gRPC server)..."
	pytest tests/integration/ -v

test-functional:
	@echo "Starting functional tests (requires Triton server)..."
	pytest tests/functional/ -v

test-coverage:
	pytest tests/ -v --cov=grammared_language --cov-report=html --cov-report=term
	@echo "Coverage report generated at htmlcov/index.html"

lint:
	ruff check grammared_language/ tests/
	mypy grammared_language/

format:
	black grammared_language/ tests/ api/
	isort grammared_language/ tests/ api/

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
