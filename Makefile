# Makefile for common development tasks using uv

.PHONY: help install install-dev test test-api test-clients test-classifier test-triton test-utils test-non-hermetic test-coverage lint format clean docker-build docker-up docker-down docker-buildx-setup docker-buildx-clean grammared-language-api grammared-language-api-gpu grammared-language-triton grammared-language-triton-arm grammared-language-api-push grammared-language-api-gpu-push grammared-language-triton-push grammared-language-triton-arm-push docker-build-all docker-push-all

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
	@echo "  make docker-build                  - Build Docker images"
	@echo "  make docker-up                     - Start Docker services"
	@echo "  make docker-down                   - Stop Docker services"
	@echo "  make docker-buildx-setup           - Setup buildx for multi-platform builds"
	@echo "  make docker-buildx-clean           - Clean up buildx builder and caches"
	@echo "  make grammared-language-api        - Build API Docker image (amd64+arm64, local)"
	@echo "  make grammared-language-api-gpu    - Build API GPU Docker image (amd64 only, local)"
	@echo "  make grammared-language-triton     - Build Triton Docker image (amd64 only, local)"
	@echo "  make grammared-language-triton-arm - Build Triton Docker image (arm64 only, local)"
	@echo "  make grammared-language-api-push   - Build and push API Docker image (amd64+arm64)"
	@echo "  make grammared-language-api-gpu-push - Build and push API GPU Docker image (amd64 only)"
	@echo "  make grammared-language-triton-push  - Build and push Triton Docker image (amd64 only)"
	@echo "  make grammared-language-triton-arm-push - Build and push Triton Docker image (arm64 only)"
	@echo "  make docker-build-all              - Build all Docker images (local)"
	@echo "  make docker-push-all               - Build and push all Docker images"

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

# Docker buildx cleanup
docker-buildx-clean:
	@echo "Removing multiarch builder..."
	@docker buildx rm multiarch 2>/dev/null || true
	@echo "Cleaning build caches..."
	@rm -rf build/buildx-cache-amd64 build/buildx-cache-arm64
	@echo "Buildx cleanup complete"

# Docker image build targets
GRRAMMARED_LANGUAGE_VERSION := $(shell grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
DOCKER_HUB_USERNAME ?= rayliuca

# Cache directories for different architectures
BUILDX_CACHE_AMD64 := build/buildx-cache-amd64
BUILDX_CACHE_ARM64 := build/buildx-cache-arm64

# Only import local cache if it exists to avoid warnings (runtime check with deferred assignment)
CACHE_FROM_AMD64 = $(if $(wildcard $(BUILDX_CACHE_AMD64)/index.json),--cache-from=type=local$(comma)src=$(BUILDX_CACHE_AMD64),)
CACHE_FROM_ARM64 = $(if $(wildcard $(BUILDX_CACHE_ARM64)/index.json),--cache-from=type=local$(comma)src=$(BUILDX_CACHE_ARM64),)

# Local cache flags (overridable per target)
LOCAL_CACHE_FROM_AMD64 = $(CACHE_FROM_AMD64)
LOCAL_CACHE_FROM_ARM64 = $(CACHE_FROM_ARM64)
LOCAL_CACHE_TO_AMD64 = --cache-to=type=local,dest=$(BUILDX_CACHE_AMD64),mode=max
LOCAL_CACHE_TO_ARM64 = --cache-to=type=local,dest=$(BUILDX_CACHE_ARM64),mode=max

# Comma variable for function arguments
comma := ,

# Setup buildx for multi-platform builds
docker-buildx-setup:
	@echo "Setting up QEMU for multi-architecture support..."
	@docker run --rm --privileged multiarch/qemu-user-static --reset -p yes > /dev/null 2>&1 || true
	@mkdir -p $(BUILDX_CACHE_AMD64) $(BUILDX_CACHE_ARM64)
	@echo "Creating or updating buildx builder 'multiarch'..."
	@docker buildx create --name multiarch --driver docker-container --use 2>/dev/null || docker buildx use multiarch
	@docker buildx inspect --bootstrap
	@echo "Verifying ARM64 platform support..."
	@docker buildx inspect | grep -q "linux/arm64" || (echo "ERROR: ARM64 platform not supported by builder" && exit 1)

# Internal build functions
define docker_build_api_amd64
	docker build \
		-t $(1)grammared-language-api:$(GRRAMMARED_LANGUAGE_VERSION)-amd64 \
		-t $(1)grammared-language-api:latest-amd64 \
		-f docker/api/Dockerfile .
endef

define docker_build_api_arm
	docker buildx build --platform linux/arm64 \
		-t $(1)grammared-language-api:$(GRRAMMARED_LANGUAGE_VERSION)-arm64 \
		-t $(1)grammared-language-api:latest-arm64 \
		-f docker/api/Dockerfile . \
		--cache-from=type=registry,ref=$(1)grammared-language-api:buildcache \
		$(LOCAL_CACHE_FROM_ARM64) \
		--cache-to=type=inline \
		$(LOCAL_CACHE_TO_ARM64) \
		$(2)
endef

define docker_build_api_gpu_amd64
	docker build \
		-t $(1)grammared-language-api-gpu:$(GRRAMMARED_LANGUAGE_VERSION) \
		-t $(1)grammared-language-api-gpu:latest \
		-f docker/api/Dockerfile-gpu .
endef

define docker_build_triton_amd64
	docker build \
		-t $(1)grammared-language-triton:$(GRRAMMARED_LANGUAGE_VERSION) \
		-t $(1)grammared-language-triton:latest \
		-f docker/triton/Dockerfile .
endef

define docker_build_triton_arm
	docker buildx build --platform linux/arm64 \
		-t $(1)grammared-language-triton-arm:$(GRRAMMARED_LANGUAGE_VERSION) \
		-t $(1)grammared-language-triton-arm:latest \
		-f docker/triton/Dockerfile-arm . \
		--cache-from=type=registry,ref=$(1)grammared-language-triton-arm:buildcache \
		$(LOCAL_CACHE_FROM_ARM64) \
		--cache-to=type=inline \
		$(LOCAL_CACHE_TO_ARM64) \
		$(2)
endef

# Build targets (local, multi-arch where applicable)
grammared-language-api: docker-buildx-setup
	$(call docker_build_api_amd64,)
	$(call docker_build_api_arm,,--load)

grammared-language-api-gpu:
	$(call docker_build_api_gpu_amd64,)

grammared-language-triton:
	$(call docker_build_triton_amd64,)

grammared-language-triton-arm: docker-buildx-setup
	$(call docker_build_triton_arm,,--load)

# Push targets (calls build with --push and registry cache)
grammared-language-api-push: docker-buildx-setup
	$(call docker_build_api_amd64,$(DOCKER_HUB_USERNAME)/)
	docker push $(DOCKER_HUB_USERNAME)/grammared-language-api:$(GRRAMMARED_LANGUAGE_VERSION)-amd64
	docker push $(DOCKER_HUB_USERNAME)/grammared-language-api:latest-amd64
	$(call docker_build_api_arm,$(DOCKER_HUB_USERNAME)/,--push --cache-to=type=registry$(comma)ref=$(DOCKER_HUB_USERNAME)/grammared-language-api:buildcache$(comma)mode=max)
	docker buildx imagetools create -t $(DOCKER_HUB_USERNAME)/grammared-language-api:$(GRRAMMARED_LANGUAGE_VERSION) \
		$(DOCKER_HUB_USERNAME)/grammared-language-api:$(GRRAMMARED_LANGUAGE_VERSION)-amd64 \
		$(DOCKER_HUB_USERNAME)/grammared-language-api:$(GRRAMMARED_LANGUAGE_VERSION)-arm64
	docker buildx imagetools create -t $(DOCKER_HUB_USERNAME)/grammared-language-api:latest \
		$(DOCKER_HUB_USERNAME)/grammared-language-api:latest-amd64 \
		$(DOCKER_HUB_USERNAME)/grammared-language-api:latest-arm64

grammared-language-api-gpu-push:
	$(call docker_build_api_gpu_amd64,$(DOCKER_HUB_USERNAME)/)
	docker push $(DOCKER_HUB_USERNAME)/grammared-language-api-gpu:$(GRRAMMARED_LANGUAGE_VERSION)
	docker push $(DOCKER_HUB_USERNAME)/grammared-language-api-gpu:latest

grammared-language-triton-push:
	$(call docker_build_triton_amd64,$(DOCKER_HUB_USERNAME)/)
	docker push $(DOCKER_HUB_USERNAME)/grammared-language-triton:$(GRRAMMARED_LANGUAGE_VERSION)
	docker push $(DOCKER_HUB_USERNAME)/grammared-language-triton:latest

grammared-language-triton-arm-push: docker-buildx-setup
	$(call docker_build_triton_arm,$(DOCKER_HUB_USERNAME)/,--push --cache-to=type=registry$(comma)ref=$(DOCKER_HUB_USERNAME)/grammared-language-triton-arm:buildcache$(comma)mode=max)

docker-build-all: grammared-language-api grammared-language-api-gpu grammared-language-triton grammared-language-triton-arm

docker-push-all: grammared-language-api-push grammared-language-api-gpu-push grammared-language-triton-push grammared-language-triton-arm-push

constraints-client:
# 	uv export -o constraints/constraints-base.txt
	uv pip compile pyproject.toml -o constraints/constraints-base.txt --emit-index-url
	grep -v '^-e \.' constraints/constraints-base.txt > constraints/constraints-base.txt.tmp && mv constraints/constraints-base.txt.tmp constraints/constraints-base.txt

constraints-triton:
# 	uv export -o constraints/constraints-triton.txt --extra triton
	uv pip compile pyproject.toml -o constraints/constraints-triton.txt --extra triton --emit-index-url
	grep -v '^-e \.' constraints/constraints-triton.txt > constraints/constraints-triton.txt.tmp && mv constraints/constraints-triton.txt.tmp constraints/constraints-triton.txt

constraints-dev:
	uv export -o constraints/constraints-all.txt --extra all
	grep -v '^-e \.' constraints/constraints-all.txt > constraints/constraints-all.txt.tmp && mv constraints/constraints-all.txt.tmp constraints/constraints-all.txt

constraints-all: | constraints-client constraints-triton constraints-triton-gpu constraints-dev