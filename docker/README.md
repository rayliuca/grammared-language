# Docker Configuration

This directory contains Docker configurations for the project services.

## Structure

```
docker/
├── api/           # Dockerfile and configs for API service
├── triton/        # Dockerfile and configs for Triton server
└── README.md      # This file
```

## Purpose

Provides containerization for:
- API service deployment
- Triton Inference Server deployment
- Development and production environments
- Docker Compose orchestration

## Containers

### API Service (`api/`)
- **Base Image**: Python 3.9+ slim or official Python image
- **Dependencies**: FastAPI/Flask, ML client libraries
- **Exposed Ports**: 8000 (API endpoint)
- **Volumes**: Configuration files, logs

### Triton Server (`triton/`)
- **Base Image**: `nvcr.io/nvidia/tritonserver`
- **Model Repository**: Mounted model repository
- **Exposed Ports**: 8000 (HTTP), 8001 (gRPC), 8002 (metrics)
- **GPU Support**: NVIDIA GPU access

## Docker Compose

A `docker-compose.yml` file at the project root will orchestrate:
- API service container
- Triton server container
- Network configuration between services
- Volume mounts for models and configs

## Future Development

- Dockerfiles for each service
- Docker Compose configuration
- Multi-stage builds for optimization
- Health check configurations
- Environment-specific configurations (dev, staging, prod)
