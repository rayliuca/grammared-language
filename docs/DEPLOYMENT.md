# Deployment Guide

## Overview

This guide covers deployment of the Grammared Language system using Docker and Docker Compose.

## Prerequisites

### Required
- Docker Engine 20.10+
- Docker Compose 2.0+
- Git

### Optional (for GPU support)
- NVIDIA GPU with CUDA support
- NVIDIA Driver 470+
- NVIDIA Container Toolkit

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/rayliuca/grammared_language.git
cd grammared_language
```

### 2. Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 3. Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Check Triton server status
curl http://localhost:8001/v2/health/ready
```

## Docker Compose Configuration

Example `docker-compose.yml` structure:

```yaml
version: '3.8'

services:
  triton-server:
    build:
      context: ./docker/triton
      dockerfile: Dockerfile
    image: grammared-triton:latest
    ports:
      - "8001:8001"  # gRPC
      - "8002:8002"  # HTTP
      - "8003:8003"  # Metrics
    volumes:
      - ./triton_server/model_repository:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/v2/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 3

  api-service:
    build:
      context: ./docker/api
      dockerfile: Dockerfile
    image: grammared-api:latest
    ports:
      - "8000:8000"
    environment:
      - TRITON_HOST=triton-server
      - TRITON_PORT=8001
    depends_on:
      triton-server:
        condition: service_healthy
    volumes:
      - ./config/api:/app/config
```

## Manual Docker Deployment

### Build Images

```bash
# Build Triton server image
docker build -t grammared-triton:latest -f docker/triton/Dockerfile .

# Build API service image
docker build -t grammared-api:latest -f docker/api/Dockerfile .
```

### Create Network

```bash
docker network create grammared-network
```

### Run Triton Server

```bash
docker run -d \
  --name triton-server \
  --network grammared-network \
  --gpus all \
  -p 8001:8001 \
  -p 8002:8002 \
  -p 8003:8003 \
  -v $(pwd)/triton_server/model_repository:/models \
  grammared-triton:latest
```

### Run API Service

```bash
docker run -d \
  --name api-service \
  --network grammared-network \
  -p 8000:8000 \
  -e TRITON_HOST=triton-server \
  -e TRITON_PORT=8001 \
  -v $(pwd)/config/api:/app/config \
  grammared-api:latest
```

## Environment Variables

### API Service

- `TRITON_HOST`: Triton server hostname (default: localhost)
- `TRITON_PORT`: Triton gRPC port (default: 8001)
- `API_PORT`: API service port (default: 8000)
- `API_HOST`: API service host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MAX_TEXT_LENGTH`: Maximum text length (default: 10000)

### Triton Server

- `MODEL_REPOSITORY`: Path to model repository (default: /models)
- `STRICT_MODEL_CONFIG`: Require model config files (default: true)
- `LOG_VERBOSE`: Verbose logging level (default: 0)

## GPU Configuration

### Check GPU Availability

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Configure GPU Resources

In docker-compose.yml:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Specific GPU
          capabilities: [gpu]
```

Or for all GPUs:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Volume Mounts

### Model Repository
```bash
-v $(pwd)/triton_server/model_repository:/models
```

### Configuration Files
```bash
-v $(pwd)/config/api:/app/config
```

### Logs
```bash
-v $(pwd)/logs:/app/logs
```

## Networking

### Port Mappings

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| API | 8000 | HTTP | REST API |
| Triton | 8001 | gRPC | Model inference |
| Triton | 8002 | HTTP | Model inference |
| Triton | 8003 | HTTP | Metrics |

## Health Checks

### API Service

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "triton_status": "connected"}
```

### Triton Server

```bash
# Liveness check
curl http://localhost:8002/v2/health/live

# Readiness check
curl http://localhost:8002/v2/health/ready

# Model status
curl http://localhost:8002/v2/models
```

## Scaling

### Horizontal Scaling (Multiple API Instances)

```yaml
api-service:
  deploy:
    replicas: 3
```

### Load Balancing

Use nginx or Traefik as reverse proxy:

```nginx
upstream api_backend {
    server api-service-1:8000;
    server api-service-2:8000;
    server api-service-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_backend;
    }
}
```

## Monitoring

### Prometheus Metrics

Triton exposes metrics at `:8003/metrics`

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['triton-server:8003']
```

### Logging

View logs:
```bash
# API service logs
docker-compose logs -f api-service

# Triton server logs
docker-compose logs -f triton-server
```

## Troubleshooting

### Common Issues

**Issue: Triton server fails to start**
- Check model repository structure
- Verify config.pbtxt files
- Check GPU availability

**Issue: API cannot connect to Triton**
- Verify network connectivity
- Check TRITON_HOST and TRITON_PORT
- Wait for Triton health check to pass

**Issue: Out of GPU memory**
- Reduce model batch size
- Use smaller model versions
- Reduce concurrent instances

### Debug Mode

Enable debug logging:

```yaml
environment:
  - LOG_LEVEL=DEBUG
  - LOG_VERBOSE=1
```

## Production Deployment

### Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Implement authentication
- [ ] Set up rate limiting
- [ ] Configure firewalls
- [ ] Use secrets management
- [ ] Enable audit logging
- [ ] Regular security updates

### Performance Optimization

- [ ] Enable model warming
- [ ] Configure dynamic batching
- [ ] Set appropriate timeouts
- [ ] Monitor resource usage
- [ ] Implement caching
- [ ] Use CDN for static assets

### Backup and Recovery

- [ ] Backup model repository
- [ ] Backup configuration files
- [ ] Document recovery procedures
- [ ] Test disaster recovery

## Cloud Deployment

### AWS

- Use ECS/EKS for container orchestration
- Use EC2 GPU instances (p3/p4)
- Store models in S3
- Use Application Load Balancer

### Google Cloud

- Use GKE for Kubernetes
- Use Compute Engine with GPUs
- Store models in Cloud Storage
- Use Cloud Load Balancing

### Azure

- Use AKS for Kubernetes
- Use NCv3/NDv2 VM series with GPUs
- Store models in Blob Storage
- Use Azure Load Balancer

## Kubernetes Deployment

> Detailed Kubernetes deployment guide to be added

Key resources:
- Deployment for API service
- StatefulSet for Triton server
- Service for load balancing
- ConfigMap for configuration
- PersistentVolume for models
