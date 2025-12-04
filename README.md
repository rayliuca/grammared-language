# Grammared Language

A complex text grammar error correction backend for LanguageTool using remote rules and ML models.

## Overview

This project provides an advanced grammar error correction system that:
- Serves as a backend API for LanguageTool using their remote rule feature
- Leverages ML models (from BERT to LLMs) for sophisticated error detection and correction
- Uses NVIDIA Triton Inference Server for high-performance model serving
- Provides fully containerized deployment via Docker

## Project Goals

1. **API Service**: Serve a LanguageTool-compatible API using various ML systems, packaged in Docker
2. **ML Model Serving**: Serve ML models (from BERT to LLMs) using Triton Inference Server and Docker

## Project Structure

```
grammared_language/
├── api/                    # API service (LanguageTool interface)
│   ├── src/               # Source code for API
│   ├── tests/             # Tests for API service
│   └── README.md          # API documentation
├── triton_server/         # Triton Inference Server setup
│   ├── model_repository/  # Triton model repository
│   ├── scripts/           # Deployment and management scripts
│   └── README.md          # Triton server documentation
├── models/                # ML model storage
│   ├── bert/              # BERT-based models
│   ├── llm/               # Large Language Models
│   └── README.md          # Model documentation
├── docker/                # Docker configurations
│   ├── api/               # API service Dockerfile
│   ├── triton/            # Triton server Dockerfile
│   └── README.md          # Docker documentation
├── config/                # Configuration files
│   ├── api/               # API configuration
│   ├── triton/            # Triton configuration
│   └── README.md          # Configuration documentation
├── docs/                  # Project documentation
│   └── README.md          # Documentation index
└── README.md              # This file
```

## Architecture

### Components

1. **API Service**
   - Implements LanguageTool remote rule API
   - Routes requests to appropriate ML models via Triton
   - Formats responses in LanguageTool-compatible format
   - Handles authentication, rate limiting, and error handling

2. **Triton Inference Server**
   - Serves multiple ML models concurrently
   - Provides GPU-accelerated inference
   - Supports dynamic batching for throughput optimization
   - Enables model versioning and A/B testing

3. **ML Models**
   - BERT-based models for fast error detection
   - Large Language Models for complex corrections
   - Fine-tuned models for specific error types

### Data Flow

```
LanguageTool Client
    ↓
API Service (FastAPI/Flask)
    ↓
Triton Inference Server
    ↓
ML Models (BERT/LLM)
    ↓
Grammar Corrections
    ↓
API Service (Format Response)
    ↓
LanguageTool Client
```

## Technology Stack

- **API Framework**: FastAPI or Flask (Python)
- **Model Serving**: NVIDIA Triton Inference Server
- **ML Frameworks**: PyTorch, TensorFlow, ONNX
- **Containerization**: Docker, Docker Compose
- **Models**: BERT, GPT-based LLMs, custom transformer models

## Quick Start

> **Note**: This section will be populated once implementation is complete.

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU (optional, for GPU acceleration)
- NVIDIA Container Toolkit (for GPU support)

### Setup
```bash
# Clone the repository
git clone https://github.com/rayliuca/grammared_language.git
cd grammared_language

# Build and start services
docker-compose up --build
```

## Development

### Project Setup
Instructions for setting up the development environment will be added here.

### Testing
Testing guidelines and commands will be documented here.

### Contributing
Contribution guidelines will be provided here.

## Documentation

For detailed documentation on specific components, refer to:
- [API Service Documentation](api/README.md)
- [Triton Server Documentation](triton_server/README.md)
- [Model Documentation](models/README.md)
- [Docker Documentation](docker/README.md)
- [Configuration Documentation](config/README.md)
- [Full Documentation](docs/README.md)

## License

To be determined.

## Contact

For questions and support, please open an issue in the repository.