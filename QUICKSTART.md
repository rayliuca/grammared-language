# Quick Start Guide

Get the Grammared Language system up and running quickly.

## Prerequisites

- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with Docker GPU support for better performance

## 5-Minute Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rayliuca/grammared_language.git
cd grammared_language
```

### 2. Set Up Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env if needed (optional for quick start)
# nano .env
```

### 3. Start Services

```bash
# Build and start all services
docker-compose up --build

# Or in detached mode
docker-compose up -d --build
```

### 4. Verify Setup

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","triton_status":"connected"}
```

## Next Steps

### Test the API

```bash
# Test grammar checking
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The text too check.",
    "language": "en-US"
  }'
```

### Configure LanguageTool

See [LanguageTool Integration Guide](docs/LANGUAGETOOL_INTEGRATION.md) for details.

## Current Status

### What's Ready
- ✅ Project folder structure
- ✅ Comprehensive documentation
- ✅ Docker setup templates
- ✅ GECToR HuggingFace model integration
- ✅ Triton server configuration for HuggingFace models

### What's Next
- ⏳ API service implementation
- ⏳ Complete model testing with running Triton server
- ⏳ LanguageTool connector

## GECToR Model Quick Start

The project now includes support for the GECToR (Grammatical Error Correction: Tag, Not Rewrite) model from HuggingFace.

### Deploy the GECToR Model

```bash
# Download and prepare the model (optional - Triton will do this on first start)
cd triton_server/scripts
./deploy_gector.sh
```

### Start Triton with GECToR

```bash
# Build and start Triton server
docker-compose up --build triton-server

# The GECToR model will be automatically loaded
# Monitor logs to see the model loading process
docker logs -f grammared-triton
```

### Test the Model

Once Triton is running:

```bash
# Install Triton client
pip install tritonclient[http]

# Run the test script
python triton_server/scripts/test_gector.py

# Or specify custom Triton URL
python triton_server/scripts/test_gector.py --url localhost:8002
```

### Model Information

- **Model**: gotutiyan/gector-roberta-base-5k
- **Task**: Grammar Error Correction
- **Backend**: Python (HuggingFace Transformers)
- **Documentation**: See [triton_server/model_repository/gector_roberta/README.md](triton_server/model_repository/gector_roberta/README.md)
- **HuggingFace Integration Guide**: See [docs/HUGGINGFACE_MODELS.md](docs/HUGGINGFACE_MODELS.md)

## Development Setup

For development setup, see the [Development Guide](docs/DEVELOPMENT.md).

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when available)
# pip install -r requirements.txt
```

## Project Structure Overview

```
grammared_language/
├── api/                    # API service (LanguageTool interface)
├── triton_server/         # Triton Inference Server setup
├── models/                # ML model storage
├── docker/                # Docker configurations
├── config/                # Configuration files
└── docs/                  # Documentation
```

## Key Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [API Documentation](docs/API.md) - API endpoints and usage
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Model Documentation](docs/MODELS.md) - ML model details
- [Development Guide](docs/DEVELOPMENT.md) - Developer setup
- [LanguageTool Integration](docs/LANGUAGETOOL_INTEGRATION.md) - Integration guide

## Getting Help

- 📖 Check the documentation in `docs/`
- 🐛 Report issues on GitHub
- 💬 Start a discussion for questions

## Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild services
docker-compose up --build

# Clean up
docker-compose down -v
```

## Troubleshooting

### Services won't start
- Check Docker is running
- Verify ports 8000-8003 are available
- Check logs: `docker-compose logs`

### GPU not detected
- Install NVIDIA Container Toolkit
- Verify GPU access: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

### Connection errors
- Ensure all services are healthy: `docker-compose ps`
- Check network configuration
- Verify environment variables

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

To be determined.
