# API Service Dockerfile
# This is a placeholder/template for the API service Docker image

# Use Python 3.12 slim as base image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git bash build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management, using shared pip cache
# Use --break-system-packages since this is a container environment
RUN pip install --no-cache-dir --break-system-packages uv

# Create a virtual environment for Python backend
ENV API_VENV=/python_api_env
COPY docker/requirements/base.txt /tmp/base.txt
RUN uv venv $API_VENV && uv pip install --python $API_VENV --requirement /tmp/base.txt torch==2.10.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm /tmp/base.txt

# Copy only necessary files for package installation
# Copy package source code
COPY pyproject.toml /grammared_language/
COPY grammared_language/ /grammared_language/grammared_language/

WORKDIR /grammared_language
RUN uv pip install --python $API_VENV .[docker] && \
    find $API_VENV -type f -name "*.pyc" -delete && \
    find $API_VENV -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find $API_VENV -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find $API_VENV -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf $API_VENV/lib/python*/site-packages/torch/test $API_VENV/lib/python*/site-packages/*/tests && \
    rm -rf /root/.cache /tmp/* && \
    PYTHON_VERSION=$($API_VENV/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    ln -s $API_VENV/lib/python${PYTHON_VERSION}/site-packages $API_VENV/site-packages

# Add virtual environment to PATH so Python uses it
ENV PATH="$API_VENV/bin:$PATH"
# Set PYTHONPATH to include the virtual environment's site-packages
# This ensures Python backend can find installed packages
# Using the symlink created earlier for version-independent path
ENV PYTHONPATH="$API_VENV/site-packages:$PYTHONPATH"

# Download spaCy English model for errant
RUN $API_VENV/bin/python -m spacy download en_core_web_sm

# Change working directory to root to avoid import conflicts with /grammared_language
WORKDIR /

COPY docker/default_model_config.yaml /default_model_config.yaml
# Copy application code
COPY api/src/ src/

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /src

USER apiuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Placeholder command
CMD ["/bin/bash", "-c", "$API_VENV/bin/python /src/grpc_server.py"]

