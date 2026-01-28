# Triton Inference Server Dockerfile
# This is a placeholder/template for the Triton server Docker image

# Use NVIDIA Triton Inference Server as base image
FROM nvcr.io/nvidia/tritonserver:25.07-vllm-python-py3

# Set pip cache directory (host should mount ./pip_cache to /pip_cache)
ENV PIP_CACHE_DIR=/pip_cache

# Install uv for fast Python package management, using shared pip cache
# Use --break-system-packages since this is a container environment
RUN pip install --no-cache-dir --break-system-packages uv

# Create a virtual environment for Triton Python backend
# Triton expects the Python backend environment at this specific path
ENV TRITON_VENV=/opt/tritonserver/python_backend_env
RUN uv venv $TRITON_VENV

# Copy requirements file
COPY triton_server/requirements.txt /tmp/requirements.txt

# Install torch packages in the virtual environment
RUN uv pip install --python $TRITON_VENV \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --cache-dir $PIP_CACHE_DIR

# Install grammared-language package with triton extras in the virtual environment
RUN uv pip install --python $TRITON_VENV \
    -r /tmp/requirements.txt --cache-dir $PIP_CACHE_DIR && \
    rm /tmp/requirements.txt

# Copy only necessary files for package installation
# Copy package source code
COPY pyproject.toml /grammared_language/
COPY grammared_language/ /grammared_language/grammared_language/

WORKDIR /grammared_language
RUN uv pip install --python $TRITON_VENV \
    .[triton] --cache-dir $PIP_CACHE_DIR

# Create a symlink to the actual Python site-packages for consistent path references
RUN PYTHON_VERSION=$($TRITON_VENV/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    ln -s $TRITON_VENV/lib/python${PYTHON_VERSION}/site-packages $TRITON_VENV/site-packages

# Copy custom backends or configurations if needed
# COPY triton_server/custom_backends/ /opt/tritonserver/backends/

# Copy model repository would typically be done via volume mount
# But can be copied for production deployments:
# COPY triton_server/model_repository/ /models/

# Set environment variables
ENV MODEL_REPOSITORY=/models
ENV LOG_VERBOSE=0
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HOME=/cache

# Add virtual environment to PATH so Python uses it
ENV PATH="$TRITON_VENV/bin:$PATH"
# Set PYTHONPATH to include the virtual environment's site-packages
# This ensures Triton's Python backend can find installed packages
# Using the symlink created earlier for version-independent path
ENV PYTHONPATH="$TRITON_VENV/site-packages:$PYTHONPATH"

# Change working directory to root to avoid import conflicts with /grammared_language
WORKDIR /

COPY docker/default_model_config.yaml /default_model_config.yaml
COPY docker/triton/build_repo.py /build_repo.py

# Expose ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Default command to build model repository then start Triton server
CMD ["/bin/bash", "-c", "$TRITON_VENV/bin/python /build_repo.py && tritonserver --model-repository=/models --log-verbose=1"]
