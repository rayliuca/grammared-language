# Triton Inference Server Dockerfile
# This is a placeholder/template for the Triton server Docker image

# Use NVIDIA Triton Inference Server as base image
# Version should be chosen based on model requirements
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Create Python backend execution environment
RUN mkdir -p /opt/tritonserver/python_backend_env && \
    uv venv /opt/tritonserver/python_backend_env

# Copy only necessary files for package installation
# Copy package source code
COPY pyproject.toml /grammared_language/
COPY grammared_language/ /grammared_language/grammared_language/

# Copy requirements file
COPY triton_server/requirements.txt /tmp/requirements.txt

WORKDIR /grammared_language

# install torch packages
RUN uv pip install --python /opt/tritonserver/python_backend_env/bin/python \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install grammared-language package with triton extras in the Python backend environment
RUN uv pip install --python /opt/tritonserver/python_backend_env/bin/python \
    -e ".[triton]" && \
    uv pip install --python /opt/tritonserver/python_backend_env/bin/python \
    -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Also install in the base environment for compatibility
RUN uv pip install --system -e ".[triton]" transformers sentencepiece protobuf

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

# Expose ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Default command to start Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
