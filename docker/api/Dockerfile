# API Service Dockerfile
# This is a placeholder/template for the API service Docker image

# Use Python 3.9 slim as base image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
# COPY requirements.txt .

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# COPY api/src/ ./src/

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Placeholder command
CMD echo "API Service Dockerfile - Implementation pending"
