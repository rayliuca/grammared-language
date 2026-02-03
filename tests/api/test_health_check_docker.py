"""
Docker-based integration tests for health check functionality.

Run these tests to verify the health check works correctly in a Docker container.
"""

import subprocess
import time
import requests
import pytest
import os

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not RUN_NON_HERMETIC,
    reason="Requires RUN_NON_HERMETIC=true (needs Docker or file system access)"
)


def test_health_check_in_docker_container():
    """
    Integration test: Start gRPC server in Docker and verify health check.
    
    Prerequisites:
    - Docker image built: docker build -t grammared-language-api:latest -f docker/api/Dockerfile .
    - curl available in the container
    
    This test:
    1. Starts a Docker container with the API service
    2. Waits for it to become healthy
    3. Tests the health check endpoint
    4. Stops the container
    """
    pytest.skip("Requires Docker and built image")
    
    container_name = "test-grammared-api-health"
    
    # Clean up any existing container
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True
    )
    
    try:
        # Start container
        result = subprocess.run(
            [
                "docker", "run",
                "--name", container_name,
                "-d",
                "-p", "50051:50051",
                "-p", "8000:8000",
                "grammared-language-api:latest"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            pytest.fail(f"Failed to start container: {result.stderr}")
        
        # Wait for container to be healthy
        for attempt in range(30):
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        else:
            pytest.fail("Health check endpoint not available after 30 seconds")
        
        # Verify health check
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        assert response.text == "OK"
        
        # Verify unhealthy path
        response = requests.get("http://localhost:8000/healthz", timeout=5)
        assert response.status_code == 200
        
    finally:
        # Stop and remove container
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True
        )
        subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True
        )


def test_docker_healthcheck_instruction():
    """
    Test that Dockerfile contains proper HEALTHCHECK instruction.
    
    This is a quick validation that the Dockerfile has been updated
    with the health check configuration.
    """
    with open("docker/api/Dockerfile", "r") as f:
        dockerfile_content = f.read()
    
    assert "HEALTHCHECK" in dockerfile_content
    assert "curl -f http://localhost:50052/health" in dockerfile_content
    assert "start-period=30s" in dockerfile_content


def test_docker_compose_health_check():
    """
    Test that docker-compose.yml contains health check configuration.
    """
    with open("docker-compose.yml", "r") as f:
        compose_content = f.read()
    
    # Check that api-service has healthcheck configured
    assert "api-service:" in compose_content
    assert "healthcheck:" in compose_content
    assert "http://localhost:50052/health" in compose_content
    assert "interval: 10s" in compose_content
    assert "start_period: 30s" in compose_content


def test_health_check_ports_exposed():
    """Test that necessary ports are exposed in Dockerfile."""
    with open("docker/api/Dockerfile", "r") as f:
        dockerfile_content = f.read()
    
    assert "EXPOSE" in dockerfile_content
    assert "50051" in dockerfile_content
    assert "50052" in dockerfile_content
