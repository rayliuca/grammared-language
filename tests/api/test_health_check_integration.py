"""
Integration tests for gRPC server health check endpoints.

These are non-hermetic tests that start the actual gRPC server and test
the health check functionality via HTTP and gRPC protocols.
"""

import pytest
import time
import threading
import requests
import os
from concurrent import futures

try:
    import grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not GRPC_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires grpcio and RUN_NON_HERMETIC=true (needs running gRPC server)"
)


def test_http_health_check_endpoint_when_healthy(grpc_server_fixture):
    """Test HTTP health check endpoint returns 200 when service is healthy."""
    # Give server time to start
    time.sleep(0.5)
    
    response = requests.get("http://localhost:50054/health", timeout=5)
    assert response.status_code == 200
    assert response.text == "OK"


def test_http_healthz_endpoint_when_healthy(grpc_server_fixture):
    """Test HTTP /healthz endpoint returns 200 when service is healthy."""
    time.sleep(0.5)
    
    response = requests.get("http://localhost:50054/healthz", timeout=5)
    assert response.status_code == 200
    assert response.text == "OK"


def test_http_health_check_invalid_path(grpc_server_fixture):
    """Test HTTP health check returns 404 for invalid paths."""
    time.sleep(0.5)
    
    response = requests.get("http://localhost:50054/invalid", timeout=5)
    assert response.status_code == 404


def test_http_health_check_content_type(grpc_server_fixture):
    """Test HTTP health check response has correct content type."""
    time.sleep(0.5)
    
    response = requests.get("http://localhost:50054/health", timeout=5)
    assert response.headers.get("Content-type") == "text/plain"


def test_multiple_http_health_checks(grpc_server_fixture):
    """Test multiple consecutive health checks work correctly."""
    time.sleep(0.5)
    
    for _ in range(5):
        response = requests.get("http://localhost:50054/health", timeout=5)
        assert response.status_code == 200
        assert response.text == "OK"


def test_http_health_check_timeout_handling(grpc_server_fixture):
    """Test HTTP health check handles timeout gracefully."""
    time.sleep(0.5)
    
    # Should succeed with reasonable timeout
    response = requests.get("http://localhost:50054/health", timeout=10)
    assert response.status_code == 200


@pytest.fixture(scope="function")
def grpc_server_fixture():
    """
    Fixture to start and stop the gRPC server for testing.
    
    This is a non-hermetic test that requires:
    - The grammared_language package to be installed
    - Model configuration file to exist
    - Triton server to be running (if using real models)
    """
    import sys
    import os
    
    # Set up test environment - use different port from docker service (50051)
    os.environ['GRAMMARED_LANGUAGE__API_PORT'] = '50053'
    os.environ['GRAMMARED_LANGUAGE__API_HOST'] = '0.0.0.0'
    
    # Import here to avoid import errors if dependencies aren't available
    try:
        from api.src.grpc_server import serve
    except ImportError:
        pytest.skip("gRPC server module not available")
    
    # Start server in background thread with different health port
    server_thread = threading.Thread(
        target=lambda: serve(host='0.0.0.0', port=50053, health_port=50054),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to be ready
    time.sleep(2)
    
    yield
    
    # Cleanup: server will stop when main thread exits
    # For graceful shutdown, you might want to signal the server
    time.sleep(0.5)


# Additional test that could be run with actual mock models
@pytest.mark.skip(reason="Requires full model setup")
def test_grpc_server_with_actual_models():
    """
    Full integration test with actual models.
    
    This test is skipped by default because it requires:
    1. Model configuration to be set up
    2. Triton inference server to be running
    3. Models to be loaded
    """
    import time
    from api.src.grpc_server import serve
    
    # Start server
    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    time.sleep(3)
    
    # Test health endpoint
    response = requests.get("http://localhost:50054/health")
    assert response.status_code == 200


# Test that can be run without full server initialization
@pytest.mark.skip(reason="Requires server to not be running")
def test_http_health_check_endpoint_not_available():
    """Test that health check endpoint is not available if server isn't running."""
    try:
        response = requests.get("http://localhost:50054/health", timeout=1)
        pytest.fail("Server should not be running")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Expected - server is not running
        pass
