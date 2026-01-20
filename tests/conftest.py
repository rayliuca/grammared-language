"""Pytest configuration and shared fixtures."""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_dir():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def triton_server_url():
    """Return the Triton server URL for testing."""
    return "localhost:8000"


@pytest.fixture(scope="session")
def grpc_server_address():
    """Return the gRPC server address for testing."""
    return "localhost:50051"


@pytest.fixture(scope="session")
def api_server_url():
    """Return the API server URL for testing."""
    return "http://localhost:8000"


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that require services (gRPC, API)"
    )
    config.addinivalue_line(
        "markers", "functional: Functional tests for Triton models"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "requires_triton: Tests that require Triton server"
    )
    config.addinivalue_line(
        "markers", "requires_grpc: Tests that require gRPC server"
    )


# Skip conditions based on dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers based on available dependencies."""
    try:
        import tritonclient
        has_triton = True
    except ImportError:
        has_triton = False
    
    try:
        import grpc
        has_grpc = True
    except ImportError:
        has_grpc = False
    
    for item in items:
        # Skip Triton tests if client not available
        if "triton" in str(item.fspath).lower() and not has_triton:
            item.add_marker(
                pytest.mark.skip(reason="tritonclient not installed")
            )
        
        # Skip gRPC tests if client not available
        if "grpc" in str(item.fspath).lower() and not has_grpc:
            item.add_marker(
                pytest.mark.skip(reason="grpcio not installed")
            )
