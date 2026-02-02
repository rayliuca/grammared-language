"""
Unit tests for health check components.

These tests verify the health check implementation without requiring
a full gRPC server or external dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from http.server import BaseHTTPRequestHandler
from io import BytesIO


def create_mock_request():
    """Create a mock socket request with makefile support."""
    mock_request = Mock()
    mock_request.makefile = Mock(side_effect=lambda *args, **kwargs: BytesIO(b"GET /health HTTP/1.1\r\n\r\n"))
    return mock_request


def test_health_check_handler_returns_200_when_healthy():
    """Test HealthCheckHTTPHandler returns 200 when service_healthy is True."""
    # Import the handler class
    from api.src.grpc_server import HealthCheckHTTPHandler
    import api.src.grpc_server as grpc_server_module
    
    # Create a mock socket request
    mock_request = create_mock_request()
    
    # Mock the handler
    handler = HealthCheckHTTPHandler(
        request=mock_request,
        client_address=("localhost", 12345),
        server=MagicMock()
    )
    
    # Mock the handler methods
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.path = "/health"
    
    # Set service as healthy
    with patch.object(grpc_server_module, 'service_healthy', True):
        handler.do_GET()
    
    # Verify response
    handler.send_response.assert_called_once_with(200)
    handler.send_header.assert_called_with('Content-type', 'text/plain')
    handler.wfile.write.assert_called_with(b'OK')


def test_health_check_handler_returns_503_when_not_healthy():
    """Test HealthCheckHTTPHandler returns 503 when service_healthy is False."""
    from api.src.grpc_server import HealthCheckHTTPHandler
    import api.src.grpc_server as grpc_server_module
    
    mock_request = create_mock_request()
    
    handler = HealthCheckHTTPHandler(
        request=mock_request,
        client_address=("localhost", 12345),
        server=MagicMock()
    )
    
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.path = "/health"
    
    # Set service as not healthy
    with patch.object(grpc_server_module, 'service_healthy', False):
        handler.do_GET()
    
    # Verify response
    handler.send_response.assert_called_once_with(503)
    handler.send_header.assert_called_with('Content-type', 'text/plain')
    handler.wfile.write.assert_called_with(b'Service not ready')


def test_health_check_handler_accepts_healthz_path():
    """Test HealthCheckHTTPHandler accepts both /health and /healthz paths."""
    from api.src.grpc_server import HealthCheckHTTPHandler
    import api.src.grpc_server as grpc_server_module
    
    mock_request = create_mock_request()
    
    handler = HealthCheckHTTPHandler(
        request=mock_request,
        client_address=("localhost", 12345),
        server=MagicMock()
    )
    
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.path = "/healthz"
    
    with patch.object(grpc_server_module, 'service_healthy', True):
        handler.do_GET()
    
    handler.send_response.assert_called_once_with(200)


def test_health_check_handler_returns_404_for_invalid_path():
    """Test HealthCheckHTTPHandler returns 404 for invalid paths."""
    from api.src.grpc_server import HealthCheckHTTPHandler
    
    mock_request = create_mock_request()
    
    handler = HealthCheckHTTPHandler(
        request=mock_request,
        client_address=("localhost", 12345),
        server=MagicMock()
    )
    
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.path = "/invalid"
    
    handler.do_GET()
    
    handler.send_response.assert_called_once_with(404)


def test_health_check_handler_suppresses_logging():
    """Test HealthCheckHTTPHandler suppresses HTTP server logging."""
    from api.src.grpc_server import HealthCheckHTTPHandler
    
    mock_request = create_mock_request()
    
    handler = HealthCheckHTTPHandler(
        request=mock_request,
        client_address=("localhost", 12345),
        server=MagicMock()
    )
    
    # log_message should return None (no output)
    result = handler.log_message("test format %s", "arg")
    assert result is None
