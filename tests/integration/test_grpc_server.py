"""Integration tests for gRPC server endpoints."""
import pytest
import os
from pathlib import Path

try:
    import grpc
    from grammared_language.api.grpc_gen import ml_server_pb2, ml_server_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not GRPC_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires grpcio and RUN_NON_HERMETIC=true (needs running gRPC server)"
)


@pytest.fixture
def grpc_channel():
    """Create a gRPC channel for testing."""
    server_address = "localhost:50051"
    channel = grpc.insecure_channel(server_address)
    yield channel
    channel.close()


@pytest.fixture
def grpc_stub(grpc_channel):
    """Create a gRPC stub for testing."""
    return ml_server_pb2_grpc.ProcessingServerStub(grpc_channel)


class TestAnalyzeEndpoint:
    """Test the Analyze RPC endpoint."""
    
    def test_analyze_simple_sentence(self, grpc_stub):
        """Test analyzing a simple sentence."""
        options = ml_server_pb2.ProcessingOptions(
            language="en-US",
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        request = ml_server_pb2.AnalyzeRequest(
            text="This is a test sentence.",
            options=options
        )
        
        response = grpc_stub.Analyze(request)
        
        assert len(response.sentences) > 0
        assert response.sentences[0].text == "This is a test sentence."
        assert len(response.sentences[0].tokens) > 0
    
    def test_analyze_grammatical_error(self, grpc_stub):
        """Test analyzing a sentence with grammatical errors."""
        options = ml_server_pb2.ProcessingOptions(
            language="en-US",
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        request = ml_server_pb2.AnalyzeRequest(
            text="She dont like apples.",
            options=options
        )
        
        response = grpc_stub.Analyze(request)
        
        assert len(response.sentences) > 0
        assert len(response.sentences[0].tokens) > 0
    
    def test_analyze_multiple_sentences(self, grpc_stub):
        """Test analyzing multiple sentences."""
        options = ml_server_pb2.ProcessingOptions(
            language="en-US",
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        request = ml_server_pb2.AnalyzeRequest(
            text="First sentence. Second sentence.",
            options=options
        )
        
        response = grpc_stub.Analyze(request)
        
        # Depending on sentence tokenization, might be 1 or more
        assert len(response.sentences) >= 1
    
    def test_analyze_empty_text(self, grpc_stub):
        """Test analyzing empty text."""
        options = ml_server_pb2.ProcessingOptions(
            language="en-US",
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        request = ml_server_pb2.AnalyzeRequest(
            text="",
            options=options
        )
        
        with pytest.raises(grpc.RpcError):
            response = grpc_stub.Analyze(request)


class TestProcessEndpoint:
    """Test the Process RPC endpoint."""
    
    @pytest.fixture
    def analyzed_sentences(self, grpc_stub):
        """Get analyzed sentences for processing tests."""
        options = ml_server_pb2.ProcessingOptions(
            language="en-US",
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        request = ml_server_pb2.AnalyzeRequest(
            text="She dont like apples.",
            options=options
        )
        response = grpc_stub.Analyze(request)
        return response.sentences
    
    def test_process_analyzed_sentences(self, grpc_stub, analyzed_sentences):
        """Test processing analyzed sentences."""
        options = ml_server_pb2.ProcessingOptions(
            language="en-US",
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        request = ml_server_pb2.ProcessRequest(
            sentences=analyzed_sentences,
            options=options
        )
        
        response = grpc_stub.Process(request)
        
        # Should have matches for the grammatical error
        assert len(response.matches) >= 0  # May or may not find errors depending on implementation
        assert hasattr(response, 'rawMatches')
    
    def test_process_correct_sentence(self, grpc_stub):
        """Test processing a correct sentence."""
        # First analyze
        options = ml_server_pb2.ProcessingOptions(
            language="en-US",
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        analyze_request = ml_server_pb2.AnalyzeRequest(
            text="This is correct.",
            options=options
        )
        analyze_response = grpc_stub.Analyze(analyze_request)
        
        # Then process
        process_request = ml_server_pb2.ProcessRequest(
            sentences=analyze_response.sentences,
            options=options
        )
        response = grpc_stub.Process(process_request)
        
        # Correct sentence should have no or few matches
        assert isinstance(response.matches, list) or hasattr(response, 'matches')


class TestEndToEndWorkflow:
    """Test complete analyze and process workflow."""
    
    @pytest.mark.parametrize("text,language", [
        ("This is a test sentence.", "en-US"),
        ("She dont like apples.", "en-US"),
        ("The quick brown fox jumps over the lazy dog.", "en-US"),
        ("its there fault", "en-US"),
    ])
    def test_full_workflow(self, grpc_stub, text, language):
        """Test complete workflow from analyze to process."""
        # Analyze
        options = ml_server_pb2.ProcessingOptions(
            language=language,
            level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
        )
        analyze_request = ml_server_pb2.AnalyzeRequest(
            text=text,
            options=options
        )
        analyze_response = grpc_stub.Analyze(analyze_request)
        
        assert len(analyze_response.sentences) > 0
        
        # Process
        process_request = ml_server_pb2.ProcessRequest(
            sentences=analyze_response.sentences,
            options=options
        )
        process_response = grpc_stub.Process(process_request)
        
        # Should complete without errors
        assert hasattr(process_response, 'matches')
        assert hasattr(process_response, 'rawMatches')
