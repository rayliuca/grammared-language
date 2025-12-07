"""
Test script for gRPC server endpoints.

Tests the ProcessingServer service with sample requests.
"""

import grpc
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
print(str(Path(__file__).parent))

from grpc_gen import ml_server_pb2, ml_server_pb2_grpc


def test_analyze_endpoint(channel, text: str, language: str = "en-US"):
    """Test the Analyze RPC endpoint."""
    stub = ml_server_pb2_grpc.ProcessingServerStub(channel)
    
    # Create request
    options = ml_server_pb2.ProcessingOptions(
        language=language,
        level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
    )
    request = ml_server_pb2.AnalyzeRequest(
        text=text,
        options=options
    )
    
    print(f"\n=== Testing Analyze Endpoint ===")
    print(f"Request text: {text}")
    print(f"Language: {language}")
    
    try:
        response = stub.Analyze(request)
        print(f"✓ Analyze RPC succeeded")
        print(f"  Response: {len(response.sentences)} sentence(s)")
        for i, sentence in enumerate(response.sentences):
            print(f"    Sentence {i}: {sentence.text}")
            print(f"      Tokens: {len(sentence.tokens)}")
        return response
    except grpc.RpcError as e:
        print(f"✗ Analyze RPC failed: {e.code()}")
        print(f"  Details: {e.details()}")
        return None


def test_process_endpoint(channel, analyzed_sentences):
    """Test the Process RPC endpoint."""
    stub = ml_server_pb2_grpc.ProcessingServerStub(channel)
    
    # Create request
    options = ml_server_pb2.ProcessingOptions(
        language="en-US",
        level=ml_server_pb2.ProcessingOptions.Level.defaultLevel
    )
    request = ml_server_pb2.ProcessRequest(
        sentences=analyzed_sentences,
        options=options
    )
    
    print(f"\n=== Testing Process Endpoint ===")
    print(f"Processing {len(analyzed_sentences)} sentence(s)")
    
    try:
        response = stub.Process(request)
        print(f"✓ Process RPC succeeded")
        print(f"  Raw matches: {len(response.rawMatches)}")
        print(f"  Processed matches: {len(response.matches)}")
        return response
    except grpc.RpcError as e:
        print(f"✗ Process RPC failed: {e.code()}")
        print(f"  Details: {e.details()}")
        return None


def main():
    """Main test function."""
    # Server configuration
    server_address = "localhost:50051"
    
    print(f"Connecting to gRPC server at {server_address}...")
    
    try:
        # Create channel
        channel = grpc.aio.secure_channel(
            server_address,
            grpc.ssl_channel_credentials()
        ) if False else grpc.aio.insecure_channel(server_address)
        
        # Use insecure channel for testing
        channel = grpc.insecure_channel(server_address)
        print(f"✓ Connected to server at {server_address}")
        
        # Test samples
        test_texts = [
            "This is a test sentence.",
            "She dont like apples.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        # Test Analyze endpoint
        for text in test_texts:
            response = test_analyze_endpoint(channel, text)
            if response:
                # Test Process endpoint with the response
                test_process_endpoint(channel, response.sentences)
        
        print("\n=== Test Summary ===")
        print("✓ All tests completed")
        
    except grpc.RpcError as e:
        print(f"✗ Connection failed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'channel' in locals():
            channel.close()


if __name__ == "__main__":
    main()
