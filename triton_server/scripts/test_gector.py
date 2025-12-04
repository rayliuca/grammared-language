#!/usr/bin/env python3
"""
Test script for GECToR model on Triton Inference Server.

This script validates that the model is properly deployed and can handle
inference requests.
"""

import argparse
import json
import sys
import time
import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError:
    print("Error: tritonclient not installed")
    print("Install with: pip install tritonclient[http]")
    sys.exit(1)


def test_model(triton_url="localhost:8000", model_name="gector_roberta"):
    """
    Test the GECToR model deployment.
    
    Args:
        triton_url: Triton server URL
        model_name: Name of the model to test
    """
    print("=" * 60)
    print("GECToR Model Test")
    print("=" * 60)
    print(f"Triton Server: {triton_url}")
    print(f"Model Name: {model_name}")
    print()
    
    # Create client
    try:
        client = httpclient.InferenceServerClient(url=triton_url)
    except Exception as e:
        print(f"❌ Failed to connect to Triton server: {e}")
        return False
    
    # Check server health
    print("Checking server health...")
    try:
        if not client.is_server_live():
            print("❌ Server is not live")
            return False
        print("✅ Server is live")
        
        if not client.is_server_ready():
            print("❌ Server is not ready")
            return False
        print("✅ Server is ready")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Check model status
    print(f"\nChecking model '{model_name}' status...")
    try:
        if not client.is_model_ready(model_name):
            print(f"❌ Model '{model_name}' is not ready")
            print("\nAvailable models:")
            model_repository = client.get_model_repository_index()
            for model in model_repository.models:
                print(f"  - {model['name']}")
            return False
        print(f"✅ Model '{model_name}' is ready")
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False
    
    # Get model metadata
    print(f"\nGetting model metadata...")
    try:
        metadata = client.get_model_metadata(model_name)
        print(f"✅ Model metadata retrieved")
        print(f"  - Version: {metadata['versions']}")
        print(f"  - Platform: {metadata['platform']}")
        print(f"  - Inputs: {len(metadata['inputs'])}")
        print(f"  - Outputs: {len(metadata['outputs'])}")
    except Exception as e:
        print(f"❌ Failed to get metadata: {e}")
        return False
    
    # Test inference with sample text
    print("\nTesting inference...")
    test_cases = [
        "I has a grammar error in this sentence.",
        "She don't like to go there.",
        "The cats is sleeping on the couch.",
        "This sentence are correct.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest case {i}: \"{text}\"")
        
        try:
            # Prepare input
            input_data = np.array([text], dtype=object)
            
            # Create input tensor
            inputs = [
                httpclient.InferInput("INPUT_TEXT", input_data.shape, "BYTES")
            ]
            inputs[0].set_data_from_numpy(input_data)
            
            # Create output request
            outputs = [
                httpclient.InferRequestedOutput("CORRECTIONS"),
                httpclient.InferRequestedOutput("LABELS"),
                httpclient.InferRequestedOutput("CONFIDENCES")
            ]
            
            # Send request and measure time
            start_time = time.time()
            response = client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get results
            corrections = response.as_numpy("CORRECTIONS")[0]
            labels = response.as_numpy("LABELS")[0]
            confidences = response.as_numpy("CONFIDENCES")[0]
            
            # Parse JSON results
            if isinstance(corrections, bytes):
                corrections = corrections.decode('utf-8')
            if isinstance(labels, bytes):
                labels = labels.decode('utf-8')
            if isinstance(confidences, bytes):
                confidences = confidences.decode('utf-8')
            
            corrections_data = json.loads(corrections)
            labels_data = json.loads(labels)
            
            print(f"  ✅ Inference successful ({inference_time:.2f}ms)")
            print(f"  - Corrections found: {len(corrections_data)}")
            
            if corrections_data:
                print(f"  - Sample corrections:")
                for corr in corrections_data[:3]:  # Show first 3
                    print(f"    * Token: '{corr['token']}' -> Label: '{corr['label']}' "
                          f"(confidence: {corr['confidence']:.2f})")
            
        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test GECToR model on Triton Inference Server"
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)"
    )
    parser.add_argument(
        "--model",
        default="gector_roberta",
        help="Model name (default: gector_roberta)"
    )
    
    args = parser.parse_args()
    
    success = test_model(args.url, args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
