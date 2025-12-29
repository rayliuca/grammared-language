#!/usr/bin/env python3
"""
Test script for GECToR model on Triton Inference Server.

This script validates that the model is properly deployed and can handle
inference requests with pre-tokenized inputs.
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

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed")
    print("Install with: pip install transformers")
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
    
    # Load tokenizer for preparing inputs
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gector-roberta-base-5k")
        print("✅ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False
    
    # Check server health
    print("\nChecking server health...")
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
            # Tokenize input text
            encoded = tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Extract input_ids and attention_mask as numpy arrays
            input_ids_np = encoded['input_ids'].astype(np.int64)
            attention_mask_np = encoded['attention_mask'].astype(np.int64)
            
            # Create input tensors for Triton
            input_ids_input = httpclient.InferInput("input_ids", input_ids_np.shape, "INT64")
            input_ids_input.set_data_from_numpy(input_ids_np)
            
            attention_mask_input = httpclient.InferInput("attention_mask", attention_mask_np.shape, "INT64")
            attention_mask_input.set_data_from_numpy(attention_mask_np)
            
            inputs = [input_ids_input, attention_mask_input]
            
            # Define outputs we expect from Triton
            outputs = [
                httpclient.InferRequestedOutput("logits")
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
            logits = response.as_numpy("logits")
            
            # Get predictions from logits
            predictions = np.argmax(logits, axis=-1)
            
            print(f"  ✅ Inference successful ({inference_time:.2f}ms)")
            print(f"  - Input shape: {input_ids_np.shape}")
            print(f"  - Logits shape: {logits.shape}")
            print(f"  - Logits sample: {logits[-1, :5, 0]}")  # Print sample logits
            print(f"  - Predictions shape: {predictions.shape}")
            
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
