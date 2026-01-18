"""
Test script for Grammared Classifier Triton model with the exact example format.

This script tests the Triton server with the same input format as the local pipeline example:
pipeline.predict('I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>')
"""
import numpy as np
import json
import tritonclient.http as httpclient


def test_grammared_classifier_example(
    server_url="localhost:8000",
    model_name="grammared_classifier",
    model_version="1"
):
    """
    Test the grammared classifier model with the example input format.

    Args:
        server_url: Triton server URL
        model_name: Name of the model in Triton
        model_version: Version of the model
    """
    # Test text matching the pipeline example
    test_text = 'I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>'

    print("=" * 80)
    print("Testing Grammared Classifier on Triton Server")
    print("=" * 80)
    print(f"Server URL: {server_url}")
    print(f"Model: {model_name}, Version: {model_version}")
    print()

    # Initialize Triton client
    try:
        triton_client = httpclient.InferenceServerClient(url=server_url)
    except Exception as e:
        print(f"Error connecting to Triton server at {server_url}: {e}")
        print("\nMake sure the Triton server is running. You can start it with:")
        print("  docker-compose up triton-server")
        return

    # Check if server is live
    try:
        if not triton_client.is_server_live():
            print("Triton server is not live!")
            return
        print("✓ Triton server is live")
    except Exception as e:
        print(f"Error checking server status: {e}")
        return

    # Check if model is ready
    try:
        if not triton_client.is_model_ready(model_name, model_version):
            print(f"✗ Model {model_name} version {model_version} is not ready!")
            print("\nAvailable models:")
            try:
                metadata = triton_client.get_server_metadata()
                print(metadata)
            except:
                pass
            return
        print(f"✓ Model {model_name} is ready")
    except Exception as e:
        print(f"Error checking model readiness: {e}")
        return

    print()
    print("-" * 80)
    print("Running inference...")
    print("-" * 80)
    print()
    print(f"Input text:")
    print(f"  '{test_text}'")
    print()

    try:
        # Prepare text input as numpy array of strings
        # Note: Model has max_batch_size configured, so we need batch dimension
        # Shape should be [batch_size, num_elements] = [1, 1]
        text_data = np.array([[test_text]], dtype=object)

        # Prepare inputs
        inputs = [
            httpclient.InferInput("TEXT", [1, 1], "BYTES")
        ]
        inputs[0].set_data_from_numpy(text_data)

        # Prepare outputs
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT")
        ]

        # Send inference request
        response = triton_client.infer(
            model_name=model_name,
            model_version=model_version,
            inputs=inputs,
            outputs=outputs
        )

        # Get results
        output_data = response.as_numpy("OUTPUT")

        # Decode JSON string
        result_json = output_data[0]
        if isinstance(result_json, bytes):
            result_json = result_json.decode('utf-8')
        result = json.loads(result_json)

        print("✓ Inference successful!")
        print()
        print("Result:")
        print(f"  {result}")
        print()
        print("Details:")
        print(f"  Label: {result['label']}")
        print(f"  Score: {result['score']:.4f}")
        print()

        # Compare with expected output
        print("-" * 80)
        print("Expected output (from local pipeline):")
        print("  {'label': 'none', 'score': 0.9268353033472629}")
        print()
        print("Note: Scores may differ slightly due to:")
        print("  - Different hardware (CPU vs GPU)")
        print("  - Floating point precision")
        print("  - Model calibration")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return


def test_multiple_examples(
    server_url="localhost:8000",
    model_name="grammared_classifier",
    model_version="1"
):
    """
    Test with multiple grammar correction examples.

    Args:
        server_url: Triton server URL
        model_name: Name of the model in Triton
        model_version: Version of the model
    """
    # Multiple test examples
    test_texts = [
        'I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>',
        'She went to the [CLS]<|start_of_replace|>store.[SEP]shop<|end_of_replace|>',
        'The weather is [CLS]<|start_of_replace|>nice.[SEP]good<|end_of_replace|> today',
    ]

    print("=" * 80)
    print("Testing Multiple Examples on Triton Server")
    print("=" * 80)
    print(f"Server URL: {server_url}")
    print(f"Model: {model_name}, Version: {model_version}")
    print()

    # Initialize Triton client
    try:
        triton_client = httpclient.InferenceServerClient(url=server_url)
    except Exception as e:
        print(f"Error connecting to Triton server: {e}")
        return

    # Check if model is ready
    try:
        if not triton_client.is_model_ready(model_name, model_version):
            print(f"Model {model_name} is not ready!")
            return
        print(f"✓ Model is ready")
    except Exception as e:
        print(f"Error checking model: {e}")
        return

    print()
    print("-" * 80)

    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"  Input: '{text}'")

        try:
            # Prepare text input with batch dimension
            text_data = np.array([[text]], dtype=object)

            # Prepare inputs
            inputs = [
                httpclient.InferInput("TEXT", [1, 1], "BYTES")
            ]
            inputs[0].set_data_from_numpy(text_data)

            # Prepare outputs
            outputs = [
                httpclient.InferRequestedOutput("OUTPUT")
            ]

            # Send inference request
            response = triton_client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs
            )

            # Get results
            output_data = response.as_numpy("OUTPUT")
            result_json = output_data[0]
            if isinstance(result_json, bytes):
                result_json = result_json.decode('utf-8')
            result = json.loads(result_json)

            print(f"  Output: {result}")
            print(f"  → Label: {result['label']}, Score: {result['score']:.4f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Grammared Classifier Triton model with example format"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="grammared_classifier",
        help="Model name (default: grammared_classifier)"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="1",
        help="Model version (default: 1)"
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Test multiple examples"
    )

    args = parser.parse_args()

    if args.multiple:
        test_multiple_examples(
            server_url=args.server_url,
            model_name=args.model_name,
            model_version=args.model_version
        )
    else:
        test_grammared_classifier_example(
            server_url=args.server_url,
            model_name=args.model_name,
            model_version=args.model_version
        )
