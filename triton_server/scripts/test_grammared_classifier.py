"""
Test script for Grammared Classifier Triton model.

This script demonstrates how to send requests to the Triton server
for the grammared_classifier model using text strings as input.
"""
import numpy as np
import json
import tritonclient.http as httpclient


def test_grammared_classifier(
    server_url="localhost:8000",
    model_name="grammared_classifier",
    model_version="1"
):
    """
    Test the grammared classifier model on Triton server.

    Args:
        server_url: Triton server URL
        model_name: Name of the model in Triton
        model_version: Version of the model
    """
    # Sample texts for classification
    test_texts = [
        "This is a good correction.",
        "This is a bad correction.",
        "The sentence looks correct now."
    ]

    print(f"Testing Grammared Classifier model on {server_url}")
    print(f"Model: {model_name}, Version: {model_version}\n")

    # Initialize Triton client
    triton_client = httpclient.InferenceServerClient(url=server_url)

    # Check if model is ready
    if not triton_client.is_model_ready(model_name, model_version):
        print(f"Model {model_name} is not ready!")
        return

    print("Model is ready. Running inference...\n")

    for text in test_texts:
        print(f"Input text: {text}")

        # Prepare text input as numpy array of strings
        text_data = np.array([text], dtype=object)

        # Prepare inputs
        inputs = [
            httpclient.InferInput("TEXT", text_data.shape, "BYTES")
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

        print(f"  Result: {result}")
        print(f"  Label: {result['label']}")
        print(f"  Score: {result['score']:.4f}")
        print()


def test_grammared_classifier_batch(
    server_url="localhost:8000",
    model_name="grammared_classifier",
    model_version="1"
):
    """
    Test the grammared classifier model with batch input.

    Args:
        server_url: Triton server URL
        model_name: Name of the model in Triton
        model_version: Version of the model
    """
    # Sample texts for classification
    test_texts = [
        "This is a good correction.",
        "This is a bad correction.",
        "The sentence looks correct now.",
        "Another example sentence here."
    ]

    print(f"Testing Grammared Classifier model with batch input on {server_url}")
    print(f"Model: {model_name}, Version: {model_version}\n")

    # Initialize Triton client
    triton_client = httpclient.InferenceServerClient(url=server_url)

    # Check if model is ready
    if not triton_client.is_model_ready(model_name, model_version):
        print(f"Model {model_name} is not ready!")
        return

    print("Model is ready. Running batch inference...\n")

    # Prepare batch of texts
    text_data = np.array([[text] for text in test_texts], dtype=object)

    print(f"Batch size: {len(test_texts)}")

    # Prepare inputs
    inputs = [
        httpclient.InferInput("TEXT", [len(test_texts), 1], "BYTES")
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

    # Decode JSON strings
    results = []
    for result_json in output_data:
        if isinstance(result_json, bytes):
            result_json = result_json.decode('utf-8')
        elif isinstance(result_json, np.ndarray):
            result_json = result_json[0]
            if isinstance(result_json, bytes):
                result_json = result_json.decode('utf-8')
        results.append(json.loads(result_json))

    print(f"Batch results: {results}\n")

    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"Input {i+1}: {text}")
        print(f"  Label: {result['label']}")
        print(f"  Score: {result['score']:.4f}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Grammared Classifier Triton model")
    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Triton server URL"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="grammared_classifier",
        help="Model name"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="1",
        help="Model version"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Test batch inference"
    )

    args = parser.parse_args()

    if args.batch:
        test_grammared_classifier_batch(
            server_url=args.server_url,
            model_name=args.model_name,
            model_version=args.model_version
        )
    else:
        test_grammared_classifier(
            server_url=args.server_url,
            model_name=args.model_name,
            model_version=args.model_version
        )
