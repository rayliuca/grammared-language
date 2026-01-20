import pytest
import requests
import os
import json

# Check if we should run non-hermetic tests
RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not RUN_NON_HERMETIC,
    reason="RUN_NON_HERMETIC=true is not set"
)

@pytest.fixture
def triton_url():
    return f"http://{os.getenv('TRITON_SERVER_URL', 'localhost:8000')}"

@pytest.fixture
def model_name():
    return "grammared_classifier"

@pytest.mark.functional
def test_triton_health_rest(triton_url):
    """Test Triton health endpoints via REST."""
    response = requests.get(f"{triton_url}/v2/health/live")
    assert response.status_code == 200
    
    response = requests.get(f"{triton_url}/v2/health/ready")
    assert response.status_code == 200

@pytest.mark.functional
def test_classifier_rest_inference(triton_url, model_name):
    """Test Grammared Classifier inference via REST API (like the curl script)."""
    inference_url = f"{triton_url}/v2/models/{model_name}/versions/1/infer"
    
    test_text = 'I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>'
    
    payload = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [test_text]
            }
        ]
    }
    
    response = requests.post(inference_url, json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert "outputs" in result
    
    # Extract the encoded JSON string from the output
    output_data = result["outputs"][0]["data"][0]
    prediction = json.loads(output_data)
    
    assert "label" in prediction
    assert "score" in prediction
    assert isinstance(prediction["score"], float)
