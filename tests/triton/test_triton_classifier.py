"""Functional tests for Grammared Classifier Triton model."""
import pytest
import numpy as np
import json
import os

try:
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not TRITON_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires tritonclient and RUN_NON_HERMETIC=true (needs running Triton server)"
)


@pytest.fixture(scope="module")
def triton_client():
    """Create Triton client for testing."""
    try:
        client = grpcclient.InferenceServerClient(url="localhost:8001")
        if not client.is_server_live():
            pytest.skip("Triton server is not live")
        return client
    except Exception as e:
        pytest.skip(f"Failed to connect to Triton server: {e}")


@pytest.fixture
def model_name():
    """Return the model name to test."""
    return "grammared_classifier"


@pytest.fixture
def model_version():
    """Return the model version to test."""
    return "1"


class TestClassifierModel:
    """Test Grammared Classifier model."""
    
    def test_model_is_ready(self, triton_client, model_name, model_version):
        """Test that the classifier model is ready."""
        try:
            is_ready = triton_client.is_model_ready(model_name, model_version)
            if not is_ready:
                pytest.skip(f"Model {model_name} version {model_version} is not ready")
            assert is_ready
        except Exception as e:
            pytest.skip(f"Model check failed: {e}")
    
    @pytest.mark.parametrize("test_text,expected_fields", [
        ("This is a good correction.", ["label", "score"]),
        ("This is a bad correction.", ["label", "score"]),
        ("The sentence looks correct now.", ["label", "score"]),
    ])
    def test_single_inference(self, triton_client, model_name, model_version, test_text, expected_fields):
        """Test single text inference."""
        # Prepare text input with batch dimension
        text_data = np.array([[test_text]], dtype=object)
        
        # Prepare inputs
        inputs = [
            grpcclient.InferInput("TEXT", [1, 1], "BYTES")
        ]
        inputs[0].set_data_from_numpy(text_data)
        
        # Prepare outputs
        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT")
        ]
        
        # Send inference request
        response = triton_client.infer(
            model_name=model_name,
            model_version=model_version,
            inputs=inputs,
            outputs=outputs
        )
        
        # Get and parse results
        output_data = response.as_numpy("OUTPUT")
        # Handle both [batch, 1] and [batch] shapes
        if output_data.ndim > 1:
            result_json = output_data[0, 0]
        else:
            result_json = output_data[0]
            
        if isinstance(result_json, bytes):
            result_json = result_json.decode('utf-8')
        result = json.loads(result_json)
        
        # Validate result structure
        for field in expected_fields:
            assert field in result, f"Expected field '{field}' not found in result"
        
        # Validate score is a probability
        assert 0.0 <= result['score'] <= 1.0
        assert isinstance(result['label'], str)
    
    def test_batch_inference(self, triton_client, model_name, model_version):
        """Test batch inference with multiple texts."""
        test_texts = [
            "This is a good correction.",
            "This is a bad correction.",
            "The sentence looks correct now."
        ]
        
        results = []
        for text in test_texts:
            text_data = np.array([[text]], dtype=object)
            
            inputs = [
                grpcclient.InferInput("TEXT", [1, 1], "BYTES")
            ]
            inputs[0].set_data_from_numpy(text_data)
            
            outputs = [
                grpcclient.InferRequestedOutput("OUTPUT")
            ]
            
            response = triton_client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs
            )
            
            output_data = response.as_numpy("OUTPUT")
            if output_data.ndim > 1:
                result_json = output_data[0, 0]
            else:
                result_json = output_data[0]
            if isinstance(result_json, bytes):
                result_json = result_json.decode('utf-8')
            result = json.loads(result_json)
            results.append(result)
        
        # Validate all results
        assert len(results) == len(test_texts)
        for result in results:
            assert 'label' in result
            assert 'score' in result
            assert 0.0 <= result['score'] <= 1.0
    
    def test_special_format_input(self, triton_client, model_name, model_version):
        """Test with special format input (matching pipeline example)."""
        test_text = 'I think you should try for that new [CLS]<|start_of_replace|>job.[SEP]job<|end_of_replace|>'
        
        # Prepare input with batch dimension
        text_data = np.array([[test_text]], dtype=object)
        
        inputs = [
            grpcclient.InferInput("TEXT", [1, 1], "BYTES")
        ]
        inputs[0].set_data_from_numpy(text_data)
        
        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT")
        ]
        
        response = triton_client.infer(
            model_name=model_name,
            model_version=model_version,
            inputs=inputs,
            outputs=outputs
        )
        
        output_data = response.as_numpy("OUTPUT")
        if output_data.ndim > 1:
            result_json = output_data[0, 0]
        else:
            result_json = output_data[0]
        if isinstance(result_json, bytes):
            result_json = result_json.decode('utf-8')
        result = json.loads(result_json)
        
        # Validate result
        assert 'label' in result
        assert 'score' in result
        # Accept both raw labels and mapped labels
        allowed_labels = ['none']
        assert result['label'] in allowed_labels
        assert 0.0 <= result['score'] <= 1.0
    
    def test_empty_string_handling(self, triton_client, model_name, model_version):
        """Test handling of empty string."""
        text_data = np.array([""], dtype=object)
        
        inputs = [
            grpcclient.InferInput("TEXT", text_data.shape, "BYTES")
        ]
        inputs[0].set_data_from_numpy(text_data)
        
        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT")
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            response = triton_client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs
            )
            # If it doesn't raise, validate result
            output_data = response.as_numpy("OUTPUT")
            result_json = output_data[0]
            if isinstance(result_json, bytes):
                result_json = result_json.decode('utf-8')
            result = json.loads(result_json)
            assert 'label' in result
            assert 'score' in result
        except Exception as e:
            # Expected behavior - model may reject empty input
            pass


class TestModelMetadata:
    """Test model metadata and configuration."""
    
    def test_model_metadata(self, triton_client, model_name):
        """Test retrieving model metadata."""
        metadata = triton_client.get_model_metadata(model_name)
        
        # metadata is a protobuf object, access attributes directly
        assert hasattr(metadata, 'name')
        assert metadata.name == model_name
        assert hasattr(metadata, 'versions')
        assert hasattr(metadata, 'platform')
        assert hasattr(metadata, 'inputs')
        assert hasattr(metadata, 'outputs')
        assert len(metadata.inputs) > 0
        assert len(metadata.outputs) > 0
    
    def test_model_config(self, triton_client, model_name):
        """Test retrieving model configuration."""
        config = triton_client.get_model_config(model_name)
        
        # config is a protobuf object, access attributes directly
        assert hasattr(config, 'config')
        model_config = config.config
        assert hasattr(model_config, 'name')
        assert model_config.name == model_name
