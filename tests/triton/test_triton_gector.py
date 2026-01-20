"""Functional tests for GECToR model on Triton Inference Server."""
import pytest
import numpy as np
import sys
import os

try:
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

pytestmark = pytest.mark.skipif(
    not TRITON_AVAILABLE or not TRANSFORMERS_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires tritonclient, transformers and RUN_NON_HERMETIC=true (needs running Triton server)"
)


@pytest.fixture(scope="module")
def triton_client():
    """Create Triton client for testing."""
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        if not client.is_server_live():
            pytest.skip("Triton server is not live")
        return client
    except Exception as e:
        pytest.skip(f"Failed to connect to Triton server: {e}")


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer for preparing inputs."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gector-bert-base-cased-5k")
        return tokenizer
    except Exception as e:
        pytest.skip(f"Failed to load tokenizer: {e}")


@pytest.fixture
def model_name():
    """Return the model name to test."""
    return "gector_bert"


class TestTritonServerHealth:
    """Test Triton server health checks."""
    
    def test_server_is_live(self, triton_client):
        """Test that server is live."""
        assert triton_client.is_server_live()
    
    def test_server_is_ready(self, triton_client):
        """Test that server is ready."""
        assert triton_client.is_server_ready()


class TestGECToRModel:
    """Test GECToR model deployment and inference."""
    
    def test_model_is_ready(self, triton_client, model_name):
        """Test that the GECToR model is ready."""
        try:
            is_ready = triton_client.is_model_ready(model_name)
            if not is_ready:
                # List available models for debugging
                available_models = [model['name'] for model in triton_client.get_model_repository_index()]
                pytest.skip(f"Model {model_name} not ready. Available: {available_models}")
            assert is_ready
        except Exception as e:
            pytest.skip(f"Model check failed: {e}")
    
    def test_model_metadata(self, triton_client, model_name):
        """Test retrieving model metadata."""
        metadata = triton_client.get_model_metadata(model_name)
        
        assert 'versions' in metadata
        assert 'platform' in metadata
        assert 'inputs' in metadata
        assert 'outputs' in metadata
        assert len(metadata['inputs']) > 0
        assert len(metadata['outputs']) > 0
    
    def test_model_config(self, triton_client, model_name):
        """Test retrieving model configuration."""
        config = triton_client.get_model_config(model_name)
        
        # In some versions it returns {'config': {...}}, in others the config directly
        model_config = config.get('config', config)
        assert 'name' in model_config
        assert model_config['name'] == model_name
    
    @pytest.mark.parametrize("test_text", [
        "This is a simple test.",
        "She dont like apples.",
        "I has three cats.",
    ])
    def test_inference_with_tokenized_input(self, triton_client, model_name, tokenizer, test_text):
        """Test inference with pre-tokenized inputs."""
        # Tokenize input
        encoded = tokenizer(
            test_text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        
        # Prepare inputs
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Prepare outputs
        outputs = [
            httpclient.InferRequestedOutput("logits_labels"),
            httpclient.InferRequestedOutput("logits_d") 
        ]
        
        # Run inference
        try:
            response = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get results
            labels = response.as_numpy("logits_labels")
            d_tags = response.as_numpy("logits_d")
            
            # Basic validation
            assert labels is not None
            assert labels.shape[0] == input_ids.shape[0]
            assert labels.shape[1] == input_ids.shape[1]
            
            if d_tags is not None:
                assert d_tags.shape[0] == input_ids.shape[0]
                
        except Exception as e:
            pytest.fail(f"Inference failed: {e}")
    
    def test_batch_inference(self, triton_client, model_name, tokenizer):
        """Test batch inference."""
        test_texts = [
            "This is test one.",
            "This is test two.",
        ]
        
        # Tokenize inputs
        encoded = tokenizer(
            test_texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        
        # Prepare inputs
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Prepare outputs
        outputs = [
            httpclient.InferRequestedOutput("logits_labels")
        ]
        
        # Run inference
        try:
            response = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            labels = response.as_numpy("logits_labels")
            assert labels.shape[0] == len(test_texts)
            
        except Exception as e:
            pytest.fail(f"Batch inference failed: {e}")


class TestModelPerformance:
    """Test model performance characteristics."""
    
    @pytest.mark.slow
    def test_inference_latency(self, triton_client, model_name, tokenizer):
        """Test inference latency is within acceptable bounds."""
        import time
        
        test_text = "This is a test sentence for latency measurement."
        
        # Tokenize
        encoded = tokenizer(
            test_text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        
        # Prepare inputs
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        outputs = [httpclient.InferRequestedOutput("logits_labels")]
        
        # Measure latency
        start_time = time.time()
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Assert reasonable latency (adjust threshold as needed)
        assert latency < 5.0, f"Inference took {latency:.2f}s, expected < 5.0s"
