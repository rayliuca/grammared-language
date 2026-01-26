import pytest
import os
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from grammared_language.clients.text2text_base_client import Text2TextBaseClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

try:
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except Exception:
    grpcclient = None
    TRITON_AVAILABLE = False


class TestText2TextBaseClient:
    """Test suite for Text2TextBaseClient."""
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_initialization(self, mock_grpcclient):
        """Test client initialization with default parameters."""
        mock_client_class = Mock()
        mock_grpcclient.InferenceServerClient = mock_client_class
        
        client = Text2TextBaseClient(model_name="coedit_large")
        
        assert client.model_name == "coedit_large"
        assert client.triton_model_version == "1"
        assert client.input_name == "text_input"
        assert client.output_name == "text_output"
        assert client.prompt_template is None
        mock_client_class.assert_called_once_with(url="localhost:8001")
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_initialization_custom_params(self, mock_grpcclient):
        """Test client initialization with custom parameters."""
        mock_client_class = Mock()
        mock_grpcclient.InferenceServerClient = mock_client_class
        
        client = Text2TextBaseClient(
            model_name="my_model",
            triton_host="custom-host",
            triton_port=9000,
            triton_model_version="2",
            input_name="custom_input",
            output_name="custom_output",
            prompt_template="Fix grammar: {{ text }}"
        )
        
        assert client.model_name == "my_model"
        assert client.triton_model_version == "2"
        assert client.input_name == "custom_input"
        assert client.output_name == "custom_output"
        assert client.prompt_template == "Fix grammar: {{ text }}"
        mock_client_class.assert_called_once_with(url="custom-host:9000")
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_preprocess_without_template(self, mock_grpcclient):
        """Test preprocessing without prompt template."""
        mock_grpcclient.InferenceServerClient = Mock()
        
        client = Text2TextBaseClient(model_name="test_model")
        text = "This are a test."
        result = client._preprocess(text)
        
        assert result == text
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_preprocess_with_template(self, mock_grpcclient):
        """Test preprocessing with chat template."""
        mock_grpcclient.InferenceServerClient = Mock()
        
        client = Text2TextBaseClient(
            model_name="test_model",
            prompt_template="Fix grammar: {{ text }}"
        )
        text = "This are a test."
        result = client._preprocess(text)
        
        assert result == "Fix grammar: This are a test."
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_predict_basic(self, mock_grpcclient):
        """Test basic prediction with text output."""
        # Setup mocks
        mock_client = Mock()
        mock_grpcclient.InferenceServerClient = Mock(return_value=mock_client)
        mock_grpcclient.InferInput = Mock()
        mock_grpcclient.InferRequestedOutput = Mock()
        
        # Mock response with batched output shape [1, 1]
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([[b"This is a test."]], dtype=object)
        mock_client.infer.return_value = mock_response
        
        client = Text2TextBaseClient(model_name="coedit_large")
        result = client._predict("This are a test.")
        
        assert result == "This is a test."
        assert mock_client.infer.called
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_predict_string_output(self, mock_grpcclient):
        """Test prediction with string output (not bytes)."""
        # Setup mocks
        mock_client = Mock()
        mock_grpcclient.InferenceServerClient = Mock(return_value=mock_client)
        mock_grpcclient.InferInput = Mock()
        mock_grpcclient.InferRequestedOutput = Mock()
        
        # Mock response with string in batched shape [1, 1]
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([["This is a test."]], dtype=object)
        mock_client.infer.return_value = mock_response
        
        client = Text2TextBaseClient(model_name="coedit_large")
        result = client._predict("This are a test.")
        
        assert result == "This is a test."
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_predict_empty_output(self, mock_grpcclient):
        """Test prediction with empty output."""
        # Setup mocks
        mock_client = Mock()
        mock_grpcclient.InferenceServerClient = Mock(return_value=mock_client)
        mock_grpcclient.InferInput = Mock()
        mock_grpcclient.InferRequestedOutput = Mock()
        
        # Mock response with empty output
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([], dtype=object)
        mock_client.infer.return_value = mock_response
        
        client = Text2TextBaseClient(model_name="coedit_large")
        input_text = "This are a test."
        result = client._predict(input_text)
        
        # Should return original text if no output
        assert result == input_text
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_predict_full_flow(self, mock_grpcclient):
        """Test full prediction flow with LanguageToolRemoteResult."""
        # Setup mocks
        mock_client = Mock()
        mock_grpcclient.InferenceServerClient = Mock(return_value=mock_client)
        mock_grpcclient.InferInput = Mock()
        mock_grpcclient.InferRequestedOutput = Mock()
        
        # Mock response with batched output shape [1, 1]
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([[b"This is a test."]], dtype=object)
        mock_client.infer.return_value = mock_response
        
        client = Text2TextBaseClient(model_name="coedit_large")
        result = client.predict("This are a test.")
        
        # Should return LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        assert len(result.matches) >= 0  # May or may not have matches
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_call_method(self, mock_grpcclient):
        """Test __call__ method works as alias for predict."""
        # Setup mocks
        mock_client = Mock()
        mock_grpcclient.InferenceServerClient = Mock(return_value=mock_client)
        mock_grpcclient.InferInput = Mock()
        mock_grpcclient.InferRequestedOutput = Mock()
        
        # Mock response with batched output shape [1, 1]
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([[b"This is a test."]], dtype=object)
        mock_client.infer.return_value = mock_response
        
        client = Text2TextBaseClient(model_name="coedit_large")
        result = client("This are a test.")
        
        # Should return LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)


@pytest.mark.skipif(
    not TRITON_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires tritonclient and RUN_NON_HERMETIC=true (needs running Triton server with coedit_large model)"
)
class TestText2TextBaseClientFunctional:
    """Functional tests requiring a running Triton server with coedit_large model."""
    
    @pytest.fixture(scope="class")
    def triton_ready(self):
        """Check if Triton server is live and has the coedit_large model."""
        if not TRITON_AVAILABLE:
            pytest.skip("tritonclient not available")
        
        try:
            client = grpcclient.InferenceServerClient(url="localhost:8001")
            if not client.is_server_live():
                pytest.skip("Triton server is not live")
            
            # Check if model is available
            model_ready = client.is_model_ready("coedit_large", "1")
            if not model_ready:
                pytest.skip("coedit_large model is not ready on Triton server")
        except Exception as e:
            pytest.skip(f"Cannot connect to Triton server: {e}")
        
        return True
    
    def test_functional_prediction_with_template(self, triton_ready):
        """Test actual prediction against running Triton server with chat template."""
        client = Text2TextBaseClient(
            model_name="coedit_large",
            triton_host="localhost",
            triton_port=8001,
            prompt_template="Fix grammar: {{ text }}"
        )
        
        test_text = "This are a test."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        assert isinstance(result.matches, list)
        
        # Log output for debugging
        print(f"\nOriginal: {test_text}")
        print(f"Matches: {result.matches}")
    
    def test_functional_without_template(self, triton_ready):
        """Test prediction without prompt template."""
        client = Text2TextBaseClient(
            model_name="coedit_large",
            triton_host="localhost",
            triton_port=8001
        )
        
        test_text = "She go to school yesterday."
        result = client(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        assert isinstance(result.matches, list)
        
        print(f"\nOriginal: {test_text}")
        print(f"Matches: {result.matches}")
    
    def test_functional_multiple_errors(self, triton_ready):
        """Test prediction on text with multiple grammar errors."""
        client = Text2TextBaseClient(
            model_name="coedit_large",
            triton_host="localhost",
            triton_port=8001,
            prompt_template="Fix grammar: {{ text }}"
        )
        
        test_text = "She go to the store and buy some milk yesterday."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        
        print(f"\nOriginal: {test_text}")
        print(f"Matches found: {len(result.matches)}")
        for i, match in enumerate(result.matches):
            print(f"  Match {i+1}: offset={match.offset}, length={match.length}")
            if match.suggestedReplacements:
                print(f"    Suggestions: {[r.replacement for r in match.suggestedReplacements]}")
    
    def test_functional_correct_text(self, triton_ready):
        """Test prediction on already correct text."""
        client = Text2TextBaseClient(
            model_name="coedit_large",
            triton_host="localhost",
            triton_port=8001,
            prompt_template="Fix grammar: {{ text }}"
        )
        
        test_text = "She went to the store yesterday."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        
        print(f"\nOriginal: {test_text}")
        print(f"Matches: {result.matches}")
        # Correct text may have 0 matches
    
    def test_functional_call_method(self, triton_ready):
        """Test __call__ method works properly in functional test."""
        client = Text2TextBaseClient(
            model_name="coedit_large",
            triton_host="localhost",
            triton_port=8001,
            prompt_template="Fix grammar: {{ text }}"
        )
        
        test_text = "I has a car."
        result = client(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        print(f"\nOriginal: {test_text}")
        print(f"Matches: {result.matches}")
    
    def test_functional_custom_prompt(self, triton_ready):
        """Test with different chat template."""
        client = Text2TextBaseClient(
            model_name="coedit_large",
            triton_host="localhost",
            triton_port=8001,
            prompt_template="Grammar: {{ text }}"
        )
        
        test_text = "They was happy."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        print(f"\nOriginal: {test_text}")
        print(f"Matches: {result.matches}")
