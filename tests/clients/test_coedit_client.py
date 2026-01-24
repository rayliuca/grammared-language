import pytest
import os
from unittest.mock import Mock, patch
import numpy as np

from grammared_language.clients.coedit_client import CoEditClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

try:
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except Exception:
    grpcclient = None
    TRITON_AVAILABLE = False


class TestCoEditClient:
    """Unit tests for CoEditClient."""
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_initialization_default(self, mock_grpcclient):
        """Test default initialization."""
        mock_client_class = Mock()
        mock_grpcclient.InferenceServerClient = mock_client_class
        
        client = CoEditClient()
        
        assert client.model_name == "coedit_large"
        assert client.triton_model_version == "1"
        assert client.prompt_template == "Fix grammatical errors: {{ text }}"
        mock_client_class.assert_called_once_with(url="localhost:8001")
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_initialization_custom_model(self, mock_grpcclient):
        """Test initialization with custom model name."""
        mock_grpcclient.InferenceServerClient = Mock()
        
        client = CoEditClient(
            model_name="coedit_xl",
            triton_host="custom-host",
            triton_port=9000
        )
        
        assert client.model_name == "coedit_xl"
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_initialization_custom_prompt(self, mock_grpcclient):
        """Test custom prompt template."""
        mock_grpcclient.InferenceServerClient = Mock()
        
        custom = "Custom instruction: {{ text }}"
        client = CoEditClient(prompt_template=custom)
        
        assert client.prompt_template == custom
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_preprocess_applies_template(self, mock_grpcclient):
        """Test preprocessing applies prompt template."""
        mock_grpcclient.InferenceServerClient = Mock()
        
        client = CoEditClient()
        text = "She go to store."
        processed = client._preprocess(text)
        
        assert processed == "Fix grammatical errors: She go to store."
    
    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_predict_flow(self, mock_grpcclient):
        """Test full prediction flow."""
        # Setup mocks
        mock_client = Mock()
        mock_grpcclient.InferenceServerClient = Mock(return_value=mock_client)
        mock_grpcclient.InferInput = Mock()
        mock_grpcclient.InferRequestedOutput = Mock()
        
        # Mock response
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([b"She went to the store."], dtype=object)
        mock_client.infer.return_value = mock_response
        
        client = CoEditClient()
        result = client.predict("She go to store.")
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"


@pytest.mark.skipif(
    not TRITON_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires tritonclient and RUN_NON_HERMETIC=true (needs running Triton server with coedit_large model)"
)
class TestCoEditClientFunctional:
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
    
    def test_functional_grammar_correction(self, triton_ready):
        """Test grammar correction task."""
        client = CoEditClient()
        
        test_text = "She go to the store yesterday."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        
        print(f"\nGrammar Correction:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
        for match in result.matches:
            orig = test_text[match.offset:match.offset + match.length]
            suggestions = [r.replacement for r in match.suggestedReplacements] if match.suggestedReplacements else []
            print(f"    '{orig}' → {suggestions}")
    
    def test_functional_custom_prompt(self, triton_ready):
        """Test custom prompt improvement."""
        client = CoEditClient(prompt_template="Make the text more fluent: {{ text }}")
        
        test_text = "The thing is that I kind of maybe want to go there sometime."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nCustom Prompt (Fluency):")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
