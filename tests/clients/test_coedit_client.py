import pytest
import os
from unittest.mock import Mock, patch
import numpy as np

from grammared_language.clients.coedit_client import CoEditClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

try:
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except Exception:
    httpclient = None
    TRITON_AVAILABLE = False


class TestCoEditClient:
    """Unit tests for CoEditClient."""
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_initialization_default(self, mock_httpclient):
        """Test default initialization with grammar task."""
        mock_client_class = Mock()
        mock_httpclient.InferenceServerClient = mock_client_class
        
        client = CoEditClient()
        
        assert client.model_name == "coedit_large"
        assert client.triton_model_version == "1"
        assert client.task == "grammar"
        assert client.chat_template == "Fix grammatical errors: {text}"
        mock_client_class.assert_called_once_with(url="localhost:8000")
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_initialization_custom_model(self, mock_httpclient):
        """Test initialization with custom model name."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(
            model_name="coedit_xl",
            triton_host="custom-host",
            triton_port=9000
        )
        
        assert client.model_name == "coedit_xl"
        assert client.task == "grammar"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_fluency(self, mock_httpclient):
        """Test fluency task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="fluency")
        
        assert client.task == "fluency"
        assert client.chat_template == "Make the text more fluent: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_coherence(self, mock_httpclient):
        """Test coherence task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="coherence")
        
        assert client.task == "coherence"
        assert client.chat_template == "Make the text more coherent: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_clarity(self, mock_httpclient):
        """Test clarity task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="clarity")
        
        assert client.task == "clarity"
        assert client.chat_template == "Make the text more clear: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_paraphrase(self, mock_httpclient):
        """Test paraphrase task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="paraphrase")
        
        assert client.task == "paraphrase"
        assert client.chat_template == "Paraphrase: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_neutralize(self, mock_httpclient):
        """Test neutralize task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="neutralize")
        
        assert client.task == "neutralize"
        assert client.chat_template == "Make the text more neutral: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_simplify(self, mock_httpclient):
        """Test simplify task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="simplify")
        
        assert client.task == "simplify"
        assert client.chat_template == "Simplify: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_formalize(self, mock_httpclient):
        """Test formalize task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="formalize")
        
        assert client.task == "formalize"
        assert client.chat_template == "Make the text more formal: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_update(self, mock_httpclient):
        """Test update task prompt."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="update")
        
        assert client.task == "update"
        assert client.chat_template == "Update: {text}"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_task_none(self, mock_httpclient):
        """Test no task (raw text input)."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task=None)
        
        assert client.task is None
        assert client.chat_template is None
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_custom_prompt(self, mock_httpclient):
        """Test custom chat template overrides task."""
        mock_httpclient.InferenceServerClient = Mock()
        
        custom = "Custom instruction: {text}"
        client = CoEditClient(task="grammar", chat_template=custom)
        
        assert client.chat_template == custom
        assert client.task == "grammar"  # Task is still stored
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_invalid_task(self, mock_httpclient):
        """Test invalid task raises ValueError."""
        mock_httpclient.InferenceServerClient = Mock()
        
        with pytest.raises(ValueError) as exc_info:
            CoEditClient(task="invalid_task")
        
        assert "Invalid task" in str(exc_info.value)
        assert "invalid_task" in str(exc_info.value)
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_preprocess_applies_template(self, mock_httpclient):
        """Test preprocessing applies chat template."""
        mock_httpclient.InferenceServerClient = Mock()
        
        client = CoEditClient(task="grammar")
        text = "She go to store."
        processed = client._preprocess(text)
        
        assert processed == "Fix grammatical errors: She go to store."
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_predict_flow(self, mock_httpclient):
        """Test full prediction flow."""
        # Setup mocks
        mock_client = Mock()
        mock_httpclient.InferenceServerClient = Mock(return_value=mock_client)
        mock_httpclient.InferInput = Mock()
        mock_httpclient.InferRequestedOutput = Mock()
        
        # Mock response
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([b"She went to the store."], dtype=object)
        mock_client.infer.return_value = mock_response
        
        client = CoEditClient(task="grammar")
        result = client.predict("She go to store.")
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
    
    @patch('grammared_language.clients.text2text_base_client.httpclient')
    def test_all_tasks_valid(self, mock_httpclient):
        """Test that all documented tasks are valid."""
        mock_httpclient.InferenceServerClient = Mock()
        
        tasks = [
            "grammar", "fluency", "coherence", "clarity",
            "paraphrase", "neutralize", "simplify", "formalize", "update"
        ]
        
        for task in tasks:
            client = CoEditClient(task=task)
            assert client.task == task
            assert client.chat_template is not None
            assert "{text}" in client.chat_template


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
            client = httpclient.InferenceServerClient(url="localhost:8000")
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
        client = CoEditClient(task="grammar")
        
        test_text = "She go to the store yesterday."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        
        print(f"\nGrammar Task:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
        for match in result.matches:
            orig = test_text[match.offset:match.offset + match.length]
            suggestions = [r.replacement for r in match.suggestedReplacements] if match.suggestedReplacements else []
            print(f"    '{orig}' → {suggestions}")
    
    def test_functional_fluency(self, triton_ready):
        """Test fluency improvement task."""
        client = CoEditClient(task="fluency")
        
        test_text = "The thing is that I kind of maybe want to go there sometime."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nFluency Task:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
    
    def test_functional_clarity(self, triton_ready):
        """Test clarity improvement task."""
        client = CoEditClient(task="clarity")
        
        test_text = "It's not uncommon to find people who don't dislike it."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nClarity Task:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
    
    def test_functional_paraphrase(self, triton_ready):
        """Test paraphrase task."""
        client = CoEditClient(task="paraphrase")
        
        test_text = "The cat sat on the mat."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nParaphrase Task:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
    
    def test_functional_formalize(self, triton_ready):
        """Test formalization task."""
        client = CoEditClient(task="formalize")
        
        test_text = "gonna check it out later"
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nFormalize Task:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
    
    def test_functional_simplify(self, triton_ready):
        """Test simplification task."""
        client = CoEditClient(task="simplify")
        
        test_text = "The implementation of the aforementioned methodology necessitates careful consideration."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nSimplify Task:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
    
    def test_functional_call_method(self, triton_ready):
        """Test __call__ method."""
        client = CoEditClient(task="grammar")
        
        test_text = "I has a car."
        result = client(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nCall Method:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
    
    def test_functional_no_template(self, triton_ready):
        """Test with no task template."""
        client = CoEditClient(task=None)
        
        test_text = "She go to school."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nNo Template:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
    
    def test_functional_custom_prompt(self, triton_ready):
        """Test with custom chat template."""
        client = CoEditClient(chat_template="Edit: {text}")
        
        test_text = "They was happy."
        result = client.predict(test_text)
        
        assert isinstance(result, LanguageToolRemoteResult)
        
        print(f"\nCustom Prompt:")
        print(f"  Original: {test_text}")
        print(f"  Matches: {len(result.matches)}")
