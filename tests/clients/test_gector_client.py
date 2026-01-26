"""Tests for GectorClient."""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# Check for optional dependencies
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from gector import GECToR, predict, load_verb_dict
    GECTOR_AVAILABLE = True
except ImportError:
    GECTOR_AVAILABLE = False

try:
    from gector import GECToRTriton
    GECTOR_TRITON_AVAILABLE = True
except (ImportError, AttributeError):
    GECTOR_TRITON_AVAILABLE = False

# Determine if non-hermetic tests should run
RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")


# Unit tests (hermetic - no external dependencies)
class TestGectorClientUnit:
    """Unit tests for GectorClient (hermetic)."""
    
    @pytest.mark.unit
    def test_gector_client_import(self):
        """Test that GectorClient can be imported."""
        from grammared_language.clients.gector_client import GectorClient
        assert GectorClient is not None
    
    @pytest.mark.unit
    def test_base_client_import(self):
        """Test that BaseClient can be imported."""
        from grammared_language.clients.base_client import BaseClient
        assert BaseClient is not None
    
    @pytest.mark.unit
    def test_language_tool_remote_result_import(self):
        """Test that LanguageToolRemoteResult can be imported."""
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert LanguageToolRemoteResult is not None
    
    @pytest.mark.unit
    @patch('grammared_language.clients.gector_client.GECToR')
    @patch('grammared_language.clients.gector_client.AutoTokenizer')
    @patch('grammared_language.clients.gector_client.load_verb_dict')
    def test_gector_client_initialization(self, mock_load_verb_dict, mock_tokenizer, mock_gector):
        """Test GectorClient initialization with mocked dependencies."""
        from grammared_language.clients.gector_client import GectorClient
        
        # Setup mocks
        mock_gector.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_load_verb_dict.return_value = ({}, {})
        
        # Create client
        client = GectorClient(
            pretrained_model_name_or_path="test-model",
            verb_dict_path="test-vocab.txt"
        )
        
        # Verify initialization
        assert client.model is not None
        assert client.tokenizer is not None
        assert hasattr(client, 'encode')
        assert hasattr(client, 'decode')
        assert hasattr(client, 'pred_config')
        
        # Verify default pred_config values
        assert client.pred_config['keep_confidence'] == 0
        assert client.pred_config['min_error_prob'] == 0
        assert client.pred_config['n_iteration'] == 5
        assert client.pred_config['batch_size'] == 2
    
    @pytest.mark.unit
    @patch('grammared_language.clients.gector_client.GECToRTriton')
    @patch('grammared_language.clients.gector_client.AutoTokenizer')
    @patch('grammared_language.clients.gector_client.load_verb_dict')
    def test_gector_client_initialization_with_triton(self, mock_load_verb_dict, mock_tokenizer, mock_gector_triton):
        """Test GectorClient initialization with Triton model."""
        from grammared_language.clients.gector_client import GectorClient
        
        # Setup mocks
        mock_gector_triton.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_load_verb_dict.return_value = ({}, {})
        
        # Create client with triton model name
        client = GectorClient(
            pretrained_model_name_or_path="test-model",
            triton_model_name="gector_bert",
            verb_dict_path="test-vocab.txt"
        )
        
        # Verify Triton model was used
        mock_gector_triton.from_pretrained.assert_called_once_with("test-model", model_name="gector_bert")
        assert client.model is not None
    
    @pytest.mark.unit
    @patch('grammared_language.clients.gector_client.predict')
    @patch('grammared_language.clients.gector_client.GECToR')
    @patch('grammared_language.clients.gector_client.AutoTokenizer')
    @patch('grammared_language.clients.gector_client.load_verb_dict')
    def test_gector_client_predict_method(self, mock_load_verb_dict, mock_tokenizer, mock_gector, mock_predict):
        """Test GectorClient._predict method with mocked predict function."""
        from grammared_language.clients.gector_client import GectorClient
        
        # Setup mocks
        mock_gector.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_load_verb_dict.return_value = ({}, {})
        mock_predict.return_value = ["This is corrected text."]
        
        # Create client and test predict
        client = GectorClient(pretrained_model_name_or_path="test-model", verb_dict_path="test-vocab.txt")
        result = client._predict("This is test text")
        
        # Verify predict was called
        mock_predict.assert_called_once()
        assert result == "This is corrected text."
    
    @pytest.mark.unit
    @patch('grammared_language.clients.gector_client.predict')
    @patch('grammared_language.clients.gector_client.GECToR')
    @patch('grammared_language.clients.gector_client.AutoTokenizer')
    @patch('grammared_language.clients.gector_client.load_verb_dict')
    def test_gector_client_full_predict_pipeline(self, mock_load_verb_dict, mock_tokenizer, mock_gector, mock_predict):
        """Test full predict pipeline returning LanguageToolRemoteResult."""
        from grammared_language.clients.gector_client import GectorClient
        
        # Setup mocks
        mock_gector.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_load_verb_dict.return_value = ({}, {})
        mock_predict.return_value = ["This is the corrected text."]
        
        # Create client
        client = GectorClient(pretrained_model_name_or_path="test-model", verb_dict_path="test-vocab.txt")
        
        # Test full predict
        result = client.predict("This is test text")
        
        # Verify result is LanguageToolRemoteResult
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        assert isinstance(result.matches, list)
    
    @pytest.mark.unit
    @patch('grammared_language.clients.gector_client.predict')
    @patch('grammared_language.clients.gector_client.GECToR')
    @patch('grammared_language.clients.gector_client.AutoTokenizer')
    @patch('grammared_language.clients.gector_client.load_verb_dict')
    def test_gector_client_callable(self, mock_load_verb_dict, mock_tokenizer, mock_gector, mock_predict):
        """Test that GectorClient is callable."""
        from grammared_language.clients.gector_client import GectorClient
        
        # Setup mocks
        mock_gector.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_load_verb_dict.return_value = ({}, {})
        mock_predict.return_value = ["Corrected text."]
        
        # Create client and test callable
        client = GectorClient(pretrained_model_name_or_path="test-model", verb_dict_path="test-vocab.txt")
        result = client("Test text")
        
        # Verify result
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)


# Non-hermetic tests (require actual models and Triton server)
@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE or not GECTOR_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires transformers, gector, and RUN_NON_HERMETIC=true"
)
class TestGectorClientNonHermetic:
    """Non-hermetic tests for GectorClient (requires actual models)."""
    
    @pytest.fixture(scope="class")
    def model_id(self):
        """Return the model ID to use for testing."""
        return "gotutiyan/gector-bert-base-cased-5k"
    
    @pytest.fixture(scope="class")
    def verb_dict_path(self):
        """Return the path to verb dictionary."""
        return "data/verb-form-vocab.txt"
    
    @pytest.fixture(scope="class")
    def gector_client_local(self, model_id, verb_dict_path):
        """Create a GectorClient with local model (no Triton)."""
        from grammared_language.clients.gector_client import GectorClient
        
        try:
            client = GectorClient(
                pretrained_model_name_or_path=model_id,
                verb_dict_path=verb_dict_path,
                triton_model_name=None
            )
            return client
        except Exception as e:
            pytest.skip(f"Failed to initialize GectorClient: {e}")
    
    @pytest.fixture(scope="class")
    def gector_client_triton(self, model_id, verb_dict_path):
        """Create a GectorClient with Triton model."""
        from grammared_language.clients.gector_client import GectorClient
        
        try:
            # Check if Triton server is available
            import tritonclient.http as httpclient
            triton_client = httpclient.InferenceServerClient(url="localhost:8000")
            if not triton_client.is_server_live():
                pytest.skip("Triton server is not live")
            
            client = GectorClient(
                pretrained_model_name_or_path=model_id,
                verb_dict_path=verb_dict_path,
                triton_model_name="gector_bert"
            )
            return client
        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")
        except Exception as e:
            pytest.skip(f"Failed to initialize GectorClient with Triton: {e}")
    
    @pytest.mark.functional
    def test_gector_client_local_simple_correction(self, gector_client_local):
        """Test simple grammar correction with local model."""
        text = "I has a cat"
        result = gector_client_local.predict(text)
        
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        
        # Should detect "has" -> "have" error
        assert len(result.matches) > 0
        
        print(f"Original: {text}")
        print(f"Matches: {result.matches}")
    
    @pytest.mark.functional
    @pytest.mark.requires_triton
    def test_gector_client_triton_simple_correction(self, gector_client_triton):
        """Test simple grammar correction with Triton model."""
        text = "She go to school"
        result = gector_client_triton.predict(text)
        
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
        assert result.language == "English"
        assert result.languageCode == "en-US"
        
        # Should detect "go" -> "goes" or "went" error
        assert len(result.matches) > 0
        
        print(f"Original: {text}")
        print(f"Matches: {result.matches}")
    
    @pytest.mark.functional
    def test_gector_client_local_no_errors(self, gector_client_local):
        """Test text with no errors returns empty matches."""
        text = "This is a correct sentence."
        result = gector_client_local.predict(text)
        
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
        
        # May have 0 matches if no errors detected
        print(f"Original: {text}")
        print(f"Matches: {result.matches}")
    
    @pytest.mark.functional
    def test_gector_client_local_multiple_errors(self, gector_client_local):
        """Test text with multiple grammar errors."""
        text = "She dont likes going to the store on monday"
        result = gector_client_local.predict(text)
        
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
        
        # Should detect multiple errors
        print(f"Original: {text}")
        print(f"Matches found: {len(result.matches)}")
        for match in result.matches:
            print(f"  - {match.message}: {match.suggestions}")
    
    @pytest.mark.functional
    def test_gector_client_callable_interface(self, gector_client_local):
        """Test that client can be called directly."""
        text = "They was happy"
        result = gector_client_local(text)
        
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
        assert len(result.matches) > 0
    
    @pytest.mark.functional
    def test_gector_client_custom_params(self, model_id, verb_dict_path):
        """Test GectorClient with custom prediction parameters."""
        from grammared_language.clients.gector_client import GectorClient
        
        client = GectorClient(
            pretrained_model_name_or_path=model_id,
            verb_dict_path=verb_dict_path,
            keep_confidence=0.5,
            min_error_prob=0.1,
            n_iteration=3,
            batch_size=1
        )
        
        # Verify custom config
        assert client.pred_config['keep_confidence'] == 0.5
        assert client.pred_config['min_error_prob'] == 0.1
        assert client.pred_config['n_iteration'] == 3
        assert client.pred_config['batch_size'] == 1
        
        # Test prediction with custom params
        text = "I has a dog"
        result = client.predict(text)
        
        from grammared_language.language_tool.output_models import LanguageToolRemoteResult
        assert isinstance(result, LanguageToolRemoteResult)
