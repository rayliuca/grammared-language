"""
Functional tests for AsyncMultiClient with real Triton server.
These tests require RUN_NON_HERMETIC=true and a running Triton server.
"""
import pytest
import os
import asyncio
from pathlib import Path

from grammared_language.clients.async_multi_client import AsyncMultiClient
from grammared_language.clients.gector_client import GectorClient
from grammared_language.clients.grammar_classification_client import GrammarClassificationClient

RUN_NON_HERMETIC = os.getenv("RUN_NON_HERMETIC", "false").lower() in ("true", "1", "yes")

try:
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except ImportError:
    grpcclient = None
    TRITON_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TRITON_AVAILABLE or not TRANSFORMERS_AVAILABLE or not RUN_NON_HERMETIC,
    reason="Requires tritonclient, transformers and RUN_NON_HERMETIC=true (needs running Triton server)"
)


@pytest.fixture(scope="module")
def triton_ready():
    """Check if Triton server is ready."""
    try:
        client = grpcclient.InferenceServerClient(url="localhost:8001")
        if not client.is_server_live():
            pytest.skip("Triton server is not live at localhost:8001")
        return True
    except Exception as e:
        pytest.skip(f"Failed to connect to Triton server: {e}")


@pytest.fixture(scope="module")
def config_path():
    """Return path to model_config.yaml."""
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / "model_config.yaml"
    if not config_file.exists():
        pytest.skip(f"Config file not found: {config_file}")
    return str(config_file)


class TestAsyncMultiClientFromConfig:
    """Test AsyncMultiClient initialization from config file."""
    
    def test_init_from_config(self, triton_ready, config_path):
        """Test that AsyncMultiClient can initialize from config."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        # Should have loaded clients from config
        assert len(async_client.clients) > 0
        print(f"Loaded {len(async_client.clients)} clients from config")
    
    @pytest.mark.asyncio
    async def test_predict_async_with_real_models(self, triton_ready, config_path):
        """Test async prediction with real Triton models."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded from config")
        
        # Test with text containing grammar errors
        text = "This are a test sentence with grammar error."
        
        results = await async_client._predict_async(text)
        
        # Should get results from at least one client
        assert len(results) > 0
        print(f"Got {len(results)} results")
        
        # Each result should be a LanguageToolRemoteResult
        for i, result in enumerate(results):
            assert hasattr(result, 'matches')
            assert hasattr(result, 'language')
            print(f"Client {i}: {len(result.matches)} matches")
            for match in result.matches:
                replacements = []
                if match.suggested_replacements:
                    replacements = [r.replacement for r in match.suggested_replacements]
                elif match.suggestions:
                    replacements = match.suggestions
                print(f"  - [{match.offset}:{match.offset+match.length}] {match.message} -> {replacements}")
    
    def test_predict_with_merge_real_models(self, triton_ready, config_path):
        """Test merging results from multiple real models."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded from config")
        
        text = "He go to school yesterday."
        
        merged_result = async_client.predict_with_merge(text)
        
        # Should get a merged result
        assert hasattr(merged_result, 'matches')
        assert merged_result.language == "English"
        assert merged_result.languageCode == "en-US"
        
        print(f"Merged result: {len(merged_result.matches)} matches")
        for match in merged_result.matches:
            replacements = []
            if match.suggested_replacements:
                replacements = [r.replacement for r in match.suggested_replacements]
            elif match.suggestions:
                replacements = match.suggestions
            print(f"  [{match.offset}:{match.offset+match.length}] {match.message}")
            print(f"    Replacements: {replacements}")
    
    def test_sync_predict_with_real_models(self, triton_ready, config_path):
        """Test synchronous predict with real models."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded from config")
        
        text = "I has a apple."
        
        # Use synchronous wrapper
        results = async_client.predict(text)
        
        assert len(results) > 0
        print(f"Sync predict: {len(results)} results")


class TestAsyncMultiClientWithManualClients:
    """Test AsyncMultiClient with manually created client instances."""
    
    @pytest.mark.asyncio
    async def test_mixed_clients_async(self, triton_ready):
        """Test AsyncMultiClient with multiple client types (Gector + Classifier)."""
        clients = []
        
        # Try to add GectorClient
        try:
            gector = GectorClient(
                model_id="gotutiyan/gector-deberta-large-5k",
                triton_model_name="gector_deberta_large"
            )
            clients.append(gector)
        except Exception as e:
            print(f"Could not add GectorClient: {e}")
        
        # Try to add GrammarClassificationClient with gRPC
        try:
            classifier = GrammarClassificationClient(
                model_id="rayliuca/grammared-classifier-deberta-v3-small",
                backend="triton",
                triton_model_name="grammared-classifier-deberta-v3-small",
                triton_host="localhost",
                triton_port=8001,
                triton_protocol="grpc"
            )
            clients.append(classifier)
        except Exception as e:
            print(f"Could not add GrammarClassificationClient: {e}")
        
        if len(clients) == 0:
            pytest.skip("No clients could be initialized")
        
        async_client = AsyncMultiClient(clients=clients)
        
        text = "He don't like apples."
        results = await async_client._predict_async(text)
        
        # Should get at least one result
        assert len(results) >= 1
        print(f"Mixed clients: got {len(results)} results from {len(clients)} clients")
    
    @pytest.mark.asyncio
    async def test_concurrent_single_text_multiple_times(self, triton_ready, config_path):
        """Test the same text being processed multiple times concurrently."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded from config")
        
        text = "This are a test."
        
        # Process same text 3 times concurrently (reduced from 5)
        tasks = [async_client._predict_async(text) for _ in range(3)]
        all_results = await asyncio.gather(*tasks)
        
        assert len(all_results) == 3
        print(f"Processed same text 3 times concurrently")
        
        # All results should be similar since it's the same text
        for i, results in enumerate(all_results):
            print(f"  Request {i}: {len(results)} client results")
            assert len(results) > 0
    


class TestAsyncMultiClientEdgeCases:
    """Test edge cases with real server."""
    
    @pytest.mark.asyncio
    async def test_empty_text(self, triton_ready, config_path):
        """Test with empty text."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded")
        
        results = await async_client._predict_async("")
        
        # Should handle empty text gracefully
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_very_long_text(self, triton_ready, config_path):
        """Test with very long text."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded")
        
        # Create long text
        text = "This is a test sentence. " * 100
        
        results = await async_client._predict_async(text)
        
        assert len(results) > 0
        print(f"Long text ({len(text)} chars): processed by {len(results)} clients")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)  # Add timeout to prevent hanging
    async def test_multiple_concurrent_requests(self, triton_ready, config_path):
        """Test multiple concurrent requests to the same client."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded")
        
        # Use smaller number of texts to avoid gevent issues
        texts = [
            "This are wrong.",
            "He go to school."
        ]
        
        # Submit multiple concurrent requests using the batch method
        all_results = await async_client._predict_batch_async(texts)
        
        assert len(all_results) == len(texts)
        print(f"Processed {len(texts)} concurrent requests")
        
        # Verify each text got processed
        for i, results in enumerate(all_results):
            print(f"  Text {i}: {len(results)} client results")
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)  # Add timeout to prevent hanging
    async def test_concurrent_single_text_multiple_times_2(self, triton_ready, config_path):
        """Test the same text being processed multiple times concurrently (second test)."""
        async_client = AsyncMultiClient(config_path=config_path)
        
        if len(async_client.clients) == 0:
            pytest.skip("No clients loaded")
        
        text = "This are a test."
        
        # Process same text 3 times concurrently (reduced from 5)
        tasks = [async_client._predict_async(text) for _ in range(3)]
        all_results = await asyncio.gather(*tasks)
        
        assert len(all_results) == 3
        print(f"Processed same text 3 times concurrently")
        
        # All results should be similar since it's the same text
        for i, results in enumerate(all_results):
            print(f"  Request {i}: {len(results)} client results")
            assert len(results) > 0
