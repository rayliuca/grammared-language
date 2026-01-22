"""
Tests for AsyncMultiClient
"""
import pytest
from unittest.mock import Mock, patch
from grammared_language.clients.async_multi_client import AsyncMultiClient
from grammared_language.clients.base_client import BaseClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult, Match


class MockClient(BaseClient):
    """Mock client for testing."""
    
    def __init__(self, name: str, return_matches=None):
        super().__init__()
        self.name = name
        self.return_matches = return_matches or []
    
    def _predict(self, text: str) -> str:
        return text
    
    def predict(self, text: str) -> LanguageToolRemoteResult:
        return LanguageToolRemoteResult(
            language="English",
            languageCode="en-US",
            matches=self.return_matches
        )


class TestAsyncMultiClient:
    """Test suite for AsyncMultiClient."""
    
    def test_init_with_clients(self):
        """Test initialization with pre-existing clients."""
        client1 = MockClient("client1")
        client2 = MockClient("client2")
        
        async_client = AsyncMultiClient(clients=[client1, client2])
        
        assert len(async_client.clients) == 2
        assert async_client.clients[0] == client1
        assert async_client.clients[1] == client2
    
    def test_init_empty(self):
        """Test initialization without clients."""
        async_client = AsyncMultiClient()
        
        assert len(async_client.clients) == 0
    
    def test_add_client(self):
        """Test adding a client after initialization."""
        async_client = AsyncMultiClient()
        client = MockClient("client1")
        
        async_client.add_client(client)
        
        assert len(async_client.clients) == 1
        assert async_client.clients[0] == client
    
    @pytest.mark.asyncio
    async def test_predict_async(self):
        """Test async prediction with multiple clients."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            replacements=["This"]
        )
        match2 = Match(
            message="Test error 2",
            shortMessage="Error 2",
            offset=5,
            length=3,
            replacements=["is"]
        )
        
        client1 = MockClient("client1", [match1])
        client2 = MockClient("client2", [match2])
        
        async_client = AsyncMultiClient(clients=[client1, client2])
        results = await async_client._predict_async("This is a test")
        
        assert len(results) == 2
        assert len(results[0].matches) == 1
        assert len(results[1].matches) == 1
        assert results[0].matches[0].message == "Test error 1"
        assert results[1].matches[0].message == "Test error 2"
    
    @pytest.mark.asyncio
    async def test_predict_with_merge(self):
        """Test merging and deduplicating results from multiple clients."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            suggestions=["This"]
        )
        match2 = Match(
            message="Test error 2",
            shortMessage="Error 2",
            offset=5,
            length=3,
            suggestions=["is"]
        )
        # Duplicate match with same offset, length, and suggestions (should be filtered)
        match3 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            suggestions=["This"]
        )
        
        client1 = MockClient("client1", [match1])
        client2 = MockClient("client2", [match2, match3])
        
        async_client = AsyncMultiClient(clients=[client1, client2])
        result = await async_client._predict_with_merge_async("This is a test")
        
        # Should have 2 matches (match3 is duplicate of match1 and filtered out)
        assert len(result.matches) == 2
        assert result.matches[0].offset == 0
        assert result.matches[1].offset == 5
        # Verify they are sorted by offset
        assert result.matches[0].offset < result.matches[1].offset
    
    def test_predict_sync(self):
        """Test synchronous predict wrapper."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            replacements=["This"]
        )
        
        client1 = MockClient("client1", [match1])
        
        async_client = AsyncMultiClient(clients=[client1])
        # Test synchronous wrapper directly (not in async context)
        results = async_client.predict("This is a test")
        
        assert len(results) == 1
        assert len(results[0].matches) == 1
    
    def test_callable(self):
        """Test that the client is callable via __call__."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            replacements=["This"]
        )
        client1 = MockClient("client1", [match1])
        
        async_client = AsyncMultiClient(clients=[client1])
        # Test __call__ method directly
        results = async_client("This is a test")
        
        assert len(results) == 1
        assert len(results[0].matches) == 1
    
    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test that exceptions from individual clients are handled."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            replacements=["This"]
        )
        client1 = MockClient("client1", [match1])
        client2 = MockClient("client2", [])
        
        # Make client2 raise an exception
        client2.predict = Mock(side_effect=ValueError("Test error"))
        
        async_client = AsyncMultiClient(clients=[client1, client2])
        results = await async_client._predict_async("This is a test")
        
        # Should only have result from client1, client2 should be filtered out
        assert len(results) == 1
        assert results[0].matches[0].message == "Test error 1"
    
    def test_init_from_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            AsyncMultiClient(config_path="nonexistent_config.yaml")
    
    @pytest.mark.asyncio
    async def test_predict_with_merge_empty_results(self):
        """Test merging with no valid results."""
        async_client = AsyncMultiClient()
        result = await async_client._predict_with_merge_async("This is a test")
        
        assert len(result.matches) == 0
        assert result.language == "English"
        assert result.languageCode == "en-US"
    
    @pytest.mark.asyncio
    async def test_predict_batch_async(self):
        """Test batch processing of multiple texts."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            replacements=["This"]
        )
        match2 = Match(
            message="Test error 2",
            shortMessage="Error 2",
            offset=0,
            length=2,
            replacements=["He"]
        )
        
        client1 = MockClient("client1", [match1])
        client2 = MockClient("client2", [match2])
        
        async_client = AsyncMultiClient(clients=[client1, client2])
        
        texts = ["This is a test", "He goes there", "Another sentence"]
        results = await async_client._predict_batch_async(texts)
        
        # Should get results for all texts
        assert len(results) == 3
        
        # Each text should have results from both clients
        for text_results in results:
            assert len(text_results) == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_same_text(self):
        """Test multiple concurrent requests with the same text."""
        client1 = MockClient("client1", [])
        
        async_client = AsyncMultiClient(clients=[client1])
        
        # Submit 10 concurrent requests for the same text
        import asyncio
        tasks = [async_client._predict_async("Test text") for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Should get 10 results
        assert len(results) == 10
        
        # Each result should be a list with one item (from client1)
        for result_list in results:
            assert len(result_list) == 1
    
    def test_context_manager(self):
        """Test context manager support."""
        client1 = MockClient("client1", [])
        
        with AsyncMultiClient(clients=[client1]) as async_client:
            assert len(async_client.clients) == 1
        
        # Executor should be shutdown after exiting context
        assert async_client._executor._shutdown
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager support."""
        client1 = MockClient("client1", [])
        
        async with AsyncMultiClient(clients=[client1]) as async_client:
            results = await async_client._predict_async("Test")
            assert len(results) == 1
        
        # Executor should be shutdown after exiting context
        assert async_client._executor._shutdown
    
    def test_predict_with_merge_sync(self):
        """Test synchronous predict_with_merge method."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            suggestions=["This"]
        )
        match2 = Match(
            message="Test error 2",
            shortMessage="Error 2",
            offset=5,
            length=3,
            suggestions=["is"]
        )
        
        client1 = MockClient("client1", [match1])
        client2 = MockClient("client2", [match2])
        
        async_client = AsyncMultiClient(clients=[client1, client2])
        result = async_client.predict_with_merge("This is a test")
        
        # Should have 2 matches
        assert len(result.matches) == 2
        assert result.matches[0].offset == 0
        assert result.matches[1].offset == 5
        assert result.language == "English"
        assert result.languageCode == "en-US"
    
    def test_predict_batch_sync(self):
        """Test synchronous predict_batch method."""
        match1 = Match(
            message="Test error 1",
            shortMessage="Error 1",
            offset=0,
            length=4,
            replacements=["This"]
        )
        
        client1 = MockClient("client1", [match1])
        
        async_client = AsyncMultiClient(clients=[client1])
        texts = ["This is a test", "He goes there"]
        results = async_client.predict_batch(texts)
        
        # Should get results for both texts
        assert len(results) == 2
        # Each text should have results from client1
        for text_results in results:
            assert len(text_results) == 1
