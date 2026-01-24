import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from grammared_language.clients.base_client import BaseClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from grammared_language.utils.config_parser import create_clients_from_config


class AsyncMultiClient:
    """Client that runs multiple grammar correction clients asynchronously."""
    
    def __init__(self, clients: Optional[List[BaseClient]] = None, config_path: Optional[str] = None, max_workers: Optional[int] = None):
        """
        Initialize AsyncMultiClient.
        
        Args:
            clients: List of pre-initialized client instances
            config_path: Path to model_config.yaml to initialize clients from
            max_workers: Maximum number of worker threads for executor (default: None = auto)
        """
        self.clients: List[BaseClient] = clients or []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        if config_path and not clients:
            self.clients = create_clients_from_config(config_path)
    
    def add_client(self, client: BaseClient):
        """Add a client to the list."""
        self.clients.append(client)
    
    async def _predict_async(self, text: str) -> List[LanguageToolRemoteResult]:
        """
        Run all clients asynchronously on the input text (internal async method).
        
        Args:
            text: Input text to process
            
        Returns:
            List of LanguageToolRemoteResult from each client
        """
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(self._executor, client.predict, text) for client in self.clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Client {i} failed with error: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _predict_batch_async(self, texts: List[str]) -> List[List[LanguageToolRemoteResult]]:
        """
        Process multiple texts concurrently across all clients (internal async method).
        
        Args:
            texts: List of input texts to process
            
        Returns:
            List of results for each text, where each result is a list from all clients
        """
        tasks = [self._predict_async(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Text {i} failed with error: {result}")
                valid_results.append([])  # Empty result for failed text
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _predict_with_merge_async(self, text: str) -> LanguageToolRemoteResult:
        """
        Run all clients asynchronously and merge results (internal async method).
        
        Args:
            text: Input text to process
            
        Returns:
            Merged LanguageToolRemoteResult
        """
        results = await self._predict_async(text)
        
        if not results:
            return LanguageToolRemoteResult(
                language="English",
                languageCode="en-US",
                matches=[]
            )
        
        # Merge matches from all results
        merged_matches = []
        seen_replacements = set()
        
        for result in results:
            for match in result.matches:
                # Create unique key for deduplication based on offset and length
                # Use suggested_replacements to get replacement strings
                replacements_list = []
                if match.suggested_replacements:
                    replacements_list = [r.replacement for r in match.suggested_replacements]
                elif match.suggestions:
                    replacements_list = match.suggestions
                    
                key = (match.offset, match.length, tuple(replacements_list))
                if key not in seen_replacements:
                    seen_replacements.add(key)
                    merged_matches.append(match)
        
        # Sort by offset
        merged_matches.sort(key=lambda m: m.offset)
        
        return LanguageToolRemoteResult(
            language="English",
            languageCode="en-US",
            matches=merged_matches
        )
    
    def predict(self, text: str) -> List[LanguageToolRemoteResult]:
        """
        Run all clients and return list of results from each client.
        
        Args:
            text: Input text to process
            
        Returns:
            List of LanguageToolRemoteResult from each client
        """
        try:
            loop = asyncio.get_running_loop()
            # If loop is already running, we can't use run_until_complete or asyncio.run in this thread
            # Create a temporary executor to run the async task in a new thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self._predict_async(text))
                return future.result()
        except RuntimeError:
            # No event loop exists or is running in this thread, create one
            return asyncio.run(self._predict_async(text))
    
    def predict_with_merge(self, text: str) -> LanguageToolRemoteResult:
        """
        Run all clients, merge their results, and return a single merged result.
        
        Args:
            text: Input text to process
            
        Returns:
            Merged LanguageToolRemoteResult with deduplicated matches from all clients
        """
        try:
            loop = asyncio.get_running_loop()
            # If loop is already running, run in new thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self._predict_with_merge_async(text))
                return future.result()
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self._predict_with_merge_async(text))
    
    def predict_batch(self, texts: List[str]) -> List[List[LanguageToolRemoteResult]]:
        """
        Process multiple texts concurrently across all clients.
        
        Args:
            texts: List of input texts to process
            
        Returns:
            List of results for each text, where each result is a list from all clients
        """
        try:
            loop = asyncio.get_running_loop()
            # If loop is already running, run in new thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self._predict_batch_async(texts))
                return future.result()
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self._predict_batch_async(texts))
    
    def __call__(self, text: str) -> List[LanguageToolRemoteResult]:
        """Make the client callable."""
        return self.predict(text)
    
    def close(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()
        return False
