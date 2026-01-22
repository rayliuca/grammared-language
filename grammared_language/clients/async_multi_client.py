import asyncio
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from grammared_language.clients.base_client import BaseClient
from grammared_language.language_tool.output_models import LanguageToolRemoteResult


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
            self._init_from_config(config_path)
    
    def _init_from_config(self, config_path: str):
        """Initialize clients from configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        for model_name, model_config in config.items():
            if isinstance(model_config, dict):
                client = self._create_client_from_config(model_name, model_config)
                if client:
                    self.clients.append(client)
    
    def _create_client_from_config(self, model_name: str, config: Dict[str, Any]) -> Optional[BaseClient]:
        """Create a client instance from configuration."""
        model_type = config.get('type')
        backend = config.get('backend')
        
        try:
            # Import appropriate client class based on type
            if model_type == 'gector':
                from grammared_language.clients.gector_client import GectorClient
                
                # Extract parameters for GectorClient
                model_id = config.get('pretrained_model_name_or_path')
                triton_model_name = config.get('triton_model_name') if backend == 'triton' else None
                
                return GectorClient(
                    model_id=model_id,
                    triton_model_name=triton_model_name,
                    **{k: v for k, v in config.items() if k not in ['type', 'backend', 'pretrained_model_name_or_path', 'triton_model_name']}
                )
            
            elif model_type == 'grammared_classifier':
                from grammared_language.clients.grammar_classification_client import GrammarClassificationClient
                
                # Extract parameters for GrammarClassificationClient
                model_id = config.get('pretrained_model_name_or_path')
                triton_model_name = config.get('triton_model_name')
                triton_host = config.get('triton_hostname', 'localhost')
                triton_port = config.get('triton_port', 8001)
                triton_protocol = config.get('triton_protocol', 'grpc')  # Default to gRPC
                
                return GrammarClassificationClient(
                    model_id=model_id,
                    backend=backend,
                    triton_model_name=triton_model_name,
                    triton_host=triton_host,
                    triton_port=triton_port,
                    triton_protocol=triton_protocol,
                    **{k: v for k, v in config.items() if k not in ['type', 'backend', 'pretrained_model_name_or_path', 'triton_model_name', 'triton_hostname', 'triton_port', 'triton_protocol']}
                )
        except Exception as e:
            print(f"Failed to create client for {model_name}: {e}")
        
        return None
    
    def add_client(self, client: BaseClient):
        """Add a client to the list."""
        self.clients.append(client)
    
    async def predict_async(self, text: str) -> List[LanguageToolRemoteResult]:
        """
        Run all clients asynchronously on the input text.
        
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
    
    async def predict_batch_async(self, texts: List[str]) -> List[List[LanguageToolRemoteResult]]:
        """
        Process multiple texts concurrently across all clients.
        
        Args:
            texts: List of input texts to process
            
        Returns:
            List of results for each text, where each result is a list from all clients
        """
        tasks = [self.predict_async(text) for text in texts]
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
    
    async def predict_with_merge(self, text: str) -> LanguageToolRemoteResult:
        """
        Run all clients asynchronously and merge results.
        
        Args:
            text: Input text to process
            
        Returns:
            Merged LanguageToolRemoteResult
        """
        results = await self.predict_async(text)
        
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
        Synchronous wrapper for predict_async.
        
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
                future = executor.submit(asyncio.run, self.predict_async(text))
                return future.result()
        except RuntimeError:
            # No event loop exists or is running in this thread, create one
            return asyncio.run(self.predict_async(text))
    
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
