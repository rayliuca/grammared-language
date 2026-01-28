import threading
import os

import os
import threading
from collections import OrderedDict
from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from .grpc_gen import ml_server_pb2

class SimpleCacheStore:
    """Thread-safe in-memory cache store."""
    def __init__(self, max_size=10000):
        self.store = OrderedDict()
        self.max_size = max_size
        self._lock = threading.RLock()
    
    def add(self, text: str, result: LanguageToolRemoteResult):
        with self._lock:
            if len(self.store) >= self.max_size:
                self.store.popitem(last=False)
            self.store[text] = result
    
    def contains(self, text: str) -> bool:
        with self._lock:
            return text in self.store
    
    def get(self, text: str):
        with self._lock:
            self.store.move_to_end(text)
            return self.store.get(text)