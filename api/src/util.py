import difflib
import threading
from .output_models import LanguageToolRemoteResult, Match, Context, Rule

def get_diff(sequence1, sequence2, rule_id="gector", rule_description="ML-based result."):
    s = difflib.SequenceMatcher(None, sequence1, sequence2)
    
    matches = []
    for opcode, i1, i2, j1, j2 in s.get_opcodes():
        if opcode == "equal":
            continue
        offset = i1
        length = i2-i1
        matches.append(
            Match(
                message=opcode,
                context=Context(
                    text=sequence2[j1:j2],
                    offset=offset,
                    length=length
                ),
                offset=offset,
                length=length,
                rule=Rule(
                    id=rule_id,
                    description=rule_description
                )
            )
        )
    return matches


class SimpleCacheStore:
    """Thread-safe in-memory cache store."""
    def __init__(self, max_size=10000):
        self.store = {}
        self.order = []
        self.max_size = max_size
        self._lock = threading.RLock()
    
    def add(self, text: str, result: LanguageToolRemoteResult):
        with self._lock:
            if len(self.store) >= self.max_size:
                oldest = self.order.pop(0)
                del self.store[oldest]
            self.store[text] = result
            self.order.append(text)
    
    def contains(self, text: str) -> bool:
        with self._lock:
            return text in self.store
    
    def get(self, text: str):
        with self._lock:
            return self.store.get(text)