import threading
import os

import os
import threading

from grammared_language.language_tool.output_models import LanguageToolRemoteResult
from .grpc_gen import ml_server_pb2

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

def get_explain(
        original_sentence: str, 
        corrected_sentence: str, 
        mistake: dict, # {offset: int, length: int, text: str}
        replacement: str
    ) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None
    """
    Generate an explanation for a grammar correction using Ollama via OpenAI API.
    
    Args:
        original_sentence: Original sentence with error
        corrected_sentence: Corrected sentence
        mistake: Dict with 'offset', 'length', and 'text' keys
        replacement: The replacement text
        
    Returns:
        Explanation string from the LLM
    """
    if OpenAI is None:
        return f"Replaced '{mistake['text']}' with '{replacement}'"
    
    try:
        # Get configuration from environment
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        ollama_model = os.getenv("OLLAMA_MODEL", "smollm2")
        
        # Initialize OpenAI client pointing to Ollama
        client = OpenAI(
            base_url=ollama_base_url,
            api_key=ollama_api_key
        )
        system_prompt = """
You are an expert English grammar teacher.

Error Categories:
- Grammar mistakes
  - subject-verb agreement
  - pronoun-antecedent agreement
  - run-on sentences
  - sentence fragments
  - misplaced modifiers
- Spelling errors
  - misspellings
  - commonly confused homophones (their/there/they're)
- Punctuation issues
  - comma splices
  - apostrophe use
  - semicolon usage
- Word choice issues
  - basic usage

Do not provide any explanations.
Example outputs: "Grammar mistake: subject-verb agreement", "Spelling error", "Punctuation issue: comma splice", "Word choice issue: basic usage"
""".strip()
        # Create prompt for explanation
        prompt = f"""
Original: {original_sentence}
Corrected: {corrected_sentence}
Error: Changed mistake `{mistake['text']}` to `{replacement}`

What is the mistake category?""".strip()
        
        # Call Ollama via OpenAI API
        response = client.chat.completions.create(
            model=ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        explanation = response.choices[0].message.content.strip()
        print(f"Prompt sent to Ollama: {prompt}")
        print(f"Generated explanation: {explanation}")
        return explanation
        
    except Exception as e:
        # Fallback explanation if LLM call fails
        return f"Replaced '{mistake['text']}' with '{replacement}'"

def pydantic_match_to_ml_match(match, offset_adjustment: int = 0) -> ml_server_pb2.Match:
    """Convert Pydantic Match model to ml_server Match."""
    return ml_server_pb2.Match(
        offset=match.offset + offset_adjustment,
        length=match.length,
        id="gector",
        sub_id="",
        suggestions=match.suggestions,
        ruleDescription=match.rule.description if match.rule else None,
        matchDescription=match.message,
        matchShortDescription=match.shortMessage or match.message,
        url="",
        suggestedReplacements=[
            ml_server_pb2.SuggestedReplacement(
                replacement=r.replacement,
                description="",
                suffix="",
                confidence=0.8
            )
            for r in (match.suggested_replacements or [])
        ],
        autoCorrect=True,
        type=ml_server_pb2.Match.MatchType.Other,  # Grammar errors are "Other" type
        contextForSureMatch=0,
        rule=ml_server_pb2.Rule(
            sourceFile="gector",
            issueType=match.rule.issueType or "grammar",
            tempOff=False,
            category=ml_server_pb2.RuleCategory(
                id=match.rule.id or "gector",
                name=match.rule.description or "Grammar Error"
            ) if match.rule.category else None,
            isPremium=False,
            tags=[]
        ) if match.rule else None
    )
