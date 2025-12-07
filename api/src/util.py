import difflib
import threading

from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import re

from .output_models import LanguageToolRemoteResult, Match, Context, Rule, SuggestedReplacement

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
        


"""
Grammar Error Correction - Extract Replacements
Extracts replacements between original and corrected text for grammar correction.  
"""
class GrammarCorrectionExtractor:
    """Extract replacement operations from original to corrected text."""
    
    def __init__(self, min_length: int = 1):
        """
        Initialize the extractor. 
        
        Args:
            min_length: Minimum length of original text to be replaced (must be > 0)
        """
        self.min_length = max(1, min_length)  # Ensure at least 1
    
    def extract_replacements(self, original: str, corrected: str, fix_tokenization=True) -> List[Match]:
        """
        Extract replacements needed to transform original text to corrected text.
        
        Args:
            original: The original text with errors
            corrected: The corrected text
            
        Returns:
            List of replacements, each containing:
            - offset: int (start index in original text)
            - length: int (length of text to replace in original)
            - replacement: str (corrected text to insert)
        """
        # Pre-process to fix common tokenization mistakes in corrected text
        if fix_tokenization:
            fixed_corrected = self._fix_tokenization_mistakes(corrected)
        else:
            fixed_corrected = corrected
        
        # Tokenize into words while preserving positions
        orig_tokens, orig_positions = self._tokenize_with_positions(original)
        corr_tokens, corr_positions = self._tokenize_with_positions(fixed_corrected)

        matches = []        
        # Use SequenceMatcher on tokens
        matcher = SequenceMatcher(None, orig_tokens, corr_tokens)
        for tag, i1, i2, j1, j2 in matcher. get_opcodes():
            if tag == 'replace':
                # Get character positions
                offset = orig_positions[i1]
                end_pos = orig_positions[i2 - 1] + len(orig_tokens[i2 - 1])
                length = end_pos - offset
                
                # Get replacement text from corrected
                if j2 > j1:
                    repl_start = corr_positions[j1]
                    repl_end = corr_positions[j2 - 1] + len(corr_tokens[j2 - 1])
                    replacement = fixed_corrected[repl_start:repl_end]
                else:
                    replacement = ""
                
                # Apply constraints:
                # - length must be >= min_length
                # - replacement cannot be empty
                if length >= self.min_length and replacement != "":
                    matches.append(
                        Match(
                            message="ML-based grammar correction",
                            suggested_replacements=[
                                SuggestedReplacement(
                                    replacement=replacement
                                )
                            ],
                            offset=offset,
                            length=length,
                            # rule=Rule(
                            #     id=rule_id,
                            #     description=rule_description
                            # )
                        )
                    )
                
        return matches
    
    def _fix_tokenization_mistakes(self, text: str) -> str:
        """
        Fix common tokenization mistakes from generation models.
        
        Common mistakes:
        - "I ' m" -> "I'm" (spaces around apostrophe in contractions)
        - "word ." -> "word." (space before period)
        - "word ," -> "word," (space before comma)
        - "word ?" -> "word?" (space before question mark)
        - "word !" -> "word!" (space before exclamation)
        - "word ;" -> "word;" (space before semicolon)
        - "word :" -> "word:" (space before colon)
        
        Args:
            text: Input text with potential tokenization mistakes
            
        Returns:
            Fixed text with proper spacing
        """
        result = text
        
        # Fix spaces around apostrophes in contractions
        # Pattern: letter + space + ' + space + letter
        result = re. sub(r"(\w)\s+'\s+(\w)", r"\1'\2", result)
        
        # Fix space before punctuation
        # Period
        result = re.sub(r'\s+\.', '. ', result)
        # Comma
        result = re. sub(r'\s+,', ',', result)
        # Question mark
        result = re.sub(r'\s+\?', '?', result)
        # Exclamation mark
        result = re.sub(r'\s+!', '! ', result)
        # Semicolon
        result = re. sub(r'\s+;', ';', result)
        # Colon (but be careful not to break things like "http : //")
        result = re.sub(r'(\w)\s+:', r'\1:', result)
        
        # Fix space after opening quotes and before closing quotes
        result = re. sub(r'"\s+', '"', result)
        result = re.sub(r'\s+"', '"', result)
        
        # Fix spaces around hyphens in compound words
        result = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', result)
        
        return result
    
    def _tokenize_with_positions(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Tokenize text into words while preserving character positions.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (tokens, positions) where positions[i] is the character offset of tokens[i]
        """
        tokens = []
        positions = []
        current_token = ""
        token_start = 0
        
        for i, char in enumerate(text):
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    positions.append(token_start)
                    current_token = ""
            else:
                if not current_token:
                    token_start = i
                current_token += char
        
        # Add last token
        if current_token:
            tokens.append(current_token)
            positions.append(token_start)
        
        return tokens, positions
    
    def apply_replacements(self, original: str, replacements: List[Dict[str, any]]) -> str:
        """
        Apply replacements to original text to get corrected text.
        
        Args:
            original: The original text
            replacements: List of replacement operations
            
        Returns:
            The corrected text
        """
        # Sort replacements by offset in reverse order to avoid index shifting
        sorted_replacements = sorted(replacements, key=lambda x: x['offset'], reverse=True)
        
        result = original
        for r in sorted_replacements:
            offset = r['offset']
            length = r['length']
            replacement = r['replacement']
            
            result = result[:offset] + replacement + result[offset + length:]
        
        return result
