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
        opcodes = list(matcher. get_opcodes())
        
        for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
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
            
            elif tag == 'insert':
                # Convert insertion to replacement by including adjacent token
                # This ensures we meet the min_length requirement
                
                # Get the inserted text
                repl_start = corr_positions[j1]
                repl_end = corr_positions[j2 - 1] + len(corr_tokens[j2 - 1])
                inserted_text = fixed_corrected[repl_start:repl_end]
                
                # Try to merge with the previous or next token to create a replacement
                converted = self._convert_insertion_to_replacement(
                    original, orig_tokens, orig_positions, 
                    i1, inserted_text, opcodes, idx
                )
                
                if converted:
                    matches. append(converted)
        
        return matches
    
    def _convert_insertion_to_replacement(self, original: str, orig_tokens: List[str], 
                                         orig_positions: List[int], insert_pos: int,
                                         inserted_text: str, opcodes: List, current_idx: int) -> Dict:
        """
        Convert an insertion to a replacement by including an adjacent token.
        
        Strategy:
        1. Try to include the next token (insert before next word)
        2. If no next token, include the previous token (append after previous word)
        
        Args:
            original: Original text
            orig_tokens: Original tokens
            orig_positions: Original token positions
            insert_pos: Position where insertion occurs
            inserted_text: Text to be inserted
            opcodes: All opcodes from SequenceMatcher
            current_idx: Current opcode index
            
        Returns:
            Replacement dict or None if conversion not possible
        """
        # Check if there's a next token (prefer inserting before next word)
        if insert_pos < len(orig_tokens):
            # Check if the next operation is not already handling this token
            next_token_used = False
            if current_idx + 1 < len(opcodes):
                next_tag, next_i1, _, _, _ = opcodes[current_idx + 1]
                if next_tag in ('replace', 'delete') and next_i1 == insert_pos:
                    next_token_used = True
            
            if not next_token_used:
                # Include the next token in the replacement
                offset = orig_positions[insert_pos]
                next_token = orig_tokens[insert_pos]
                length = len(next_token)
                replacement = inserted_text + " " + next_token
                
                return Match(
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
        
        # Otherwise, try to include the previous token
        if insert_pos > 0:
            prev_idx = insert_pos - 1
            
            # Check if the previous token is already being replaced/deleted
            prev_token_used = False
            for other_idx, (other_tag, other_i1, other_i2, _, _) in enumerate(opcodes):
                if other_idx != current_idx and other_tag in ('replace', 'delete'):
                    if prev_idx >= other_i1 and prev_idx < other_i2:
                        prev_token_used = True
                        break
            
            if not prev_token_used:
                # Include the previous token in the replacement
                offset = orig_positions[prev_idx]
                prev_token = orig_tokens[prev_idx]
                length = len(prev_token)
                replacement = prev_token + " " + inserted_text
                
                return {
                    'offset': offset,
                    'length': length,
                    'replacement': replacement
                }
        
        return None
    
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
        result = re.sub(r"(\w)\s+'\s+(\w)", r"\1'\2", result)
        
        # Fix space before punctuation
        # Period
        result = re.sub(r'\s+\.', '. ', result)
        # Comma
        result = re.sub(r'\s+,', ',', result)
        # Question mark
        result = re.sub(r'\s+\?', '?', result)
        # Exclamation mark
        result = re.sub(r'\s+!', '! ', result)
        # Semicolon
        result = re. sub(r'\s+;', ';', result)
        # Colon (but be careful not to break things like "http : //")
        result = re.sub(r'(\w)\s+:', r'\1:', result)
        
        # Fix space after opening quotes and before closing quotes
        result = re.sub(r'"\s+', '"', result)
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
