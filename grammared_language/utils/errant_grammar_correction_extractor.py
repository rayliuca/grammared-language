"""
Grammar Error Correction - Extract Replacements using ERRANT
Extracts replacements between original and corrected text using the ERRANT library.
"""
from typing import List
import errant

from grammared_language.language_tool.output_models import Match, SuggestedReplacement, MatchType


class ErrantGrammarCorrectionExtractor:
    """Extract replacement operations from original to corrected text using ERRANT."""

    def __init__(self, language: str = 'en', min_length: int = 1):
        """
        Initialize the extractor with ERRANT annotator.

        Args:
            language: Language code for ERRANT (default: 'en')
            min_length: Minimum length of original text to be replaced (must be > 0)
        """
        self.annotator = errant.load(language)
        self.min_length = max(1, min_length)

    def extract_replacements(self, original: str, corrected: str, fix_tokenization=True) -> List[Match]:
        """
        Extract replacements needed to transform original text to corrected text.

        Args:
            original: The original text with errors
            corrected: The corrected text
            fix_tokenization: Ignored for compatibility (ERRANT handles tokenization)

        Returns:
            List of Match objects containing:
            - offset: int (start character index in original text)
            - length: int (length of text to replace in original)
            - message: str (error type from ERRANT)
            - suggested_replacements: List[SuggestedReplacement] (corrected text)
            - type: MatchType (hint for grammar errors)
        """
        # Parse texts with ERRANT
        orig_parsed = self.annotator.parse(original)
        cor_parsed = self.annotator.parse(corrected)
        
        # Get edits from ERRANT
        edits = self.annotator.annotate(orig_parsed, cor_parsed)
        
        matches = []
        
        for edit in edits:
            # Convert token indices to character indices
            offset, length = self._token_span_to_char_span(
                orig_parsed, edit.o_start, edit.o_end
            )
            
            # Get the replacement text (use character-based string from edit)
            replacement = edit.c_str
            
            # Handle insertions (where o_start == o_end, length == 0)
            # Convert to a replacement by including adjacent token
            if length == 0 and replacement.strip():
                # This is an insertion - need to convert to replacement with context
                offset, length, replacement = self._convert_insertion_to_replacement(
                    orig_parsed, original, edit.o_start, replacement
                )
            
            # Apply constraints:
            # - length must be >= min_length
            # - replacement cannot be empty (unless it's a deletion, which we skip)
            if length >= self.min_length and replacement.strip():
                # Create error message from ERRANT error type
                error_type = edit.type if edit.type else "UNKNOWN"
                original_text = edit.o_str if edit.o_str else f"[insert at position {edit.o_start}]"
                message = f"{error_type}: {original_text} -> {replacement}"
                
                matches.append(
                    Match(
                        message=message,
                        suggested_replacements=[
                            SuggestedReplacement(
                                replacement=replacement.strip(),
                                description=error_type
                            )
                        ],
                        offset=offset,
                        length=length,
                        type=MatchType.Other,  # Grammar errors typically map to "Other"
                        id=error_type
                    )
                )
        
        return matches

    def _token_span_to_char_span(self, parsed_doc, token_start: int, token_end: int) -> tuple[int, int]:
        """
        Convert token span to character span.

        Args:
            parsed_doc: Spacy Doc object from ERRANT
            token_start: Start token index
            token_end: End token index

        Returns:
            Tuple of (char_offset, char_length)
        """
        if token_start == token_end:
            # Empty span - use position at token_start
            if token_start < len(parsed_doc):
                offset = parsed_doc[token_start].idx
                return offset, 0
            else:
                # End of document
                if len(parsed_doc) > 0:
                    last_token = parsed_doc[-1]
                    offset = last_token.idx + len(last_token.text)
                else:
                    offset = 0
                return offset, 0
        
        # Get character offset of first token
        start_token = parsed_doc[token_start]
        offset = start_token.idx
        
        # Get character offset of last token + its length
        end_token = parsed_doc[token_end - 1]
        end_char = end_token.idx + len(end_token.text)
        
        # Calculate length
        length = end_char - offset
        
        return offset, length

    def _convert_insertion_to_replacement(self, parsed_doc, original_text: str, 
                                         insert_position: int, inserted_text: str) -> tuple[int, int, str]:
        """
        Convert an insertion to a replacement by including an adjacent token.
        
        Args:
            parsed_doc: Spacy Doc object from ERRANT
            insert_position: Token position where text should be inserted
            inserted_text: The text being inserted
            original_text: Original text string
            
        Returns:
            Tuple of (offset, length, replacement_with_context)
        """
        # Try to include the following token (preferred)
        if insert_position < len(parsed_doc):
            token = parsed_doc[insert_position]
            offset = token.idx
            length = len(token.text)
            # Build replacement: inserted text + space + following token
            replacement = f"{inserted_text} {token.text}"
            return offset, length, replacement
        
        # If at end, try previous token
        if insert_position > 0:
            token = parsed_doc[insert_position - 1]
            offset = token.idx
            length = len(token.text)
            # Build replacement: previous token + space + inserted text
            replacement = f"{token.text} {inserted_text}"
            return offset, length, replacement
        
        # Edge case: empty document or single insertion
        # Return the insertion at position 0
        return 0, 0, inserted_text
