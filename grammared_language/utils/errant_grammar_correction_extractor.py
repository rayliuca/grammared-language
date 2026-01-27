"""
Grammar Error Correction - Extract Replacements using ERRANT
Extracts replacements between original and corrected text using the ERRANT library.
"""
from typing import List
import re
import errant

from grammared_language.language_tool.output_models import Match, SuggestedReplacement, MatchType


ERROR_TYPE_LABELS = {
    "NOUN:POSS": "Noun Possessive",
    "CONTR": "Contraction",
    "VERB:FORM": "Verb Form",
    "VERB:TENSE": "Verb Tense",
    "NOUN": "Noun Choice",
    "VERB": "Verb Choice",
    "ADJ": "Adjective",
    "ADV": "Adverb",
    "PREP": "Preposition",
    "DET": "Determiner / Article",
    "PART": "Particle / Phrasal Verb",
    "PUNCT": "Punctuation",
    "SPACE": "Whitespace",
    "PRON": "Pronoun",
    "CONJ": "Conjunction",
    "NUM": "Number",
    "SYM": "Symbol",
    "X": "Other Token",
    "OTHER": "Other / Uncategorized",
    "ORTH": "Orthography (Case / Space)",
    "WO": "Word Order",
    "SPELL": "Spelling",
    "MORPH": "Morphology / Derivation",
    "ADJ:FORM": "Adjective Form",
    "NOUN:NUM": "Noun Number",
    "VERB:SVA": "Subject–Verb Agreement",
}

class ErrantGrammarCorrectionExtractor:
    """Extract replacement operations from original to corrected text using ERRANT."""

    def __init__(self, language: str = 'en', min_length: int = 1, rule_id:str="grammared_language") -> None:
        """
        Initialize the extractor with ERRANT annotator.

        Args:
            language: Language code for ERRANT (default: 'en')
            min_length: Minimum length of original text to be replaced (must be > 0)
        """
        self.annotator = errant.load(language)
        self.min_length = max(1, min_length)
        self.rule_id = rule_id

    def extract_replacements(self, original: str, corrected: str, fix_tokenization=True) -> List[Match]:
        """
        Extract replacements needed to transform original text to corrected text.

        Args:
            original: The original text with errors
            corrected: The corrected text
            fix_tokenization: Whether to fix common tokenization mistakes in corrected text

        Returns:
            List of Match objects containing:
            - offset: int (start character index in original text)
            - length: int (length of text to replace in original)
            - message: str (error type from ERRANT)
            - suggested_replacements: List[SuggestedReplacement] (corrected text)
            - type: MatchType (hint for grammar errors)
        """
        # Pre-process to fix common tokenization mistakes in corrected text
        if fix_tokenization:
            fixed_corrected = self._fix_tokenization_mistakes(corrected)
        else:
            fixed_corrected = corrected
        
        # Parse texts with ERRANT
        orig_parsed = self.annotator.parse(original)
        cor_parsed = self.annotator.parse(fixed_corrected)
        
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
            error_type = edit.type
            
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
                if ":" in error_type:
                    error_type = ":".join(error_type.split(":")[1:])
                else:
                    error_type = "OTHER"
                error_type = ERROR_TYPE_LABELS.get(error_type, error_type)
                original_text = edit.o_str if edit.o_str else f"[insert at position {edit.o_start}]"
                message = error_type
                
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
                        type=MatchType.Other,  # Grammar errors typically map to "Other". Tho this has no effect in LT.
                        id=self.rule_id
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
        # Pattern 1: letter + space(s) + ' + space(s) + letter (e.g., "I ' m")
        result = re.sub(r"(\w)\s+'\s+(\w)", r"\1'\2", result)
        # Pattern 2: letter + ' + space(s) + letter (e.g., "I 'm")
        result = re.sub(r"(\w)'\s+(\w)", r"\1'\2", result)
        # Pattern 3: letter + space(s) + ' + letter (e.g., "it 's" - less common)
        result = re.sub(r"(\w)\s+'(\w)", r"\1'\2", result)

        # Fix space before punctuation
        # Period - don't add extra space
        result = re.sub(r'\s+\.', '.', result)
        # Comma
        result = re.sub(r'\s+,', ',', result)
        # Question mark
        result = re.sub(r'\s+\?', '?', result)
        # Exclamation mark
        result = re.sub(r'\s+!', '!', result)
        # Semicolon
        result = re.sub(r'\s+;', ';', result)
        # Colon (but be careful not to break things like "http : //")
        result = re.sub(r'(\w)\s+:', r'\1:', result)

        # Fix space after opening quotes and before closing quotes
        # But preserve the space between the quote and the surrounding words
        result = re.sub(r'"\s+(\S)', r'" \1', result)
        result = re.sub(r'(\S)\s+"', r'\1 "', result)

        # Fix spaces around hyphens in compound words (letter-letter only)
        # This preserves numeric ranges (10 - 20) and math operations (5 - 3)
        result = re.sub(r'([a-zA-Z])\s+-\s+([a-zA-Z])', r'\1-\2', result)

        return result
