"""
Grammar Error Correction - Extract Replacements
Extracts replacements between original and corrected text for grammar correction.
"""
from typing import List, Tuple
import re
from difflib import SequenceMatcher

from grammared_language.language_tool.output_models import Match, SuggestedReplacement, MatchType, SuggestionType

class GrammarCorrectionExtractor:
    """Extract replacement operations from original to corrected text."""

    def __init__(self, min_length: int = 1):
        """
        Initialize the extractor.

        Args:
            min_length: Minimum length of original text to be replaced (must be > 0)
        """
        self.min_length = max(1, min_length)  # Ensure at least 1

    def extract_replacements(self, original: str, corrected: str, fix_tokenization=True) -> list[Match]:
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
        opcodes = list(matcher.get_opcodes())

        # Track which opcodes have been processed to avoid double-processing
        consumed_indices = set()

        for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
            # Skip if this opcode was already consumed by a previous operation
            if idx in consumed_indices:
                continue

            if tag == 'replace':
                # Get character positions
                offset = orig_positions[i1]
                end_pos = orig_positions[i2 - 1] + len(orig_tokens[i2 - 1])
                length = end_pos - offset
                mistake = {"offset": offset, "length": length, "text": original[offset:end_pos]}
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
                            message=replacement,
                            suggested_replacements=[
                                SuggestedReplacement(
                                    replacement=replacement,
                                    # type=SuggestionType.Translation
                                )
                            ],
                            offset=offset,
                            length=length,

                            # /** Spelling errors, typically red. */
                            # UnknownWord = 0;
                            # /** Style errors, typically light blue. */
                            # Hint = 1;
                            # /** Other errors (including grammar), typically yellow/orange. */
                            # Other = 2;
                            # !!! not used/ accepted by LanguageTool Server
                            type=MatchType.Hint,
                            # rule=Rule(
                            #     id=rule_id,
                            #     description=rule_description
                            # )
                        )
                    )

            elif tag == 'insert':
                # Convert insertion to replacement by including adjacent token
                # or by merging with a following delete operation

                # Get the inserted text
                repl_start = corr_positions[j1]
                repl_end = corr_positions[j2 - 1] + len(corr_tokens[j2 - 1])
                inserted_text = fixed_corrected[repl_start:repl_end]

                # Try to merge with consecutive operations (delete/replace)
                converted = self._try_merge_consecutive_operations(
                    original, orig_tokens, orig_positions, corr_tokens, corr_positions,
                    fixed_corrected, i1, None, inserted_text, opcodes, idx
                )

                if converted:
                    match, consumed_idx = converted
                    matches.append(match)
                    if consumed_idx is not None:
                        consumed_indices.add(consumed_idx)

            elif tag == 'delete':
                # Try to merge delete with following insert, or with adjacent token
                converted = self._try_merge_consecutive_operations(
                    original, orig_tokens, orig_positions, corr_tokens, corr_positions,
                    fixed_corrected, i1, i2, None, opcodes, idx
                )

                if converted:
                    match, consumed_idx = converted
                    matches.append(match)
                    if consumed_idx is not None:
                        consumed_indices.add(consumed_idx)

        return matches

    def _try_merge_consecutive_operations(self, original: str, orig_tokens: List[str],
                                          orig_positions: List[int], corr_tokens: List[str],
                                          corr_positions: List[int], fixed_corrected: str,
                                          start_idx: int, end_idx: int, inserted_text: str,
                                          opcodes: List, current_idx: int) -> tuple:
        """
        Attempt to merge consecutive operations or with adjacent tokens.
        Handles:
        - Insert + next delete/replace -> include next token
        - Delete + next insert/replace -> merge into replacement
        - Standalone insert/delete -> merge with adjacent token

        Args:
            original: Original text
            orig_tokens: Original tokens
            orig_positions: Original token positions
            corr_tokens: Corrected tokens
            corr_positions: Corrected token positions
            fixed_corrected: Fixed corrected text
            start_idx: Start token index in original
            end_idx: End token index in original (None for insert)
            inserted_text: Text being inserted (None for delete)
            opcodes: All opcodes from SequenceMatcher
            current_idx: Current opcode index

        Returns:
            Tuple of (Match object, consumed_opcode_index) or None if no merge possible
        """
        is_insert = end_idx is None
        is_delete = inserted_text is None

        # Check if there's a next operation we can merge with
        if current_idx + 1 < len(opcodes):
            next_tag, next_i1, next_i2, next_j1, next_j2 = opcodes[current_idx + 1]

            # Case 1: insert followed by delete/replace at the same position
            if is_insert and next_tag in ('delete', 'replace') and next_i1 == start_idx:
                next_token = orig_tokens[start_idx]
                replacement = inserted_text + " " + next_token
                match = self._create_match(
                    orig_positions[start_idx], len(next_token), replacement
                )[0]
                return (match, current_idx + 1)

            # Case 2: delete followed by insert/replace at the same position
            if is_delete and next_tag in ('insert', 'replace') and next_i1 == end_idx:
                offset = orig_positions[start_idx]
                end_pos = orig_positions[end_idx - 1] + len(orig_tokens[end_idx - 1])
                length = end_pos - offset

                # Get the replacement text from corrected
                repl_start = corr_positions[next_j1]
                repl_end = corr_positions[next_j2 - 1] + len(corr_tokens[next_j2 - 1])
                replacement = fixed_corrected[repl_start:repl_end]

                if length >= self.min_length and replacement != "":
                    match = self._create_match(offset, length, replacement)[0]
                    return (match, current_idx + 1)

        # No consecutive merge possible, try merging with adjacent tokens
        if is_insert:
            return self._merge_with_adjacent_token(
                original, orig_tokens, orig_positions,
                start_idx, inserted_text, opcodes, current_idx, is_insert=True
            )
        else:
            return self._merge_with_adjacent_token(
                original, orig_tokens, orig_positions,
                start_idx, end_idx, opcodes, current_idx, is_insert=False
            )

    def _create_match(self, offset: int, length: int, replacement: str) -> tuple:
        """Helper to create a Match object with the given parameters."""
        return (Match(
            message=replacement,
            suggested_replacements=[
                SuggestedReplacement(replacement=replacement)
            ],
            offset=offset,
            length=length,
        ), None)

    def _merge_with_adjacent_token(self, original: str, orig_tokens: List[str],
                                   orig_positions: List[int], start_idx: int,
                                   end_idx_or_text: int, opcodes: List, current_idx: int,
                                   is_insert: bool) -> tuple:
        """
        Merge an insert or delete with an adjacent token to create a valid replacement.

        Args:
            original: Original text
            orig_tokens: Original tokens
            orig_positions: Original token positions
            start_idx: Start token index
            end_idx_or_text: End index (for delete) or inserted text (for insert)
            opcodes: All opcodes
            current_idx: Current opcode index
            is_insert: True if this is an insert, False if delete

        Returns:
            Tuple of (Match, consumed_idx) or None
        """
        if is_insert:
            inserted_text = end_idx_or_text
            # Try next token first
            if start_idx < len(orig_tokens):
                next_token_used = False
                if current_idx + 1 < len(opcodes):
                    next_tag, next_i1, _, _, _ = opcodes[current_idx + 1]
                    if next_tag in ('replace', 'delete') and next_i1 == start_idx:
                        next_token_used = True

                if not next_token_used:
                    next_token = orig_tokens[start_idx]
                    replacement = inserted_text + " " + next_token
                    return self._create_match(
                        orig_positions[start_idx], len(next_token), replacement
                    )

            # Try previous token
            if start_idx > 0:
                prev_token = orig_tokens[start_idx - 1]
                replacement = prev_token + " " + inserted_text
                return self._create_match(
                    orig_positions[start_idx - 1], len(prev_token), replacement
                )
        else:
            # Delete case
            del_end = end_idx_or_text
            # Try previous token
            if start_idx > 0:
                prev_idx = start_idx - 1
                prev_token = orig_tokens[prev_idx]
                end_pos = orig_positions[del_end - 1] + len(orig_tokens[del_end - 1])
                length = end_pos - orig_positions[prev_idx]

                if length >= self.min_length:
                    return self._create_match(
                        orig_positions[prev_idx], length, prev_token
                    )

            # Try next token
            if del_end < len(orig_tokens):
                offset = orig_positions[start_idx]
                next_token = orig_tokens[del_end]
                end_pos = orig_positions[del_end] + len(next_token)
                length = end_pos - offset

                if length >= self.min_length:
                    return self._create_match(offset, length, next_token)

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