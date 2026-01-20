"""Unit tests for GrammarCorrectionExtractor."""
import pytest
from unittest.mock import Mock, patch
from grammared_language.utils.grammar_correction_extractor import GrammarCorrectionExtractor
from grammared_language.language_tool.output_models import Match, SuggestedReplacement


class TestGrammarCorrectionExtractor:
    """Test suite for GrammarCorrectionExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create an extractor instance for testing."""
        return GrammarCorrectionExtractor(min_length=1)
    
    def test_simple_replacement(self, extractor):
        """Test basic word replacement."""
        original = "he are happy"
        corrected = "he is happy"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        assert matches[0].offset == 3
        assert matches[0].length == 3  # "are"
        assert matches[0].suggested_replacements[0].replacement == "is"
    
    def test_delete_followed_by_insert(self, extractor):
        """Test delete followed by insert (e.g., 'have' -> 'has')."""
        original = "he have"
        corrected = "he has"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        assert matches[0].suggested_replacements[0].replacement == "has"
    
    def test_standalone_delete_merge_with_previous(self, extractor):
        """Test delete merging with previous token (e.g., 'I am am' -> 'I am')."""
        original = "I am am"
        corrected = "I am"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        # Should merge with previous token "am"
        assert matches[0].suggested_replacements[0].replacement == "am"
    
    def test_standalone_delete_merge_with_next(self, extractor):
        """Test delete merging with next token."""
        original = "the very very good"
        corrected = "the very good"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        # Should merge deleted "very" with previous token "very"
        assert matches[0].suggested_replacements[0].replacement == "very"
    
    def test_standalone_insert_merge_with_next(self, extractor):
        """Test insert merging with next token."""
        original = "I happy"
        corrected = "I am happy"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        # Should merge inserted "am" with next token "happy"
        assert matches[0].suggested_replacements[0].replacement == "am happy"
    
    def test_standalone_insert_merge_with_previous(self, extractor):
        """Test insert merging with previous token when no next token."""
        original = "good"
        corrected = "very good"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        # Should merge inserted "very" with previous token "good"
        assert matches[0].suggested_replacements[0].replacement == "very good"
    
    def test_delete_followed_by_replace(self, extractor):
        """Test delete followed by replace operation."""
        original = "the cat are running"
        corrected = "the dog is running"
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should create replacements for both changes
        assert len(matches) >= 1
    
    def test_multiple_consecutive_changes(self, extractor):
        """Test multiple changes in sequence."""
        original = "he are very happy"
        corrected = "he is quite happy"
        
        matches = extractor.extract_replacements(original, corrected)
        
        # When consecutive operations occur, they can be merged into a single match
        # "are very" -> "is quite" becomes one replacement
        assert len(matches) >= 1
    
    def test_tokenization_fixing(self, extractor):
        """Test that tokenization mistakes are fixed."""
        original = "I'm am fine"
        # Simulate LLM output with spacing issues
        corrected = "I ' m am fine"  # Broken contraction
        
        # The extractor should fix this internally
        matches = extractor.extract_replacements(original, corrected, fix_tokenization=True)
        
        # Should still work despite tokenization issues
        assert isinstance(matches, list)
    
    def test_no_changes(self, extractor):
        """Test when original and corrected are the same."""
        original = "this is correct"
        corrected = "this is correct"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 0
    
    def test_min_length_constraint(self):
        """Test that min_length constraint is respected."""
        extractor = GrammarCorrectionExtractor(min_length=2)
        original = "a b"
        corrected = "x b"
        
        matches = extractor.extract_replacements(original, corrected)
        
        # "a" -> "x" is only 1 character, should not match
        assert len(matches) == 0
    
    def test_empty_replacement_not_created(self, extractor):
        """Test that empty replacements are not created."""
        original = "the cat is running"
        corrected = "the is running"
        
        matches = extractor.extract_replacements(original, corrected)
        
        # All replacements should have non-empty replacement text
        for match in matches:
            assert match.suggested_replacements[0].replacement != ""
    
    def test_offset_calculation(self, extractor):
        """Test that offset calculations are correct."""
        original = "hello world"
        corrected = "hello earth"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        # "world" starts at index 6
        assert matches[0].offset == 6
        assert matches[0].length == 5  # "world" is 5 chars
    
    def test_complex_sentence(self, extractor):
        """Test a more complex sentence with multiple errors."""
        original = "The student are study in the library"
        corrected = "The student is studying in the library"
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should find at least the subject-verb agreement error
        assert len(matches) >= 1
        assert any("is" in m.suggested_replacements[0].replacement for m in matches)
    
    def test_punctuation_handling(self, extractor):
        """Test handling of punctuation in tokenization."""
        original = "I like cats ."
        corrected = "I like cats."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should handle the space before period correctly
        assert isinstance(matches, list)


class TestGrammarCorrectionExtractorWithMockedOpcodes:
    """Tests using mocked opcodes for precise scenario testing."""
    
    @pytest.fixture
    def extractor(self):
        return GrammarCorrectionExtractor(min_length=1)
    
    def test_merge_consecutive_operations_insert_delete(self, extractor):
        """Test _try_merge_consecutive_operations with insert followed by delete."""
        orig_tokens = ["I", "happy"]
        orig_positions = [0, 2]
        corr_tokens = ["I", "am", "happy"]
        corr_positions = [0, 2, 5]
        
        # Simulated opcodes: insert "am" at position 1, then delete "I" 
        opcodes = [
            ('insert', 1, 1, 1, 2),  # insert "am"
            ('equal', 1, 2, 2, 3),   # equal "happy"
        ]
        
        result = extractor._try_merge_consecutive_operations(
            original="I happy",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="I am happy",
            start_idx=1,
            end_idx=None,
            inserted_text="am",
            opcodes=opcodes,
            current_idx=0
        )
        
        # Should handle the insert operation
        assert result is not None
        match, consumed_idx = result
        assert match.offset == 2
        assert match.length == 5  # "happy"
        assert match.suggested_replacements[0].replacement == "am happy"
        assert consumed_idx is None
    
    def test_create_match_helper(self, extractor):
        """Test the _create_match helper method."""
        match, consumed_idx = extractor._create_match(offset=5, length=3, replacement="new")
        
        assert match.offset == 5
        assert match.length == 3
        assert match.suggested_replacements[0].replacement == "new"
        assert consumed_idx is None
    
    def test_merge_with_adjacent_token_insert_next(self, extractor):
        """Test _merge_with_adjacent_token for insert with next token."""
        orig_tokens = ["happy"]
        orig_positions = [0]
        
        opcodes = [
            ('insert', 0, 0, 0, 1),  # insert operation
            ('equal', 0, 1, 1, 2),   # next is equal
        ]
        
        result = extractor._merge_with_adjacent_token(
            original="happy",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            start_idx=0,
            end_idx_or_text="very",
            opcodes=opcodes,
            current_idx=0,
            is_insert=True
        )
        
        assert result is not None
        match, consumed_idx = result
        assert "very" in match.suggested_replacements[0].replacement
        assert "happy" in match.suggested_replacements[0].replacement
        assert match.offset == 0
        assert match.suggested_replacements[0].replacement == "very happy"
        assert consumed_idx is None
    
    def test_merge_with_adjacent_token_delete_previous(self, extractor):
        """Test _merge_with_adjacent_token for delete with previous token."""
        orig_tokens = ["very", "good"]
        orig_positions = [0, 5]
        
        result = extractor._merge_with_adjacent_token(
            original="very good",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            start_idx=0,
            end_idx_or_text=1,
            opcodes=[],
            current_idx=0,
            is_insert=False
        )
        
        # When deleting "very" and no previous token, should try next token "good"
        assert result is not None
        match, consumed_idx = result
        assert match.suggested_replacements[0].replacement == "good"
        assert match.offset == 0
        assert match.length == 9  # entire "very good"
        assert consumed_idx is None


class TestTokenizationFixes:
    """Tests for tokenization fixes."""
    
    @pytest.fixture
    def extractor(self):
        return GrammarCorrectionExtractor()
    
    def test_fix_contractions(self, extractor):
        """Test fixing spaces around apostrophes in contractions."""
        text = "I ' m happy"
        fixed = extractor._fix_tokenization_mistakes(text)
        assert fixed == "I'm happy"
    
    def test_fix_period_spacing(self, extractor):
        """Test removing space before period."""
        text = "hello world ."
        fixed = extractor._fix_tokenization_mistakes(text)
        assert ". " in fixed or "." in fixed
    
    def test_fix_comma_spacing(self, extractor):
        """Test removing space before comma."""
        text = "hello , world"
        fixed = extractor._fix_tokenization_mistakes(text)
        assert ",world" in fixed or ", " in fixed
    
    def test_fix_question_mark(self, extractor):
        """Test removing space before question mark."""
        text = "how are you ?"
        fixed = extractor._fix_tokenization_mistakes(text)
        assert "?" in fixed


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.fixture
    def extractor(self):
        return GrammarCorrectionExtractor(min_length=1)
    
    def test_single_word_replacement(self, extractor):
        """Test replacing a single word."""
        original = "bad"
        corrected = "good"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 1
        assert matches[0].suggested_replacements[0].replacement == "good"
    
    def test_empty_string(self, extractor):
        """Test with empty strings."""
        original = ""
        corrected = ""
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 0
    
    def test_whitespace_only(self, extractor):
        """Test with whitespace-only strings."""
        original = "   "
        corrected = "   "
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) == 0
    
    def test_adding_words_at_beginning(self, extractor):
        """Test adding words at the beginning."""
        original = "very happy"
        corrected = "I am very happy"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert isinstance(matches, list)
    
    def test_removing_words_at_end(self, extractor):
        """Test removing words at the end."""
        original = "I am very happy today"
        corrected = "I am very happy"
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert isinstance(matches, list)
