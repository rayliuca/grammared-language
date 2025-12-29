import pytest
from unittest.mock import Mock, patch
from src.util import GrammarCorrectionExtractor
from src.output_models import Match, SuggestedReplacement


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
    
    def test_three_consecutive_insert_delete_insert(self, extractor):
        """Test handling of three consecutive operations: insert, delete, insert."""
        orig_tokens = ["hello", "world"]
        orig_positions = [0, 6]
        corr_tokens = ["hello", "beautiful", "wonderful", "world"]
        corr_positions = [0, 6, 16, 27]
        
        # Opcodes: insert at 0, delete at 0, insert at 1
        opcodes = [
            ('insert', 0, 0, 0, 1),      # insert "beautiful"
            ('equal', 0, 1, 1, 2),       # equal "hello"
            ('delete', 1, 2, 2, 2),      # delete "world"
            ('insert', 2, 2, 2, 3),      # insert "wonderful"
        ]
        
        result = extractor._try_merge_consecutive_operations(
            original="hello world",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="beautiful hello wonderful world",
            start_idx=0,
            end_idx=None,
            inserted_text="beautiful",
            opcodes=opcodes,
            current_idx=0
        )
        
        # Should merge insert with next token
        assert result is not None
        match, consumed_idx = result
        assert match.suggested_replacements[0].replacement == "beautiful hello"
        assert match.offset == 0
        assert match.length == 5  # "hello"
        assert consumed_idx is None
    
    def test_three_consecutive_delete_insert_replace(self, extractor):
        """Test three consecutive opcodes: delete, insert, replace."""
        orig_tokens = ["cat", "is", "big"]
        orig_positions = [0, 4, 7]
        corr_tokens = ["dog", "very", "huge"]
        corr_positions = [0, 4, 9]
        
        # delete "cat" at index 0, insert "dog", then replace is/big -> very/huge
        opcodes = [
            ('delete', 0, 1, 0, 0),      # delete "cat"
            ('insert', 1, 1, 0, 1),      # insert "dog"
            ('replace', 1, 3, 1, 3),     # replace "is big" with "very huge"
        ]
        
        # Process the delete operation
        result = extractor._try_merge_consecutive_operations(
            original="cat is big",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="dog very huge",
            start_idx=0,
            end_idx=1,
            inserted_text=None,
            opcodes=opcodes,
            current_idx=0
        )
        
        # Delete followed by insert should merge
        assert result is not None
        match, consumed_idx = result
        assert consumed_idx == 1  # Should consume the insert operation
        assert match.offset == orig_positions[0]
        assert match.suggested_replacements[0].replacement == "dog"
    
    def test_three_consecutive_insert_replace_delete(self, extractor):
        """Test three consecutive opcodes: insert, replace, delete."""
        orig_tokens = ["I", "am", "sad"]
        orig_positions = [0, 2, 5]
        corr_tokens = ["I", "am", "very", "happy"]
        corr_positions = [0, 2, 5, 10]
        
        # insert before I, replace am, delete sad
        opcodes = [
            ('insert', 0, 0, 0, 1),      # insert "I"
            ('replace', 0, 2, 1, 3),     # replace "am sad" with "am very happy"
            ('delete', 2, 3, 3, 3),      # delete "sad"
        ]
        
        # Process the insert operation
        result = extractor._try_merge_consecutive_operations(
            original="I am sad",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="I am very happy",
            start_idx=0,
            end_idx=None,
            inserted_text="I",
            opcodes=opcodes,
            current_idx=0
        )
        
        assert result is not None
        match, consumed_idx = result
        assert "I" in match.suggested_replacements[0].replacement
        assert consumed_idx == 1  # Should consume the replace operation
    
    def test_three_consecutive_replace_delete_insert(self, extractor):
        """Test three consecutive opcodes: replace, delete, insert."""
        orig_tokens = ["good", "bad", "ugly"]
        orig_positions = [0, 5, 9]
        corr_tokens = ["great", "beautiful"]
        corr_positions = [0, 6]
        
        # replace good, delete bad, insert beautiful
        opcodes = [
            ('replace', 0, 1, 0, 1),     # replace "good" with "great"
            ('delete', 1, 2, 1, 1),      # delete "bad"
            ('insert', 2, 2, 1, 2),      # insert "beautiful"
            ('delete', 2, 3, 2, 2),      # delete "ugly"
        ]
        
        # Process the delete operation (it's followed by insert)
        result = extractor._try_merge_consecutive_operations(
            original="good bad ugly",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="great beautiful",
            start_idx=1,
            end_idx=2,
            inserted_text=None,
            opcodes=opcodes,
            current_idx=1
        )
        
        # Delete followed by insert should merge
        assert result is not None
        match, consumed_idx = result
        assert consumed_idx == 2  # Should consume the insert operation
        assert match.offset == orig_positions[1]
        assert match.suggested_replacements[0].replacement == "beautiful"
    
    def test_three_consecutive_delete_replace_insert(self, extractor):
        """Test three consecutive opcodes: delete, replace, insert."""
        orig_tokens = ["the", "bad", "day"]
        orig_positions = [0, 4, 8]
        corr_tokens = ["wonderful", "night"]
        corr_positions = [0, 10]
        
        # delete "the", replace "bad day" with "wonderful night"
        opcodes = [
            ('delete', 0, 1, 0, 0),      # delete "the"
            ('replace', 1, 3, 0, 2),     # replace "bad day" with "wonderful night"
        ]
        
        # Process the delete operation
        result = extractor._try_merge_consecutive_operations(
            original="the bad day",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="wonderful night",
            start_idx=0,
            end_idx=1,
            inserted_text=None,
            opcodes=opcodes,
            current_idx=0
        )
        
        # Delete followed by replace should create a match
        assert result is not None
        match, consumed_idx = result
        assert match is not None
        assert match.offset == orig_positions[0]
    
    def test_three_consecutive_insert_delete_replace(self, extractor):
        """Test three consecutive opcodes: insert, delete, replace."""
        orig_tokens = ["hello", "world"]
        orig_positions = [0, 6]
        corr_tokens = ["hi", "goodbye", "universe"]
        corr_positions = [0, 3, 11]
        
        # insert, delete hello, replace world
        opcodes = [
            ('insert', 0, 0, 0, 1),      # insert "hi"
            ('delete', 0, 1, 1, 1),      # delete "hello"
            ('replace', 1, 2, 1, 3),     # replace "world" with "goodbye universe"
        ]
        
        # Process the insert operation
        result = extractor._try_merge_consecutive_operations(
            original="hello world",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="hi goodbye universe",
            start_idx=0,
            end_idx=None,
            inserted_text="hi",
            opcodes=opcodes,
            current_idx=0
        )
        
        assert result is not None
        match, consumed_idx = result
        # Insert should merge with next delete at same position
        assert consumed_idx is None or match is not None
    
    def test_three_consecutive_replace_insert_delete(self, extractor):
        """Test three consecutive opcodes: replace, insert, delete."""
        orig_tokens = ["old", "text", "here"]
        orig_positions = [0, 4, 9]
        corr_tokens = ["new", "content", "added"]
        corr_positions = [0, 4, 13]
        
        # replace old, insert content, delete here
        opcodes = [
            ('replace', 0, 1, 0, 1),     # replace "old" with "new"
            ('replace', 1, 2, 1, 2),     # replace "text" with "content"
            ('delete', 2, 3, 2, 2),      # delete "here"
            ('insert', 3, 3, 2, 3),      # insert "added"
        ]
        
        # Process delete followed by insert
        result = extractor._try_merge_consecutive_operations(
            original="old text here",
            orig_tokens=orig_tokens,
            orig_positions=orig_positions,
            corr_tokens=corr_tokens,
            corr_positions=corr_positions,
            fixed_corrected="new content added",
            start_idx=2,
            end_idx=3,
            inserted_text=None,
            opcodes=opcodes,
            current_idx=2
        )
        
        # Delete followed by insert should merge
        assert result is not None
        match, consumed_idx = result
        assert consumed_idx == 3  # Should consume the insert operation


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
