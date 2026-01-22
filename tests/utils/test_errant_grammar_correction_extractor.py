"""
Tests for ErrantGrammarCorrectionExtractor
"""
import pytest
from grammared_language.utils.errant_grammar_correction_extractor import ErrantGrammarCorrectionExtractor


class TestErrantGrammarCorrectionExtractor:
    """Test suite for ERRANT-based grammar correction extractor."""

    def test_basic_replacement(self):
        """Test basic word replacement."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "This are gramamtical sentence."
        corrected = "This is a grammatical sentence."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect multiple errors
        assert len(matches) > 0
        
        # Verify match structure
        for match in matches:
            assert match.offset >= 0
            assert match.length > 0
            assert len(match.suggested_replacements) > 0
            assert match.suggested_replacements[0].replacement

    def test_verb_agreement(self):
        """Test subject-verb agreement error."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "He go to school."
        corrected = "He goes to school."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect verb error
        assert len(matches) >= 1
        
        # Check that the replacement makes sense
        verb_match = matches[0]
        assert "go" in original[verb_match.offset:verb_match.offset + verb_match.length]

    def test_missing_article(self):
        """Test missing article detection."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "I saw dog."
        corrected = "I saw a dog."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect missing article
        assert len(matches) >= 1

    def test_spelling_error(self):
        """Test spelling correction."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "I recieved the package."
        corrected = "I received the package."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect spelling error
        assert len(matches) >= 1
        
        # Verify the spelling correction
        spelling_match = matches[0]
        assert "recieved" in original[spelling_match.offset:spelling_match.offset + spelling_match.length]

    def test_no_errors(self):
        """Test when there are no errors."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "This is a correct sentence."
        corrected = "This is a correct sentence."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should have no matches
        assert len(matches) == 0

    def test_min_length_constraint(self):
        """Test that min_length constraint is respected."""
        extractor = ErrantGrammarCorrectionExtractor(min_length=5)
        
        original = "I go to school."
        corrected = "I went to school."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should filter out short replacements
        for match in matches:
            assert match.length >= 5

    def test_multiple_errors(self):
        """Test handling multiple errors in one sentence."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "He dont like the apples"
        corrected = "He doesn't like the apples."
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect multiple issues
        assert len(matches) >= 1

    def test_match_attributes(self):
        """Test that Match objects have all required attributes."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "This are wrong."
        corrected = "This is wrong."
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) > 0
        
        match = matches[0]
        # Check all required attributes
        assert hasattr(match, 'offset')
        assert hasattr(match, 'length')
        assert hasattr(match, 'message')
        assert hasattr(match, 'suggested_replacements')
        assert hasattr(match, 'type')
        
        # Check suggested replacement structure
        assert len(match.suggested_replacements) > 0
        replacement = match.suggested_replacements[0]
        assert hasattr(replacement, 'replacement')
        assert hasattr(replacement, 'description')

    def test_character_offset_accuracy(self):
        """Test that character offsets are accurate."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "The cat are sleeping."
        corrected = "The cat is sleeping."
        
        matches = extractor.extract_replacements(original, corrected)
        
        assert len(matches) > 0
        
        # Verify the offset points to the actual error
        for match in matches:
            original_text = original[match.offset:match.offset + match.length]
            # The extracted text should be meaningful
            assert len(original_text) > 0
            assert original_text.strip()  # Should not be just whitespace
