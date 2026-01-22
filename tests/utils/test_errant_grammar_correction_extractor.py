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

    def test_separate_consecutive_errors(self):
        """Test that consecutive errors are detected separately, not combined."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "its there food"
        corrected = "It's their food"
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect exactly 2 separate errors
        assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
        
        # Extract the matched text segments
        matched_texts = [
            original[match.offset:match.offset + match.length].lower()
            for match in matches
        ]
        
        # Verify individual matches
        assert "its" in matched_texts, "Should match 'its' separately"
        assert "there" in matched_texts, "Should match 'there' separately"
        
        # Ensure no match spans both words (combined match)
        for match in matches:
            matched_text = original[match.offset:match.offset + match.length]
            assert "its there" not in matched_text.lower(), "Should not have combined 'its there' match"
            
        # Verify the replacements
        replacements = [match.suggested_replacements[0].replacement for match in matches]
        assert any("it's" in r.lower() or "it" in r.lower() for r in replacements), "Should suggest correction for 'its'"
        assert any("their" in r.lower() for r in replacements), "Should suggest 'their'"

    @pytest.mark.parametrize("original,corrected_with_errors,expected_fixed,description", [
        # Apostrophe spacing in contractions
        ("He said its wrong", "He said it 's wrong", "He said it's wrong", "it 's -> it's"),
        ("I am going", "I ' m going", "I'm going", "I ' m -> I'm"),
        ("I am here", "I 'm here", "I'm here", "I 'm -> I'm"),
        ("They are here", "They ' re here", "They're here", "They ' re -> They're"),
        ("We have done it", "We ' ve done it", "We've done it", "We ' ve -> We've"),
        ("She is there", "She 's there", "She's there", "She 's -> She's"),
        ("He would go", "He ' d go", "He'd go", "He ' d -> He'd"),
        ("I will go", "I 'll go", "I'll go", "I 'll -> I'll"),
        
        # Period spacing
        ("I like food", "I like food .", "I like food.", "food . -> food."),
        ("Hello world", "Hello world . ", "Hello world. ", "world . -> world."),
        
        # Comma spacing
        ("Yes I agree", "Yes , I agree", "Yes, I agree", "space , -> ,"),
        ("One two three", "One , two , three", "One, two, three", "multiple commas"),
        
        # Question mark spacing
        ("What is this", "What is this ?", "What is this?", "space ? -> ?"),
        
        # Exclamation mark spacing
        ("This is great", "This is great !", "This is great!", "space ! -> !"),
        
        # Semicolon spacing
        ("I agree however", "I agree ; however", "I agree; however", "space ; -> ;"),
        
        # Colon spacing
        ("Note the following", "Note the following :", "Note the following:", "space : -> :"),
        
        # Quote spacing
        ('He said hello', 'He said " hello "', 'He said " hello "', "quotes spacing"),
        
        # Hyphen spacing in compound words (should fix)
        ("well known fact", "well - known fact", "well-known fact", "space - space -> -"),
        ("up to date", "up - to - date", "up-to-date", "multiple hyphens"),
        
        # Edge cases: spaces that should be PRESERVED
        # Numeric ranges - spaces should NOT be removed
        ("pages ten to twenty", "pages 10 - 20", "pages 10 - 20", "numeric range preserved"),
        ("years", "2020 - 2025", "2020 - 2025", "year range preserved"),
        
        # Math operations - spaces should NOT be removed
        ("result", "5 - 3 = 2", "5 - 3 = 2", "math operation preserved"),
        ("sum", "x + y", "x + y", "addition preserved"),
        
        # Emoticons - spaces should NOT be removed
        ("happy face", ": )", ": )", "emoticon preserved"),
        ("sad face", ": (", ": (", "sad emoticon preserved"),
        
        # Multiple issues combined
        ("I am here", "I ' m here .", "I'm here.", "contraction + period"),
        ("She is happy", "She 's happy !", "She's happy!", "contraction + exclamation"),
    ])
    def test_tokenization_fixes_parametrized(self, original, corrected_with_errors, expected_fixed, description):
        """Test fixing various tokenization mistakes from generation models."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        # Test that the internal _fix_tokenization_mistakes method works correctly
        fixed = extractor._fix_tokenization_mistakes(corrected_with_errors)
        assert fixed == expected_fixed, f"Tokenization fix failed for {description}: got '{fixed}', expected '{expected_fixed}'"
        
        # Test that extract_replacements handles it properly with fix_tokenization=True
        matches = extractor.extract_replacements(original, corrected_with_errors, fix_tokenization=True)
        
        # Should handle the tokenization issues properly
        assert isinstance(matches, list), f"Failed for case: {description}"
        for match in matches:
            assert match.offset >= 0, f"Invalid offset for case: {description}"
            assert match.length >= 0, f"Invalid length for case: {description}"
            assert len(match.suggested_replacements) > 0, f"No replacements for case: {description}"

    def test_tokenization_fix_disabled(self):
        """Test that tokenization fixing can be disabled."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        original = "I am here"
        corrected = "I ' m here"  # Incorrectly tokenized
        
        # With tokenization fix enabled (default)
        matches_with_fix = extractor.extract_replacements(original, corrected, fix_tokenization=True)
        
        # With tokenization fix disabled
        matches_without_fix = extractor.extract_replacements(original, corrected, fix_tokenization=False)
        
        # Both should work but may produce different results
        assert isinstance(matches_with_fix, list)
        assert isinstance(matches_without_fix, list)
