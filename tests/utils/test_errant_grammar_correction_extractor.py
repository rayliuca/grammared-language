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

    @pytest.mark.parametrize("original,corrected,num_tokens,should_skip,description", [
        # Short text (< 7 tokens) - should skip
        ("item one here", "Item one here", 3, True, "3 tokens - skip capitalization"),
        ("this is a short item list", "This is a short item list", 6, True, "6 tokens - skip capitalization"),
        # Long text (>= 7 tokens) - should detect
        ("this is a complete grammatical sentence here", "This is a complete grammatical sentence here", 8, False, "8 tokens - detect capitalization"),
        ("this is a very long sentence with many words", "This is a very long sentence with many words", 10, False, "10 tokens - detect capitalization"),
    ])
    def test_heuristic_sentence_start_capitalization(self, original, corrected, num_tokens, should_skip, description):
        """Test heuristic for sentence-start capitalization based on text length."""
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        if should_skip:
            assert len(matches) == 0, f"{description}: Should skip sentence-start capital changes"
        else:
            assert len(matches) > 0, f"{description}: Should detect sentence-start capital changes"
            # Verify the first match is about the first word
            assert matches[0].offset == 0, f"{description}: First match should be at the start"

    @pytest.mark.parametrize("original,corrected,num_tokens,should_skip,description", [
        # Short text (< 5 tokens) - should skip only if colon is at the END
        ("Item list", "Item list:", 2, True, "2 tokens - skip trailing colon"),
        ("Here are the items", "Here are the items:", 4, True, "4 tokens - skip trailing colon"),
        # Long text (>= 5 tokens) - should detect
        ("Here is the list of items", "Here is the list of items:", 6, False, "6 tokens - detect trailing colon"),
        ("This is a complete list of things", "This is a complete list of things:", 7, False, "7 tokens - detect trailing colon"),
        # Short text with colon NOT at the end - should detect
        ("Item list here", "Item: list here", 3, False, "3 tokens - colon in middle, should detect"),
        ("Say hello", "Say: hello", 2, False, "2 tokens - colon in middle, should detect"),
        # Short text where original ALREADY has colon at end - should NOT skip, but may not detect if no changes needed
        ("Item list:", "Item list:", 3, None, "3 tokens - original has colon, no change"),
        # Note: ERRANT won't generate edits for same text, so we can't test correction of existing colon properly here
    ])
    def test_heuristic_trailing_colon(self, original, corrected, num_tokens, should_skip, description):
        """Test heuristic for trailing colon additions based on text length.
        
        The heuristic only skips when:
        - Text is short (< 5 tokens)
        - Colon is at the END of corrected text (.endswith(':'))
        - Original doesn't already end with colon
        """
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        if should_skip is None:
            # No assertion - just checking it doesn't crash
            pass
        elif should_skip:
            assert len(matches) == 0, f"{description}: Should skip trailing colon"
        else:
            assert len(matches) > 0, f"{description}: Should detect trailing colon"

    @pytest.mark.parametrize("original,corrected,description", [
        # Trailing spaces replaced with colon - should skip
        ("Item list ", "Item list:", "trailing space to colon"),
        ("Item list  ", "Item list:", "multiple trailing spaces to colon"),
        ("Here are items   ", "Here are items:", "many trailing spaces to colon"),
        ("Note the following ", "Note the following:", "phrase with trailing space to colon"),
        # Non-trailing space cases - should detect
        ("Item list", "Item list:", "no trailing space - normal colon addition"),
        ("Item list x", "Item list: x", "colon in middle with content after"),
        # Cases where original part is not just spaces - should detect
        ("Item list.", "Item list:", "period to colon replacement"),
        ("Item list,", "Item list:", "comma to colon replacement"),
    ])
    def test_heuristic_trailing_spaces_to_colon(self, original, corrected, description):
        """Test heuristic for trailing spaces being replaced with colon.
        
        ML models are prone to suggest ':' at the end when users are still typing.
        The heuristic should skip when:
        - Edit is at the end of text
        - Original has trailing spaces
        - Corrected ends with ':'
        - Original part being replaced is just whitespace
        """
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        if original.rstrip() != original and corrected.endswith(':'):
            # Should skip if original has trailing spaces and corrected ends with colon
            assert len(matches) == 0, f"{description}: Should skip trailing spaces to colon"
        else:
            # Should detect normal colon additions/replacements
            # Note: may be 0 or more matches depending on other heuristics
            pass  # Just ensure it doesn't crash

    @pytest.mark.parametrize("original,corrected,num_tokens,should_skip,description", [
        # Short text (< 10 tokens) - should skip
        ("This is short", "This is short.", 3, True, "3 tokens - skip trailing period"),
        ("This is a somewhat longer sentence without period", "This is a somewhat longer sentence without period.", 9, True, "9 tokens - skip trailing period"),
        # Long text (>= 10 tokens) - should detect
        ("This is a complete sentence that is long enough to warrant a period", "This is a complete sentence that is long enough to warrant a period.", 14, False, "14 tokens - detect trailing period"),
        ("This is indeed a very long sentence that should definitely have a period at the end", "This is indeed a very long sentence that should definitely have a period at the end.", 17, False, "17 tokens - detect trailing period"),
    ])
    def test_heuristic_trailing_period(self, original, corrected, num_tokens, should_skip, description):
        """Test heuristic for trailing period additions based on text length."""
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        if should_skip:
            assert len(matches) == 0, f"{description}: Should skip trailing period"
        else:
            assert len(matches) > 0, f"{description}: Should detect trailing period"

    @pytest.mark.parametrize("original,corrected,description", [
        # Emoji-only changes that should be skipped
        ("Hello world", "Hello world 😊", "emoji addition"),
        ("Hello world 😊", "Hello world", "emoji removal"),
        ("Hello 😊", "Hello 😃", "emoji replacement"),
        ("Great work", "Great work 🎉🎊", "multiple emoji addition"),
        ("Test 🔥💯", "Test", "multiple emoji removal"),
        ("[🟢emoji test]", "[emoji test]", "emoji removal from bracketed text"),
        # ZWJ sequences (composite emojis)
        ("Hello world", "Hello world 👨‍💻", "ZWJ sequence addition - man technologist"),
        ("Team 👨‍💻", "Team", "ZWJ sequence removal"),
        ("Flag 🏴‍☠️", "Flag", "ZWJ sequence removal - pirate flag"),
        ("Family 👨‍👩‍👧‍👦", "Family", "complex ZWJ sequence removal"),
        # Skin tone modifiers
        ("Thanks", "Thanks 👍", "thumbs up without skin tone"),
        ("Thanks", "Thanks 👍🏽", "thumbs up with skin tone"),
        ("Thanks 👍", "Thanks 👍🏽", "skin tone change"),
        ("Thanks 👍🏻", "Thanks 👍🏿", "different skin tone"),
        # Variation selectors (emoji vs text presentation)
        ("Note", "Note ✅", "check mark emoji"),
        ("Hearts ❤️", "Hearts", "red heart with variation selector"),
        # Modern emojis (newer Unicode versions)
        ("Mood", "Mood 🫠", "melting face (Unicode 14)"),
        ("Salute 🫡", "Salute", "saluting face (Unicode 14)"),
        ("Love 🫶", "Love", "heart hands (Unicode 14)"),
    ])
    def test_heuristic_skip_emoji_only_changes(self, original, corrected, description):
        """Test that emoji-only changes are skipped."""
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        # Should skip emoji-only changes
        assert len(matches) == 0, f"Should skip {description}"

    def test_heuristic_allow_text_with_emoji_changes(self):
        """Test that changes including both text and emoji are not skipped."""
        extractor = ErrantGrammarCorrectionExtractor()
        
        # Test text change that also modifies emoji
        original = "I are happy"
        corrected = "I am happy 😊"  # Fix grammar AND add emoji
        
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect the grammar error (not skip because there's text change too)
        # The exact number may vary, but there should be at least one match for "are" -> "am"
        grammar_matches = [m for m in matches if 'are' in original[m.offset:m.offset + m.length].lower()]
        assert len(grammar_matches) > 0, "Should detect grammar errors even when emoji is also present"

    @pytest.mark.parametrize("original,corrected,description", [
        # Whitespace changes with emoji should be ignored
        ("🟢emoji test", " emoji test", "leading space added with emoji removal"),
        ("🟢emoji test", "emoji test ", "trailing space added with emoji removal"),
        # ("🟢emoji test", "Emoji test ", "capitalization, trailing space added with emoji removal"),
        ("Mood🫠", "Mood ", "emoji removed, space added"),
        ("Mood 🫠", "Mood", "emoji and space removed"),
        ("Mood 🫠", "Mood ", "emoji removed, space kept"),
        ("🟢 test", "test", "emoji and space both removed"),
        ("test 🟢", "test", "trailing emoji removed"),
        ("test  🟢", "test", "emoji with multiple spaces"),
        (" 🟢emoji", "emoji", "leading space + emoji removed"),
        ("text 🟢 more", "text more", "emoji with surrounding spaces removed"),
    ])
    def test_heuristic_skip_emoji_with_whitespace_changes(self, original, corrected, description):
        """Test that emoji removal with whitespace changes is still considered emoji-only."""
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        # Should skip emoji+whitespace-only changes
        assert len(matches) == 0, f"Should skip {description}"

    @pytest.mark.parametrize("original,corrected,description", [
        # Multiple heuristics (capitalization + period on short text)
        ("test item", "Test item.", "capital + period on 2 tokens"),
        ("hello world", "Hello world.", "capital + period on 2 tokens"),
        # Multiple heuristics (capitalization + colon on short text)
        ("item list", "Item list:", "capital + colon on 2 tokens"),
    ])
    def test_heuristic_multiple_heuristics_combined(self, original, corrected, description):
        """Test behavior when multiple heuristics could apply."""
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        # Multiple heuristics should fire - no matches
        assert len(matches) == 0, f"Should skip when multiple heuristics apply: {description}"

    @pytest.mark.parametrize("original,corrected,description", [
        # Mid-sentence changes on short text
        ("item list here", "item lists here", "plural change in middle - 3 tokens"),
        ("go back", "goes back", "verb change in middle - 2 tokens"),
        ("the big dog", "the bigger dog", "adjective change in middle - 3 tokens"),
    ])
    def test_heuristic_mid_sentence_changes_not_affected(self, original, corrected, description):
        """Test that heuristics don't affect mid-sentence changes."""
        extractor = ErrantGrammarCorrectionExtractor()
        matches = extractor.extract_replacements(original, corrected)
        
        # Should detect the mid-sentence change (not affected by start/end heuristics)
        assert len(matches) > 0, f"Should detect mid-sentence changes: {description}"
