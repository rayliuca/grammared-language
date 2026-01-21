"""
Example usage of Text2TextBaseClient for grammar correction with Triton models.

This example demonstrates how to use the Text2TextBaseClient to interact with
text-to-text generation models deployed on Triton Inference Server, such as
CoEdit, T5, FLAN-T5, or other grammar correction models.

Requirements:
- Triton Inference Server running on localhost:8000
- A text-to-text model deployed (e.g., coedit_large)
- tritonclient[all] installed
"""

from grammared_language.clients.text2text_base_client import Text2TextBaseClient


def example_basic_usage():
    """Basic usage without prompt template."""
    print("=" * 60)
    print("Example 1: Basic Usage (No Prompt Template)")
    print("=" * 60)
    
    # Initialize client
    client = Text2TextBaseClient(
        model_name="coedit_large",
        triton_host="localhost",
        triton_port=8000
    )
    
    # Test sentence with grammar error
    text = "She go to the store yesterday."
    print(f"\nOriginal: {text}")
    
    # Get corrections
    result = client.predict(text)
    
    print(f"Language: {result.language}")
    print(f"Matches found: {len(result.matches)}")
    
    for i, match in enumerate(result.matches):
        original = text[match.offset:match.offset + match.length]
        print(f"\n  Match {i+1}:")
        print(f"    Position: {match.offset}-{match.offset + match.length}")
        print(f"    Original: '{original}'")
        if match.suggestedReplacements:
            suggestions = [r.replacement for r in match.suggestedReplacements]
            print(f"    Suggestions: {suggestions}")


def example_with_prompt_template():
    """Usage with custom prompt template."""
    print("\n" + "=" * 60)
    print("Example 2: With Prompt Template")
    print("=" * 60)
    
    # Initialize client with prompt template
    # The model will receive "Fix grammar: {your text here}"
    client = Text2TextBaseClient(
        model_name="coedit_large",
        triton_host="localhost",
        triton_port=8000,
        prompt_template="Fix grammar: {text}"
    )
    
    # Test sentence
    text = "This are a test."
    print(f"\nOriginal: {text}")
    
    # Get corrections using __call__ method
    result = client(text)
    
    print(f"Matches found: {len(result.matches)}")
    for match in result.matches:
        original = text[match.offset:match.offset + match.length]
        suggestions = [r.replacement for r in match.suggestedReplacements] if match.suggestedReplacements else []
        print(f"  '{original}' → {suggestions}")


def example_multiple_sentences():
    """Process multiple sentences."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple Sentences")
    print("=" * 60)
    
    client = Text2TextBaseClient(
        model_name="coedit_large",
        triton_host="localhost",
        triton_port=8000,
        prompt_template="Fix grammar: {text}"
    )
    
    sentences = [
        "I has a car.",
        "They was happy.",
        "She don't like it.",
    ]
    
    for i, text in enumerate(sentences, 1):
        print(f"\nSentence {i}: {text}")
        result = client.predict(text)
        print(f"  Matches: {len(result.matches)}")
        
        if result.matches:
            for match in result.matches:
                original = text[match.offset:match.offset + match.length]
                suggestions = [r.replacement for r in match.suggestedReplacements] if match.suggestedReplacements else []
                print(f"    '{original}' → {suggestions}")


def example_custom_model():
    """Use with a different text-to-text model."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Model Configuration")
    print("=" * 60)
    
    # Example with different model and custom I/O names
    client = Text2TextBaseClient(
        model_name="my_custom_t5_model",
        triton_host="localhost",
        triton_port=8000,
        triton_model_version="1",
        input_name="text_input",  # Match your model's input name
        output_name="text_output",  # Match your model's output name
        prompt_template="grammar: {text}"  # T5-style prompt
    )
    
    text = "I can has cheezburger?"
    print(f"\nOriginal: {text}")
    print("(This example requires 'my_custom_t5_model' to be deployed)")
    
    # Uncomment when you have the model deployed:
    # result = client.predict(text)
    # print(f"Matches: {result.matches}")


if __name__ == "__main__":
    print("Text2TextBaseClient Examples")
    print("=" * 60)
    print("\nNote: These examples require a running Triton Inference Server")
    print("with the coedit_large model deployed.\n")
    
    try:
        # Run examples
        example_basic_usage()
        example_with_prompt_template()
        example_multiple_sentences()
        example_custom_model()
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\nError: Missing required package - {e}")
        print("Install with: pip install tritonclient[all]")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Triton Inference Server is running with the model deployed:")
        print("  docker-compose up triton")
