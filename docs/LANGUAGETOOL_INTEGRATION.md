# LanguageTool Integration Guide

## Overview

This document explains how to integrate the Grammared Language backend with LanguageTool as a remote rule service.

## What is LanguageTool?

LanguageTool is an open-source grammar, style, and spell checker that supports multiple languages. It provides a plugin architecture that allows external services to provide additional grammar checking rules.

## Remote Rules

LanguageTool supports "remote rules" which allow an external HTTP service to provide grammar checking results. This is how the Grammared Language backend integrates with LanguageTool.

## Integration Architecture

```
LanguageTool (Client/Plugin)
        ↓ HTTP POST
    API Service (This Project)
        ↓ gRPC/HTTP
    Triton Inference Server
        ↓
    ML Models (BERT/LLM)
        ↓
    Grammar Corrections
        ↓ Formatted Response
    LanguageTool (Client/Plugin)
```

## LanguageTool Remote Rule Configuration

### 1. Configure LanguageTool to Use Remote Rules

In your LanguageTool configuration (e.g., `server.properties` or client settings):

```properties
# Enable remote rules
remoteRules.enabled=true

# Configure remote rule endpoint
remoteRules.url=http://localhost:8000/check

# Optional: Timeout for remote rule requests
remoteRules.timeout=5000
```

### 2. For LanguageTool Browser Extension

If using the browser extension, configure remote rules in the extension settings:

1. Open LanguageTool extension settings
2. Navigate to "Experimental Settings" or "Advanced"
3. Add remote rule URL: `http://localhost:8000/check`

### 3. For LanguageTool Desktop/Server

Edit the configuration file:

**Linux/Mac**: `~/.languagetool/server.properties`
**Windows**: `%USERPROFILE%\.languagetool\server.properties`

Add:
```properties
remoteRules=http://localhost:8000/check
```

## API Request/Response Format

### Request Format

LanguageTool sends requests in this format:

```json
{
  "text": "The text too check for errors.",
  "language": "en-US",
  "offset": 0,
  "length": 28
}
```

**Parameters:**
- `text` (required): The text to check
- `language` (required): Language code (e.g., "en-US", "en-GB", "de-DE")
- `offset` (optional): Character offset where checking should start
- `length` (optional): Number of characters to check

### Response Format

The API must return matches in LanguageTool's expected format:

```json
{
  "matches": [
    {
      "message": "Possible typo detected. Did you mean 'to'?",
      "shortMessage": "Typo",
      "replacements": [
        {"value": "to"}
      ],
      "offset": 9,
      "length": 3,
      "context": {
        "text": "The text too check for errors.",
        "offset": 9,
        "length": 3
      },
      "rule": {
        "id": "ML_GRAMMAR_CHECK",
        "description": "ML-powered grammar checker",
        "issueType": "misspelling",
        "category": {
          "id": "TYPOS",
          "name": "Possible Typo"
        }
      },
      "type": {
        "typeName": "Other"
      }
    }
  ]
}
```

**Match Object Fields:**
- `message`: Detailed error description
- `shortMessage`: Brief error description
- `replacements`: Array of suggested corrections
- `offset`: Character position where error starts
- `length`: Length of the error in characters
- `context`: Context showing the error in the text
- `rule`: Information about the rule that detected the error
  - `id`: Unique rule identifier
  - `description`: Human-readable rule description
  - `issueType`: Type of issue (misspelling, grammar, style, etc.)
  - `category`: Category information

## Error Types and Categories

Map your ML model outputs to these LanguageTool categories:

### Issue Types
- `misspelling`: Spelling errors
- `typographical`: Typos
- `grammar`: Grammar errors
- `style`: Style suggestions
- `redundancy`: Redundant words
- `consistency`: Consistency issues
- `punctuation`: Punctuation errors

### Categories
- `TYPOS`: Spelling and typos
- `GRAMMAR`: Grammar errors
- `PUNCTUATION`: Punctuation issues
- `STYLE`: Style suggestions
- `REDUNDANCY`: Redundant expressions
- `WORD_CHOICE`: Incorrect word usage

## Example Implementation

### Python API Endpoint Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()


class GrammarRequest(BaseModel):
    text: str
    language: str
    offset: Optional[int] = 0
    length: Optional[int] = None


class Replacement(BaseModel):
    value: str


class Context(BaseModel):
    text: str
    offset: int
    length: int


class Category(BaseModel):
    id: str
    name: str


class Rule(BaseModel):
    id: str
    description: str
    issueType: str = "grammar"
    category: Category


class Match(BaseModel):
    message: str
    shortMessage: str
    replacements: List[Replacement]
    offset: int
    length: int
    context: Context
    rule: Rule


class GrammarResponse(BaseModel):
    matches: List[Match]


@app.post("/check", response_model=GrammarResponse)
async def check_grammar(request: GrammarRequest):
    """
    Check text for grammar errors.
    
    This endpoint implements LanguageTool's remote rule format.
    """
    # Extract text to check
    text = request.text
    if request.length:
        text = text[request.offset:request.offset + request.length]
    
    # Call ML model via Triton (implementation needed)
    # errors = await get_errors_from_model(text, request.language)
    
    # Example response
    matches = [
        Match(
            message="Possible typo detected. Did you mean 'to'?",
            shortMessage="Typo",
            replacements=[Replacement(value="to")],
            offset=9,
            length=3,
            context=Context(
                text=request.text,
                offset=9,
                length=3
            ),
            rule=Rule(
                id="ML_TYPO_DETECTION",
                description="ML-based typo detection",
                issueType="misspelling",
                category=Category(
                    id="TYPOS",
                    name="Possible Typo"
                )
            )
        )
    ]
    
    return GrammarResponse(matches=matches)
```

## Testing Integration

### 1. Test with cURL

```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The text too check.",
    "language": "en-US"
  }'
```

### 2. Test with LanguageTool CLI

```bash
# Install LanguageTool CLI
wget https://languagetool.org/download/LanguageTool-stable.zip
unzip LanguageTool-stable.zip

# Run with remote rules
java -jar languagetool-commandline.jar \
  --language en-US \
  --remote-rules http://localhost:8000/check \
  test.txt
```

### 3. Test with Browser Extension

1. Install LanguageTool browser extension
2. Configure remote rule URL in settings
3. Test on any web page text field

## Performance Considerations

### Timeouts

LanguageTool expects responses within a few seconds. Configure your service accordingly:

```yaml
# API configuration
request_timeout: 5  # seconds
triton_timeout: 4   # leave buffer for API processing
```

### Batching

For better performance, implement request batching when possible:

```python
# Batch multiple small requests
if len(pending_requests) >= batch_size:
    results = await process_batch(pending_requests)
```

### Caching

Cache results for identical text:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_result(text: str, language: str):
    # Return cached result if available
    pass
```

## Multi-language Support

To support multiple languages:

1. Train models for each language
2. Route requests based on language parameter
3. Map language codes correctly:

```python
LANGUAGE_TO_MODEL = {
    "en-US": "bert_english_us",
    "en-GB": "bert_english_gb",
    "de-DE": "bert_german",
    "fr-FR": "bert_french",
}

def get_model_for_language(language: str) -> str:
    return LANGUAGE_TO_MODEL.get(language, "bert_english_us")
```

## Security Considerations

### Rate Limiting

Implement rate limiting to prevent abuse:

```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/check")
@limiter.limit("100/minute")
async def check_grammar(request: Request, grammar_request: GrammarRequest):
    # Implementation
    pass
```

### Input Validation

Validate and sanitize input:

```python
MAX_TEXT_LENGTH = 10000

def validate_request(request: GrammarRequest):
    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(400, f"Text too long (max {MAX_TEXT_LENGTH} chars)")
    
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")
```

## Monitoring and Logging

Track usage and errors:

```python
import logging

logger = logging.getLogger(__name__)

@app.post("/check")
async def check_grammar(request: GrammarRequest):
    logger.info(f"Grammar check request: language={request.language}, length={len(request.text)}")
    
    try:
        result = await process_grammar_check(request)
        logger.info(f"Found {len(result.matches)} issues")
        return result
    except Exception as e:
        logger.error(f"Grammar check failed: {str(e)}")
        raise
```

## Troubleshooting

### Common Issues

**Issue: LanguageTool not detecting remote rules**
- Verify URL is accessible from LanguageTool
- Check LanguageTool configuration
- Ensure API is running and responding

**Issue: Slow response times**
- Optimize model inference
- Implement caching
- Use dynamic batching in Triton
- Consider model quantization

**Issue: Incorrect match positions**
- Ensure offset calculation is correct
- Account for offset parameter in request
- Test with various text lengths

## References

- [LanguageTool HTTP API Documentation](https://languagetool.org/http-api/)
- [LanguageTool Development Documentation](https://dev.languagetool.org/)
- [Remote Rules Configuration](https://dev.languagetool.org/http-server)

## Next Steps

1. Implement the `/check` endpoint
2. Integrate with Triton Inference Server
3. Convert ML model outputs to LanguageTool format
4. Add comprehensive error handling
5. Test with actual LanguageTool clients
6. Deploy and monitor
