# API Documentation

## Overview

The API service provides a RESTful interface compatible with LanguageTool's remote rule format, allowing LanguageTool clients to use ML-powered grammar checking.

## LanguageTool Remote Rule Integration

LanguageTool supports remote rules, which allow external services to provide grammar checking. This API implements that interface.

### Remote Rule Format

LanguageTool expects a specific JSON format for requests and responses:

**Request Format:**
```json
{
  "text": "The text too check for errors.",
  "language": "en-US",
  "offset": 0,
  "length": 28
}
```

**Response Format:**
```json
{
  "matches": [
    {
      "message": "Possible typo detected",
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
        "category": {
          "id": "GRAMMAR",
          "name": "Grammar"
        }
      }
    }
  ]
}
```

## API Endpoints

### POST /check

Main endpoint for grammar checking.

**Request:**
```http
POST /check HTTP/1.1
Content-Type: application/json

{
  "text": "Your text here",
  "language": "en-US"
}
```

**Parameters:**
- `text` (required): The text to check
- `language` (required): Language code (e.g., "en-US", "en-GB")
- `offset` (optional): Starting position in text
- `length` (optional): Length of text to check

**Response:**
```json
{
  "matches": [
    {
      "message": "Error description",
      "shortMessage": "Brief description",
      "replacements": [{"value": "suggestion"}],
      "offset": 0,
      "length": 5,
      "context": {...},
      "rule": {...}
    }
  ]
}
```

**Status Codes:**
- `200 OK`: Request processed successfully
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "triton_status": "connected",
  "models_loaded": 2
}
```

### GET /models

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "bert_error_detection",
      "version": "1",
      "status": "ready"
    },
    {
      "name": "llm_correction",
      "version": "1",
      "status": "ready"
    }
  ]
}
```

### POST /feedback (Future)

Endpoint for collecting user feedback on corrections.

## Request Flow

1. **Receive Request**: API receives text from LanguageTool
2. **Preprocessing**: Tokenize and prepare input
3. **Model Inference**: 
   - Send to BERT model for error detection
   - Send detected errors to LLM for corrections
4. **Postprocessing**: Format results as LanguageTool matches
5. **Return Response**: Send formatted response

## Error Handling

### Input Validation
- Text length limits
- Language support validation
- Offset/length boundary checks

### Error Responses
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": "Additional details"
}
```

### Error Codes
- `INVALID_INPUT`: Malformed request
- `UNSUPPORTED_LANGUAGE`: Language not supported
- `MODEL_ERROR`: Model inference failed
- `TIMEOUT`: Request timeout

## Authentication

> To be implemented

Options:
- API key in header: `X-API-Key: your-api-key`
- Bearer token: `Authorization: Bearer token`
- OAuth 2.0

## Rate Limiting

> To be implemented

- Per-API key limits
- Rate limit headers in response:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

## Model Selection

### Strategy 1: Sequential Pipeline
- First: BERT for error detection
- Then: LLM for correction suggestions
- Fastest for most use cases

### Strategy 2: Ensemble
- Multiple models run in parallel
- Results aggregated with confidence scoring
- Higher accuracy, more resource-intensive

### Strategy 3: Model Routing
- Simple errors → Fast BERT model
- Complex errors → LLM model
- Optimizes for performance

## Configuration

API service configuration (example):

```yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
triton:
  host: triton-server
  port: 8001
  protocol: grpc
  
models:
  error_detection:
    name: bert_error_detection
    version: 1
  correction:
    name: llm_correction
    version: 1
    
limits:
  max_text_length: 10000
  timeout_seconds: 30
```

## Testing

### Unit Tests
- Test request validation
- Test response formatting
- Mock Triton client

### Integration Tests
- Test with actual Triton server
- Test end-to-end flow
- Test error scenarios

### Load Tests
- Concurrent requests
- Large text inputs
- Sustained load testing

## Examples

### Python Client Example

```python
import requests

def check_grammar(text, language="en-US"):
    url = "http://localhost:8000/check"
    data = {
        "text": text,
        "language": language
    }
    response = requests.post(url, json=data)
    return response.json()

# Usage
result = check_grammar("The text too check.")
for match in result["matches"]:
    print(f"Error at position {match['offset']}: {match['message']}")
    print(f"Suggestions: {[r['value'] for r in match['replacements']]}")
```

### cURL Example

```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The text too check.",
    "language": "en-US"
  }'
```

## Performance Considerations

- **Async Processing**: Use async/await for non-blocking I/O
- **Connection Pooling**: Maintain persistent connections to Triton
- **Caching**: Cache model results for identical text
- **Batching**: Batch multiple requests when possible

## Monitoring

Metrics to track:
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate
- Model inference time
- Triton communication time
