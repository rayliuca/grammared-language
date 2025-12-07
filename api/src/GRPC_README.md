# gRPC Grammar Check Server - LanguageTool Compatible

This directory contains a gRPC server implementation compatible with LanguageTool's ML Server protocol (ml_server.proto).

## Files

- **ml_server.proto** - Protocol Buffer definition (LanguageTool standard)
- **grpc_gen/** - Auto-generated Protocol Buffer and gRPC code
- **grpc_server.py** - gRPC server implementation
- **grpc_client.py** - gRPC client for testing
- **requirements_grpc.txt** - gRPC-specific dependencies

## Services Implemented

### ProcessingServer
- **Analyze** - Tokenize and analyze text, return AnalyzedSentence objects
- **Process** - Process analyzed sentences and return grammar matches

### MLServer
- **Match** - Match grammar errors in raw sentences
- **MatchAnalyzed** - Match grammar errors in pre-analyzed sentences

### PostProcessingServer
- **Process** - Post-process and resort suggestions (placeholder)

## Setup

### 1. Install gRPC Dependencies

```bash
pip install -r requirements_grpc.txt
```

Or install individually:

```bash
pip install grpcio grpcio-tools protobuf
```

### 2. Regenerate Proto Files (if modified)

```bash
python generate_grpc.py
```

## Running the gRPC Server

### Basic Usage

```bash
python -m api.src.grpc_server
```

The server will start on `0.0.0.0:50051` by default.

### With Custom Host/Port

```python
from api.src.grpc_server import serve

serve(host="127.0.0.1", port=9000)
```

## Using the gRPC Client

### Testing with Python Client

```python
from api.src.grpc_client import test_match, test_analyze

# Test MLServer.Match
sentences = ["He go to the store.", "She are smart."]
response = test_match(sentences)

# Test ProcessingServer.Analyze
text = "He go to the store. She are smart."
response = test_analyze(text)
```

Or run the client directly:

```bash
python -m api.src.grpc_client
```

### Using grpcurl

```bash
# Install grpcurl if not already installed
# https://github.com/fullstorydev/grpcurl

# Test MLServer.Match
grpcurl -plaintext \
  -d '{"sentences":["He go to store."]}' \
  localhost:50051 \
  lt_ml_server.MLServer/Match

# Test ProcessingServer.Analyze
grpcurl -plaintext \
  -d '{"text":"He go to store.","options":{"language":"en"}}' \
  localhost:50051 \
  lt_ml_server.ProcessingServer/Analyze
```

## Proto Message Definitions

### MatchRequest
- `sentences` (repeated string): Sentences to check
- `inputLogging` (bool): Enable input logging for errors

### MatchResponse
- `sentenceMatches` (repeated MatchList): Matches for each sentence

### Match
- `offset` (uint32): Position in sentence
- `length` (uint32): Length of error
- `id` (string): Rule ID
- `suggestions` (repeated string): Legacy suggestions
- `ruleDescription` (string): Description
- `matchDescription` (string): Match details
- `suggestedReplacements` (repeated SuggestedReplacement): New suggestions
- `autoCorrect` (bool): Auto-correction eligible
- `type` (MatchType): Error type (UnknownWord, Hint, Other)
- `rule` (Rule): Rule information

### AnalyzeRequest
- `text` (string): Text to analyze
- `options` (ProcessingOptions): Processing configuration

### ProcessingOptions
- `language` (string): Language code
- `level` (Level): Check level (picky, academic, clarity, etc.)
- `premium` (bool): Premium features
- `enabledRules` (repeated string): Rules to enable
- `disabledRules` (repeated string): Rules to disable

## Conversion Strategy

The server converts Pydantic models (from output_models.py) to ml_server Protocol Buffer format:

```
Pydantic Match 
    ↓
pydantic_match_to_ml_match() conversion function
    ↓
ml_server.Match (protobuf)
```

This allows the existing GECToR-based grammar checking to work with LanguageTool's standard protocol.

## Performance Characteristics

### gRPC Benefits
- **Binary Protocol**: More efficient than JSON
- **Multiplexing**: Multiple requests over single connection
- **Type Safety**: Compile-time checking with protobuf
- **LanguageTool Compatible**: Works with LanguageTool ecosystem

## Docker Support

Add to docker-compose.yml:

```yaml
grpc-server:
  build:
    context: .
    dockerfile: docker/api/Dockerfile
  ports:
    - "50051:50051"
  command: python -m api.src.grpc_server
```

## Development

### Adding New Features

1. Modify `ml_server.proto`
2. Regenerate: `python generate_grpc.py`
3. Implement in `grpc_server.py` servicer classes
4. Add test cases in `grpc_client.py`

### Error Handling

Comprehensive error handling with proper gRPC status codes:

```python
context.set_code(grpc.StatusCode.INTERNAL)
context.set_details(f"Error message")
```

## References

- [LanguageTool ML Server](https://github.com/languagetool-org/languagetool)
- [gRPC Documentation](https://grpc.io/docs/)
- [Protocol Buffers Guide](https://developers.google.com/protocol-buffers)
- [GECToR Documentation](https://github.com/grammarly/gector)
