# System Architecture

## Overview

The Grammared Language system is designed as a microservices architecture with two main components:
1. API Service - LanguageTool interface
2. Triton Inference Server - ML model serving

## Architecture Diagram

```
┌─────────────────────┐
│  LanguageTool       │
│  Client/Plugin      │
└──────────┬──────────┘
           │ HTTP/REST
           ▼
┌─────────────────────────────────────────────┐
│         API Service Container                │
│  ┌─────────────────────────────────────┐   │
│  │  FastAPI/Flask Application          │   │
│  │  - Route Handling                   │   │
│  │  - Request Validation               │   │
│  │  - Response Formatting              │   │
│  └─────────────┬───────────────────────┘   │
│                │ gRPC/HTTP                  │
└────────────────┼────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    Triton Inference Server Container        │
│  ┌──────────────────────────────────────┐  │
│  │  Model Repository                    │  │
│  │  ├── bert_error_detection/          │  │
│  │  │   └── 1/model.pt                 │  │
│  │  ├── llm_correction/                │  │
│  │  │   └── 1/model.onnx              │  │
│  │  └── ensemble_model/                │  │
│  │      └── 1/config.pbtxt             │  │
│  └──────────────────────────────────────┘  │
│                                              │
│  ┌──────────────────────────────────────┐  │
│  │  Inference Backends                  │  │
│  │  - PyTorch Backend                   │  │
│  │  - ONNX Runtime                      │  │
│  │  - TensorFlow Backend                │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Component Details

### API Service

**Responsibilities:**
- Accept HTTP requests from LanguageTool
- Validate input text and parameters
- Communicate with Triton server for model inference
- Aggregate and format results
- Return LanguageTool-compatible responses

**Key Endpoints:**
- `POST /check` - Main grammar checking endpoint
- `GET /health` - Health check endpoint
- `GET /models` - List available models

**Technologies:**
- Python 3.9+
- FastAPI (async support) or Flask
- Triton Client Library (tritonclient)
- Pydantic (data validation)

### Triton Inference Server

**Responsibilities:**
- Load and manage ML models
- Perform model inference
- Handle batching and optimization
- Provide metrics and monitoring

**Features:**
- Multiple model support
- Dynamic batching
- Model versioning
- GPU acceleration
- Concurrent model execution

**Supported Backends:**
- PyTorch
- TensorFlow
- ONNX Runtime
- Python backend (for custom logic)

## Data Flow

### Request Processing

1. **LanguageTool Request**
   - Client sends text with context
   - Includes language, region settings
   - May include previous error ranges

2. **API Service Processing**
   - Parse and validate request
   - Extract text and metadata
   - Prepare input for model(s)

3. **Model Inference**
   - API sends request to Triton
   - Triton routes to appropriate model(s)
   - Models process in parallel if possible
   - Results returned to API

4. **Response Generation**
   - API aggregates model outputs
   - Formats as LanguageTool matches
   - Includes suggestions and replacements
   - Returns JSON response

### Model Pipeline

```
Input Text
    ↓
Tokenization
    ↓
BERT Model (Error Detection)
    ↓
Error Regions Identified
    ↓
LLM Model (Correction Suggestions)
    ↓
Ranked Suggestions
    ↓
Response Formatting
```

## Scaling Considerations

### Horizontal Scaling
- Multiple API service instances behind load balancer
- Shared Triton server or dedicated per-API instance

### Vertical Scaling
- GPU allocation for Triton server
- Model instance groups for parallelism
- Dynamic batching for throughput

### Performance Optimization
- Model quantization (INT8, FP16)
- Batch processing
- Caching frequently checked text
- Async request handling

## Security

### API Security
- API key authentication
- Rate limiting per client
- Input validation and sanitization
- HTTPS/TLS encryption

### Model Security
- Model integrity verification
- Access control to model repository
- Audit logging

## Monitoring and Observability

### Metrics
- Request latency
- Model inference time
- Error rates
- Throughput (requests/second)

### Logging
- Request/response logging
- Model prediction logging
- Error and exception tracking

### Health Checks
- API service health endpoint
- Triton server readiness/liveness
- Model availability checks

## Deployment Strategies

### Development
- Docker Compose for local development
- Hot-reload for API changes
- Model mounting via volumes

### Production
- Kubernetes for orchestration
- Horizontal pod autoscaling
- Persistent volumes for models
- Load balancing
- Rolling updates

## Future Enhancements

1. **Model Ensemble**: Combine multiple models for better accuracy
2. **A/B Testing**: Compare different model versions
3. **Feedback Loop**: Collect corrections for model improvement
4. **Multi-language Support**: Extend beyond English
5. **Custom Rules**: User-defined grammar rules
6. **Analytics Dashboard**: Usage statistics and insights
