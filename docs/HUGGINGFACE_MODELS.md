# Using HuggingFace Models with Triton

This guide explains how to serve HuggingFace transformer models using Triton Inference Server in the Grammared Language project.

## Overview

The Grammared Language project now supports serving HuggingFace models directly through Triton's Python backend. This allows us to leverage the vast ecosystem of pre-trained models available on the HuggingFace Hub.

## Currently Deployed Models

### GECToR (gotutiyan/gector-roberta-base-5k)

The first HuggingFace model integrated is GECToR, a state-of-the-art grammar error correction model that uses sequence tagging.

- **Model**: gotutiyan/gector-roberta-base-5k
- **Task**: Grammar Error Correction
- **Approach**: Token classification (sequence tagging)
- **Location**: `triton_server/model_repository/gector_roberta/`

See the [GECToR model documentation](../triton_server/model_repository/gector_roberta/README.md) for detailed information.

## Architecture

### HuggingFace Model Serving Stack

```
┌─────────────────────────────────────────┐
│     Client Application                  │
└────────────────┬────────────────────────┘
                 │
                 │ HTTP/gRPC
                 ▼
┌─────────────────────────────────────────┐
│   Triton Inference Server               │
│   ┌─────────────────────────────────┐   │
│   │  Python Backend                 │   │
│   │  ┌───────────────────────────┐  │   │
│   │  │ HuggingFace Transformers  │  │   │
│   │  │ - AutoTokenizer           │  │   │
│   │  │ - AutoModel               │  │   │
│   │  │ - PyTorch/TensorFlow      │  │   │
│   │  └───────────────────────────┘  │   │
│   └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                 │
                 │ Model Files
                 ▼
┌─────────────────────────────────────────┐
│   HuggingFace Model Hub                 │
│   - Model weights                       │
│   - Tokenizer                           │
│   - Configuration                       │
└─────────────────────────────────────────┘
```

## How It Works

### Model Loading

1. **On First Start**: When Triton loads the model for the first time, the Python backend downloads the model from HuggingFace Hub
2. **Caching**: The model is cached locally in `/models/.cache`
3. **Subsequent Starts**: The cached model is loaded directly, no download needed

### Inference Flow

1. **Input**: Client sends text to Triton via HTTP/gRPC
2. **Tokenization**: Python backend tokenizes the text using HuggingFace tokenizer
3. **Inference**: Model processes the tokenized input
4. **Postprocessing**: Results are formatted and returned to client
5. **Output**: Client receives corrections, labels, and confidence scores

## Benefits

### Why Use HuggingFace Models?

1. **Pre-trained Models**: Access to thousands of pre-trained models
2. **Easy Updates**: Simple to swap or update models
3. **Standardized Interface**: Consistent API across different model types
4. **Community Support**: Active community and regular updates
5. **Fine-tuning**: Easy to fine-tune models on custom data

### Why Use Triton?

1. **Performance**: GPU acceleration and optimized inference
2. **Scalability**: Dynamic batching and multi-instance serving
3. **Production-Ready**: Built for high-throughput production environments
4. **Monitoring**: Built-in metrics and health checks
5. **Multi-Framework**: Support for PyTorch, TensorFlow, ONNX, etc.

## Adding New HuggingFace Models

### Step 1: Create Model Directory

```bash
cd triton_server/model_repository
mkdir -p your_model_name/1
```

### Step 2: Create Python Backend (`model.py`)

Create `your_model_name/1/model.py`:

```python
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModel
import torch

class TritonPythonModel:
    def initialize(self, args):
        # Load your HuggingFace model
        model_name = "your-username/your-model-name"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def execute(self, requests):
        responses = []
        for request in requests:
            # Process request
            input_text = pb_utils.get_input_tensor_by_name(request, "INPUT")
            # ... your inference code ...
            responses.append(response)
        return responses
        
    def finalize(self):
        # Cleanup
        del self.model
        del self.tokenizer
```

### Step 3: Create Configuration (`config.pbtxt`)

Create `your_model_name/config.pbtxt`:

```protobuf
name: "your_model_name"
backend: "python"
max_batch_size: 8

input [
  {
    name: "INPUT"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "OUTPUT"
    data_type: TYPE_STRING
    dims: [1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100000
}
```

### Step 4: Test the Model

```bash
# Start Triton
docker-compose up triton-server

# Test with Python client
python triton_server/scripts/test_your_model.py
```

## Model Types Supported

### Text Classification

Models that classify text into categories (sentiment, topic, etc.):

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Token Classification

Models that classify each token (NER, POS tagging, GEC):

```python
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(model_name)
```

### Text Generation

Models that generate text (summarization, translation):

```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### Question Answering

Models that answer questions based on context:

```python
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
```

## Configuration Options

### GPU vs CPU

**GPU (Recommended for Production)**:
```protobuf
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
```

**CPU (Development/Testing)**:
```protobuf
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

### Dynamic Batching

Enable for better throughput:

```protobuf
dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100000
}
```

### Multiple Versions

Support A/B testing:

```protobuf
version_policy: {
  all { }  # Load all versions
}
```

Or specific versions:

```protobuf
version_policy: {
  specific {
    versions: [1, 2]
  }
}
```

## Best Practices

### 1. Model Optimization

- **Quantization**: Use INT8 or FP16 for faster inference
- **ONNX**: Convert to ONNX for better performance
- **Pruning**: Remove unnecessary weights

### 2. Resource Management

- **Batch Size**: Tune based on GPU memory
- **Instance Count**: Balance between latency and throughput
- **Timeout**: Set appropriate timeouts for long-running models

### 3. Caching

- **Model Cache**: Use persistent volumes for model cache
- **Result Cache**: Cache common queries if applicable

### 4. Monitoring

- **Metrics**: Monitor latency, throughput, error rates
- **Logs**: Enable appropriate logging levels
- **Health Checks**: Regular health checks for model availability

## Troubleshooting

### Model Won't Load

**Issue**: Model fails to load on Triton startup

**Solutions**:
1. Check internet connectivity (for first download)
2. Verify model name is correct on HuggingFace
3. Check GPU memory availability
4. Review Triton logs: `docker logs triton-server`

### Out of Memory

**Issue**: CUDA out of memory errors

**Solutions**:
1. Reduce `max_batch_size`
2. Reduce instance `count`
3. Use smaller model variant
4. Enable model quantization

### Slow Inference

**Issue**: High latency

**Solutions**:
1. Enable dynamic batching
2. Increase GPU instances
3. Use FP16 precision
4. Consider ONNX conversion

### Model Not Found

**Issue**: HuggingFace model not found

**Solutions**:
1. Verify model exists on HuggingFace Hub
2. Check if model requires authentication
3. Set HuggingFace token if needed:
   ```bash
   export HF_TOKEN=your_token_here
   ```

## Examples

### Example 1: Sentiment Analysis

```python
# model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TritonPythonModel:
    def initialize(self, args):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def execute(self, requests):
        # Implementation
        pass
```

### Example 2: Named Entity Recognition

```python
# model.py
from transformers import AutoModelForTokenClassification, AutoTokenizer

class TritonPythonModel:
    def initialize(self, args):
        model_name = "dslim/bert-base-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
    def execute(self, requests):
        # Implementation
        pass
```

## Resources

### Documentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [Triton Server Documentation](https://github.com/triton-inference-server/server)

### Model Hubs
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Grammar Error Correction Models](https://huggingface.co/models?pipeline_tag=text2text-generation&other=grammar-error-correction)

### Papers
- [GECToR: Grammatical Error Correction - Tag, Not Rewrite](https://arxiv.org/abs/2005.12592)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

## Future Enhancements

- [ ] Support for multi-GPU inference
- [ ] Model ensemble pipelines
- [ ] Automatic model optimization
- [ ] Custom preprocessing pipelines
- [ ] Model versioning and rollback
- [ ] A/B testing framework
