# GECToR RoBERTa Model for Triton

This directory contains the GECToR (Grammatical Error Correction: Tag, Not Rewrite) model configuration for Triton Inference Server.

## Model Information

- **Model Name**: gotutiyan/gector-roberta-base-5k
- **Source**: HuggingFace Model Hub
- **Task**: Grammar Error Correction (Token Classification)
- **Architecture**: RoBERTa-based sequence tagger
- **Backend**: Python (HuggingFace Transformers)

## About GECToR

GECToR is a sequence tagging approach to grammar error correction that treats GEC as a token-level classification task. Instead of generating corrected text directly, it predicts edit operations (tags) for each token.

### Key Features

- **Efficient**: Faster than sequence-to-sequence approaches
- **Accurate**: Competitive performance with state-of-the-art models
- **Flexible**: Can handle various types of grammatical errors
- **Tag-based**: Uses a predefined set of edit operations

### How It Works

1. Input text is tokenized
2. Model predicts an edit tag for each token
3. Tags indicate operations like:
   - `KEEP`: No change needed
   - `DELETE`: Remove this token
   - `REPLACE_<word>`: Replace with specified word
   - `APPEND_<word>`: Add word after this token

## Model Structure

```
gector_roberta/
├── config.pbtxt          # Triton model configuration
├── 1/                    # Model version 1
│   └── model.py          # Python backend implementation
└── README.md             # This file
```

## Configuration

The model is configured with the following parameters:

- **Max Batch Size**: 8
- **Backend**: Python
- **Input**: Pre-tokenized input_ids and attention_mask (INT64 tensors)
- **Output**: Raw logits for classification
- **GPU Support**: Yes (configurable)
- **Dynamic Batching**: Enabled

## Input/Output Specification

### Input

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| input_ids | INT64 | [batch, seq_len] | Tokenized input sequence IDs |
| attention_mask | INT64 | [batch, seq_len] | Attention mask for the input |

### Output

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| logits_labels | FP32 | [batch, seq_len, num_labels] | Classification logits for each token |
| logits_d | FP32 | [batch, seq_len, num_labels] | Detection logits (same as logits_labels for GECToR) |

## Usage Example

### With Triton Client (Python)

```python
import tritonclient.http as httpclient
import numpy as np
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gector-roberta-base-5k")

# Create Triton client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input
text = "I has a grammar error in this sentence."
encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

# Convert to INT64 numpy arrays
input_ids_np = encoded['input_ids'].astype(np.int64)
attention_mask_np = encoded['attention_mask'].astype(np.int64)

# Create input tensors
input_ids_input = httpclient.InferInput("input_ids", input_ids_np.shape, "INT64")
input_ids_input.set_data_from_numpy(input_ids_np)

attention_mask_input = httpclient.InferInput("attention_mask", attention_mask_np.shape, "INT64")
attention_mask_input.set_data_from_numpy(attention_mask_np)

inputs = [input_ids_input, attention_mask_input]

# Create output request
outputs = [
    httpclient.InferRequestedOutput("logits_labels"),
    httpclient.InferRequestedOutput("logits_d")
]

# Send request
response = client.infer(
    model_name="gector_roberta",
    inputs=inputs,
    outputs=outputs
)

# Get results
logits_labels = response.as_numpy("logits_labels")
logits_d = response.as_numpy("logits_d")

# Get predictions
predictions = np.argmax(logits_labels, axis=-1)
print("Predictions shape:", predictions.shape)
print("Logits shape:", logits_labels.shape)
```

### With gRPC Client (Python)

```python
import tritonclient.grpc as grpcclient
import numpy as np
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gector-roberta-base-5k")

# Create Triton client
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Prepare input
text = "I has a grammar error in this sentence."
encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

# Convert to INT64 numpy arrays
input_ids_np = encoded['input_ids'].astype(np.int64)
attention_mask_np = encoded['attention_mask'].astype(np.int64)

# Create input tensors for Triton
input_ids_input = grpcclient.InferInput("input_ids", input_ids_np.shape, "INT64")
input_ids_input.set_data_from_numpy(input_ids_np)

attention_mask_input = grpcclient.InferInput("attention_mask", attention_mask_np.shape, "INT64")
attention_mask_input.set_data_from_numpy(attention_mask_np)

inputs = [input_ids_input, attention_mask_input]

# Define outputs we expect from Triton
outputs = [
    grpcclient.InferRequestedOutput("logits_labels"),
    grpcclient.InferRequestedOutput("logits_d")
]

# Call Triton server
response = client.infer(
    model_name="gector_roberta",
    inputs=inputs,
    outputs=outputs
)

# Get results
logits_labels = response.as_numpy("logits_labels")
logits_d = response.as_numpy("logits_d")
```

### With curl

Note: curl example is complex for INT64 tensor inputs. Use the Python client for easier integration.

```bash
# You need to base64-encode the binary INT64 data
# It's recommended to use the Python client library instead
```

## Deployment

### Option 1: Using Deployment Script

```bash
cd triton_server/scripts
./deploy_gector.sh
```

This script will:
1. Download the model from HuggingFace
2. Cache it locally
3. Prepare the model for Triton serving

### Option 2: Manual Deployment

The model files are already in place. When you start Triton, it will automatically download the HuggingFace model to cache on first load.

```bash
# Start Triton server
docker-compose up triton-server
```

The model will be automatically loaded from HuggingFace's model hub when Triton starts.

## Performance

### Expected Metrics (NVIDIA T4 GPU)

| Batch Size | Latency (p50) | Latency (p99) | Throughput |
|-----------|---------------|---------------|------------|
| 1 | ~80ms | ~120ms | ~12 req/s |
| 4 | ~150ms | ~200ms | ~25 req/s |
| 8 | ~280ms | ~350ms | ~28 req/s |

*Note: Actual performance may vary based on hardware and text length*

## Requirements

### Docker Image Dependencies

The Triton Docker image must include:
- Python 3.8+
- transformers
- torch
- sentencepiece
- protobuf

These are included in the custom Dockerfile at `docker/triton/Dockerfile`.

### Storage

- Model cache: ~500MB
- Runtime memory: ~2GB GPU memory

## Troubleshooting

### Model Fails to Load

**Error**: "Failed to load model"

**Solutions**:
1. Check internet connectivity (first load requires downloading from HuggingFace)
2. Verify HuggingFace credentials if model requires authentication
3. Check GPU availability and memory
4. Review Triton logs: `docker logs <triton-container>`

### Out of Memory

**Error**: "CUDA out of memory"

**Solutions**:
1. Reduce max_batch_size in config.pbtxt
2. Reduce instance count
3. Use CPU instead of GPU
4. Increase GPU memory allocation

### Slow Inference

**Solutions**:
1. Enable dynamic batching (already enabled)
2. Use GPU acceleration
3. Optimize batch sizes
4. Consider model quantization

## Model Customization

### Using a Different GECToR Model

To use a different GECToR model from HuggingFace:

1. Edit `1/model.py`
2. Change `model_name = "gotutiyan/gector-roberta-base-5k"` to your model
3. Restart Triton

### Adjusting Batch Size

Edit `config.pbtxt`:

```protobuf
max_batch_size: 16  # Change from 8 to 16
```

### Using CPU Instead of GPU

Edit `config.pbtxt`:

```protobuf
instance_group [
  {
    count: 1
    kind: KIND_CPU  # Changed from KIND_GPU
  }
]
```

## References

- [GECToR Paper](https://arxiv.org/abs/2005.12592)
- [HuggingFace Model](https://huggingface.co/gotutiyan/gector-roberta-base-5k)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Python Backend](https://github.com/triton-inference-server/python_backend)

## License

The model follows the license from the original HuggingFace repository.
