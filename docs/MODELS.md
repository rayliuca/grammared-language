# Model Documentation

## Overview

This document describes the ML models used in the Grammared Language system for grammar error detection and correction.

## Model Types

### 1. BERT-based Models

BERT (Bidirectional Encoder Representations from Transformers) models are used for fast error detection.

#### Use Cases
- Grammatical error detection
- Error type classification
- Context understanding
- Quick preliminary checks

#### Advantages
- Fast inference time (< 100ms)
- Good accuracy for common errors
- Smaller model size (~110M-340M parameters)
- Works well with limited context

#### Model Variants
- **bert-base**: 12 layers, 110M parameters
- **bert-large**: 24 layers, 340M parameters
- **distilbert**: 6 layers, 66M parameters (faster)

#### Fine-tuning
Models should be fine-tuned on grammar error correction datasets:
- CoNLL-2014 Shared Task
- FCE (First Certificate in English)
- NUCLE (NUS Corpus of Learner English)
- C4-200M corpus

### 2. Large Language Models (LLMs)

LLMs provide sophisticated correction suggestions and context-aware improvements.

#### Use Cases
- Complex grammar corrections
- Style improvements
- Context-aware suggestions
- Explanation generation

#### Advantages
- Higher accuracy for complex errors
- Better context understanding
- Multi-task capabilities
- Natural correction suggestions

#### Model Options
- **GPT-2/GPT-3**: General language models
- **T5**: Text-to-text transformer
- **BART**: Denoising autoencoder
- **Fine-tuned variants**: Grammar-specific models

#### Considerations
- Larger model size (350M-7B+ parameters)
- Slower inference (100ms-1s+)
- Higher GPU memory requirements
- May need quantization for deployment

## Model Repository Structure

Models in Triton must follow this structure:

```
model_repository/
├── bert_error_detection/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
├── llm_correction/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.onnx
│       └── tokenizer.json
└── ensemble_grammar/
    └── config.pbtxt
```

### Config File Example (config.pbtxt)

#### BERT Model Configuration

```protobuf
name: "bert_error_detection"
platform: "pytorch_libtorch"
max_batch_size: 32
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1, 512]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [-1, 512]
  }
]
output [
  {
    name: "error_logits"
    data_type: TYPE_FP32
    dims: [-1, 512, 2]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100000
}
```

#### LLM Model Configuration

```protobuf
name: "llm_correction"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1, 256]
  }
]
output [
  {
    name: "generated_ids"
    data_type: TYPE_INT64
    dims: [-1, 256]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

## Model Pipeline

### Sequential Pipeline

```
Input Text
    ↓
Tokenization (Preprocessing)
    ↓
BERT Error Detection
    ↓
Error Spans Identified
    ↓
LLM Correction (for each span)
    ↓
Correction Suggestions
    ↓
Ranking and Filtering
    ↓
Final Output
```

### Ensemble Model

Triton supports ensemble models that chain multiple models:

```protobuf
name: "ensemble_grammar"
platform: "ensemble"
input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }
]
output [
  {
    name: "corrections"
    data_type: TYPE_STRING
    dims: [-1]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "bert_error_detection"
      model_version: 1
      input_map {
        key: "text"
        value: "text"
      }
      output_map {
        key: "error_spans"
        value: "error_spans"
      }
    },
    {
      model_name: "llm_correction"
      model_version: 1
      input_map {
        key: "error_spans"
        value: "error_spans"
      }
      output_map {
        key: "corrections"
        value: "corrections"
      }
    }
  ]
}
```

## Model Optimization

### Quantization

Reduce model size and increase inference speed:

**INT8 Quantization:**
- 4x smaller model size
- 2-4x faster inference
- Minimal accuracy loss (~1%)

**FP16 Quantization:**
- 2x smaller model size
- 1.5-2x faster inference
- Negligible accuracy loss

### Model Conversion

Convert models to optimized formats:

**PyTorch → ONNX:**
```python
import torch
import torch.onnx

model = load_model()
dummy_input = torch.randn(1, 512)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 1: 'sequence'}
    }
)
```

**TensorFlow → ONNX:**
```python
import tf2onnx

spec = (tf.TensorSpec((None, 512), tf.int64, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path=output_path
)
```

## Performance Benchmarks

### Expected Performance Targets

| Model | Batch Size | Latency (p50) | Latency (p99) | Throughput |
|-------|-----------|---------------|---------------|------------|
| BERT-base | 1 | 50ms | 80ms | 20 req/s |
| BERT-base | 8 | 80ms | 120ms | 100 req/s |
| LLM-350M | 1 | 200ms | 300ms | 5 req/s |
| LLM-350M | 4 | 400ms | 600ms | 10 req/s |

*Benchmarks on NVIDIA T4 GPU*

## Model Evaluation

### Metrics

**Precision**: Correct detections / Total detections
**Recall**: Correct detections / Total errors
**F1 Score**: Harmonic mean of precision and recall

### Evaluation Datasets

- CoNLL-2014 Shared Task test set
- JFLEG (JHU FLuency-Extended GUG) corpus
- BEA-2019 Shared Task

### Example Evaluation

```python
from sklearn.metrics import precision_recall_fscore_support

# Ground truth and predictions
y_true = [...]  # True error labels
y_pred = [...]  # Model predictions

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

## Model Versioning

### Version Strategy

- **Version 1**: Initial model
- **Version 2**: Improved accuracy
- **Version 3**: Faster inference

### A/B Testing

Triton supports multiple versions simultaneously:

```bash
model_repository/
└── bert_error_detection/
    ├── config.pbtxt
    ├── 1/          # Stable version
    │   └── model.pt
    └── 2/          # Experimental version
        └── model.pt
```

## Training Recommendations

### Data Requirements

- Minimum: 10K annotated examples
- Recommended: 100K+ examples
- Diverse error types and contexts

### Training Pipeline

1. **Data Collection**: Gather training data
2. **Preprocessing**: Tokenization, cleaning
3. **Fine-tuning**: Train on task-specific data
4. **Evaluation**: Test on held-out set
5. **Optimization**: Quantize and convert
6. **Deployment**: Deploy to Triton

### Hyperparameters

**BERT Fine-tuning:**
- Learning rate: 2e-5 to 5e-5
- Batch size: 16-32
- Epochs: 3-5
- Warmup steps: 500-1000

**LLM Fine-tuning:**
- Learning rate: 1e-5 to 3e-5
- Batch size: 4-16
- Epochs: 1-3
- Gradient accumulation: 2-4 steps

## Error Types Supported

1. **Spelling Errors**: Typos, misspellings
2. **Grammar Errors**: Subject-verb agreement, tense
3. **Punctuation**: Missing or incorrect punctuation
4. **Word Choice**: Incorrect word usage
5. **Style**: Formal vs. informal, clarity
6. **Consistency**: Number agreement, capitalization

## Future Improvements

- [ ] Multi-language support
- [ ] Domain-specific models (academic, business)
- [ ] Real-time model updates
- [ ] Active learning pipeline
- [ ] Explainability features
- [ ] Confidence scores for suggestions
