# Triton Inference Server

This directory contains the configuration and setup for NVIDIA Triton Inference Server to serve ML models.

## Structure

```
triton_server/
├── model_repository/    # Triton model repository containing model versions
├── scripts/            # Utility scripts for model deployment and management
└── README.md           # This file
```

## Purpose

Triton Inference Server provides:
- High-performance inference for multiple model frameworks
- Model versioning and A/B testing
- Dynamic model loading/unloading
- GPU acceleration support
- Multiple concurrent model serving

## Supported Models

### HuggingFace Models

#### GECToR (Grammar Error Correction: Tag, Not Rewrite)
- **Model**: gotutiyan/gector-roberta-base-5k
- **Task**: Grammar error correction via sequence tagging
- **Backend**: Python (HuggingFace Transformers)
- **Location**: `model_repository/gector_roberta/`
- **Status**: ✅ Ready for deployment

See [gector_roberta/README.md](model_repository/gector_roberta/README.md) for detailed documentation.

### BERT-based Models
- Grammar error detection models
- Contextual embeddings for language understanding
- Fine-tuned models for specific error types

### Large Language Models (LLMs)
- Grammar correction models
- Text generation for suggestions
- Context-aware error correction

## Model Repository Structure

Each model in the `model_repository` should follow Triton's standard format:

```
model_repository/
└── <model_name>/
    ├── config.pbtxt           # Model configuration
    └── <version>/
        └── model.<ext>        # Model files
```

## Configuration

- **Backend Support**: PyTorch, TensorFlow, ONNX Runtime
- **Batching**: Dynamic batching for improved throughput
- **Instance Groups**: Multiple instances for parallel processing

## Future Development

- Model deployment scripts
- Performance optimization configurations
- Model ensemble definitions
- Preprocessing/postprocessing pipelines
