# Models

This directory stores ML model files and related resources.

## Structure

```
models/
├── bert/          # BERT and transformer-based models
├── llm/           # Large Language Models
└── README.md      # This file
```

## Purpose

This directory is for:
- Storing trained model weights
- Model checkpoints
- Tokenizer files and vocabularies
- Model metadata and configuration

## Model Types

### BERT Models (`bert/`)
- Pre-trained BERT models fine-tuned for grammar error detection
- Smaller, faster models suitable for real-time processing
- Specific error type classifiers

### Large Language Models (`llm/`)
- Larger models for complex grammar correction
- Context-aware suggestion generation
- Multi-task models handling various error types

## Model Storage Guidelines

1. **Version Control**: Use Git LFS for large model files or store separately
2. **Naming Convention**: `<model-type>_<task>_<version>` (e.g., `bert_grammar_v1.0`)
3. **Documentation**: Each model should include a README with:
   - Model architecture details
   - Training data information
   - Performance metrics
   - Usage examples

## Integration with Triton

Models stored here should be converted to Triton-compatible formats and deployed to the `triton_server/model_repository/` directory.

## Future Development

- Model conversion scripts
- Model evaluation tools
- Benchmark results
- Model registry/catalog
