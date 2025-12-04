# Development Guide

## Overview

This guide helps developers set up their environment and contribute to the Grammared Language project.

## Prerequisites

### Required Software

- **Python 3.9+**: Main development language
- **Docker & Docker Compose**: For containerization
- **Git**: Version control

### Optional Tools

- **NVIDIA GPU + CUDA**: For GPU-accelerated development
- **Visual Studio Code**: Recommended IDE
- **Postman/Insomnia**: API testing
- **Jupyter Notebook**: Model experimentation

## Development Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/rayliuca/grammared_language.git
cd grammared_language
```

### 2. Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# API service dependencies
pip install fastapi uvicorn[standard] tritonclient[all] pydantic

# Development dependencies
pip install pytest pytest-cov black flake8 mypy

# Model dependencies (for model development)
pip install torch transformers onnx onnxruntime
```

### 4. IDE Setup

#### Visual Studio Code

Recommended extensions:
- Python (Microsoft)
- Docker (Microsoft)
- GitLens
- Pylance
- YAML

`.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true
}
```

## Project Structure for Development

```
grammared_language/
├── api/                    # API service
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI application
│   │   ├── models.py      # Pydantic models
│   │   ├── triton_client.py
│   │   └── utils.py
│   └── tests/
│       ├── test_api.py
│       └── test_triton_client.py
├── triton_server/
│   ├── model_repository/
│   └── scripts/
├── models/
├── docker/
├── config/
└── docs/
```

## Coding Standards

### Python Style Guide

Follow PEP 8 with some modifications:

- **Line length**: 100 characters (instead of 79)
- **Imports**: Use absolute imports
- **Docstrings**: Google style
- **Type hints**: Required for all functions

### Code Formatting

```bash
# Format code with black
black api/src/

# Check code with flake8
flake8 api/src/ --max-line-length=100

# Type checking with mypy
mypy api/src/
```

### Example Code Style

```python
from typing import List, Optional
from pydantic import BaseModel


class GrammarRequest(BaseModel):
    """Request model for grammar checking.
    
    Attributes:
        text: The text to check for grammar errors.
        language: Language code (e.g., 'en-US').
        offset: Optional starting position in text.
    """
    
    text: str
    language: str
    offset: Optional[int] = None


def check_grammar(request: GrammarRequest) -> List[dict]:
    """Check text for grammar errors.
    
    Args:
        request: Grammar checking request.
        
    Returns:
        List of grammar error matches.
        
    Raises:
        ValueError: If text is empty or invalid.
    """
    if not request.text:
        raise ValueError("Text cannot be empty")
    
    # Implementation here
    return []
```

## Testing

### Unit Testing

Location: `api/tests/`

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=api/src --cov-report=html

# Run specific test file
pytest api/tests/test_api.py

# Run specific test
pytest api/tests/test_api.py::test_check_endpoint
```

### Test Structure

```python
import pytest
from fastapi.testclient import TestClient
from api.src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_check_endpoint(client):
    """Test grammar check endpoint."""
    data = {
        "text": "The text too check.",
        "language": "en-US"
    }
    response = client.post("/check", json=data)
    assert response.status_code == 200
    assert "matches" in response.json()
```

### Integration Testing

Test with actual Triton server:

```python
import tritonclient.grpc as grpcclient


def test_triton_connection():
    """Test connection to Triton server."""
    triton_client = grpcclient.InferenceServerClient(
        url="localhost:8001"
    )
    assert triton_client.is_server_live()
```

## Local Development Workflow

### 1. Start Triton Server

```bash
# Using Docker
docker run -d \
  --name triton-dev \
  --gpus all \
  -p 8001:8001 \
  -v $(pwd)/triton_server/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### 2. Run API Service Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TRITON_HOST=localhost
export TRITON_PORT=8001

# Run with uvicorn (auto-reload)
cd api
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test grammar check
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "The text too check.", "language": "en-US"}'
```

## Model Development

### Setting Up Model Development Environment

```bash
# Install model development dependencies
pip install torch transformers datasets accelerate

# Install Jupyter for experimentation
pip install jupyter notebook
```

### Model Training Template

```python
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)


def train_grammar_model():
    """Train BERT model for grammar error detection."""
    
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=2  # Error or no error
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/bert/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        logging_dir="./logs",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained("./models/bert/trained_model")
    tokenizer.save_pretrained("./models/bert/trained_model")


if __name__ == "__main__":
    train_grammar_model()
```

### Converting Models for Triton

```python
import torch
import torch.onnx


def convert_to_onnx(model_path: str, output_path: str):
    """Convert PyTorch model to ONNX format."""
    
    # Load model
    model = torch.load(model_path)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_length = 512
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length))
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    print(f"Model converted to ONNX: {output_path}")
```

## Debugging

### Debugging API Service

Use Python debugger or IDE debugging:

```python
import pdb

def some_function():
    # Set breakpoint
    pdb.set_trace()
    # Code here
```

### Debugging Triton Issues

```bash
# Check Triton logs
docker logs triton-dev

# Verbose logging
docker run ... tritonserver --log-verbose=1 ...

# Check model status
curl http://localhost:8002/v2/models/<model_name>
```

## Git Workflow

### Branch Naming

- `feature/feature-name`: New features
- `bugfix/bug-description`: Bug fixes
- `docs/doc-update`: Documentation updates
- `refactor/refactor-description`: Code refactoring

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]
[optional footer]
```

Examples:
- `feat(api): add grammar check endpoint`
- `fix(triton): resolve connection timeout issue`
- `docs(readme): update installation instructions`
- `test(api): add unit tests for check endpoint`

### Pull Request Process

1. Create feature branch
2. Make changes
3. Write/update tests
4. Run tests and linters
5. Commit changes
6. Push to remote
7. Create pull request
8. Address review comments
9. Merge after approval

## Performance Profiling

### API Profiling

```python
import cProfile
import pstats

def profile_api():
    """Profile API performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run API code
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

Example load test:
```python
from locust import HttpUser, task, between


class GrammarCheckUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def check_grammar(self):
        self.client.post("/check", json={
            "text": "The text too check.",
            "language": "en-US"
        })
```

## Continuous Integration

### GitHub Actions (Example)

`.github/workflows/test.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov
    
    - name: Run linters
      run: |
        black --check api/src/
        flake8 api/src/
```

## Documentation

### Adding Documentation

- Update relevant markdown files in `docs/`
- Use clear, concise language
- Include code examples
- Add diagrams where helpful

### Generating API Documentation

```bash
# Install pdoc
pip install pdoc

# Generate documentation
pdoc --html --output-dir docs/api api/src/
```

## Tips and Best Practices

1. **Write tests first**: TDD approach
2. **Keep PRs small**: Easier to review
3. **Document as you code**: Don't leave it for later
4. **Use type hints**: Helps catch errors early
5. **Profile before optimizing**: Don't guess
6. **Review your own code**: Before requesting review
7. **Keep dependencies updated**: Security and features
8. **Monitor resource usage**: Memory, CPU, GPU

## Getting Help

- Check existing documentation
- Search closed issues
- Ask in discussions
- Open an issue with details
