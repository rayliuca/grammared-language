# Tests

This directory contains all tests for the grammared_language project, organized using pytest.

## Directory Structure

```
tests/
├── conftest.py                    # Shared pytest fixtures and configuration
├── unit/                          # Unit tests (no external dependencies)
│   └── test_grammar_correction_extractor.py
├── integration/                   # Integration tests (require services)
│   └── test_grpc_server.py       # Tests for gRPC server endpoints
└── functional/                    # Functional/E2E tests
    ├── test_triton_gector.py     # GECToR model on Triton
    └── test_triton_classifier.py # Classifier model on Triton
```

## Running Tests

### Install Test Dependencies

```bash
# Install test dependencies
pip install -e ".[test]"

# Or install all dev dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Tests by Category

```bash
# Run only unit tests (no external services needed)
pytest tests/unit/

# Run only integration tests (requires gRPC/API servers)
pytest tests/integration/

# Run only functional tests (requires Triton server)
pytest tests/functional/

# Run tests with specific markers
pytest -m unit
pytest -m integration
pytest -m functional
pytest -m "not slow"  # Skip slow tests
```

### Run Specific Test Files

```bash
# Run specific test file
pytest tests/unit/test_grammar_correction_extractor.py

# Run specific test class
pytest tests/unit/test_grammar_correction_extractor.py::TestGrammarCorrectionExtractor

# Run specific test function
pytest tests/unit/test_grammar_correction_extractor.py::TestGrammarCorrectionExtractor::test_simple_replacement
```

### Verbose Output

```bash
# Show detailed output
pytest -v

# Show even more detail (including print statements)
pytest -vv -s

# Show local variables on failure
pytest -l
```

### Coverage Reports

```bash
# Install coverage support
pip install pytest-cov

# Run with coverage
pytest --cov=grammared_language --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html
```

### Parallel Execution

```bash
# Install parallel execution support
pip install pytest-xdist

# Run tests in parallel (faster)
pytest -n auto
```

## Test Categories

### Unit Tests
- **Location**: `tests/unit/`
- **Requirements**: No external services
- **Marker**: `@pytest.mark.unit`
- **Description**: Test individual components in isolation

### Integration Tests
- **Location**: `tests/integration/`
- **Requirements**: gRPC server, API server running
- **Marker**: `@pytest.mark.integration`
- **Description**: Test interaction between components

**Before running integration tests:**
```bash
# Start gRPC server
python -m api.src.grpc_server

# Or start via docker-compose
docker-compose up grpc-api
```

### Functional Tests
- **Location**: `tests/functional/`
- **Requirements**: Triton Inference Server running with models
- **Marker**: `@pytest.mark.functional`
- **Description**: End-to-end tests with real models

**Before running functional tests:**
```bash
# Start Triton server
docker-compose up triton-server

# Or manually
tritonserver --model-repository=/path/to/model_repository
```

## Writing Tests

### Test File Naming
- Unit tests: `test_<module_name>.py`
- Integration tests: `test_<service>_<component>.py`
- Functional tests: `test_<system>_<feature>.py`

### Test Function Naming
```python
def test_<what_is_being_tested>():
    """Clear description of what this test validates."""
    pass
```

### Using Fixtures

Common fixtures are defined in [conftest.py](conftest.py):

```python
def test_something(triton_server_url, grpc_server_address):
    """Use shared fixtures from conftest.py"""
    client = create_client(triton_server_url)
    assert client.is_ready()
```

### Adding Custom Markers

```python
@pytest.mark.slow
def test_long_running_operation():
    """This test takes a long time."""
    pass

@pytest.mark.requires_triton
def test_triton_feature():
    """This test requires Triton server."""
    pass
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[test]"
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
      
      - name: Run integration tests
        run: |
          # Start services
          docker-compose up -d grpc-api
          # Wait for services
          sleep 10
          # Run tests
          pytest tests/integration/ -v
```

## Troubleshooting

### Tests Failing Due to Missing Services

If integration or functional tests fail:

1. Check if required services are running:
   ```bash
   docker-compose ps
   ```

2. Check service logs:
   ```bash
   docker-compose logs triton-server
   docker-compose logs grpc-api
   ```

3. Skip tests that require unavailable services:
   ```bash
   pytest -m "not requires_triton"
   ```

### Import Errors

If you get import errors, ensure the package is installed:
```bash
pip install -e .
```

### Slow Tests

Skip slow tests during development:
```bash
pytest -m "not slow"
```

## Test Metrics

Run tests with timing information:
```bash
pytest --durations=10  # Show 10 slowest tests
```

## Best Practices

1. **Keep tests independent**: Each test should be able to run standalone
2. **Use fixtures**: Share setup code via fixtures in conftest.py
3. **Clear assertions**: Use descriptive assertion messages
4. **Mock external dependencies**: Use `pytest-mock` for unit tests
5. **Test edge cases**: Include tests for error conditions and edge cases
6. **Document complex tests**: Add docstrings explaining what's being tested

## Migration Notes

Tests have been consolidated from:
- `api/tests/` → `tests/unit/` and `tests/integration/`
- `triton_server/scripts/test_*.py` → `tests/functional/`

Old test locations are deprecated and should not be used for new tests.
