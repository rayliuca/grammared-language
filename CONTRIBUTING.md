# Contributing to Grammared Language

Thank you for your interest in contributing to Grammared Language! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to:

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/grammared_language.git
   cd grammared_language
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/rayliuca/grammared_language.git
   ```
4. **Set up development environment** (see [DEVELOPMENT.md](docs/DEVELOPMENT.md))

## How to Contribute

### Reporting Bugs

Before creating a bug report:
- Check existing issues to avoid duplicates
- Gather relevant information (OS, Python version, etc.)

When creating a bug report, include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Screenshots if applicable
- Environment details

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Check if the enhancement is already suggested
- Provide clear use case and benefits
- Be open to discussion

### Contributing Code

Areas where contributions are especially welcome:
- Bug fixes
- New features
- Performance improvements
- Documentation
- Tests
- Model improvements

## Development Process

### 1. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/feature-name` - New features
- `bugfix/bug-description` - Bug fixes
- `docs/doc-update` - Documentation
- `test/test-description` - Test additions
- `refactor/refactor-description` - Code refactoring

### 2. Make Your Changes

- Write clear, readable code
- Follow existing code style
- Add comments where necessary
- Update documentation

### 3. Write Tests

- Add unit tests for new features
- Ensure all tests pass
- Maintain or improve code coverage

```bash
pytest
pytest --cov=api/src --cov-report=html
```

### 4. Commit Your Changes

Use conventional commit messages:

```
type(scope): subject

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Examples:
```bash
git commit -m "feat(api): add support for multiple languages"
git commit -m "fix(triton): resolve connection timeout issue"
git commit -m "docs(readme): update installation instructions"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Python Code Style

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints
- Write docstrings (Google style)

### Formatting

```bash
# Format with black
black api/src/

# Check with flake8
flake8 api/src/ --max-line-length=100

# Type check with mypy
mypy api/src/
```

### Example

```python
from typing import List, Optional


def process_text(
    text: str, 
    language: str = "en-US",
    max_length: Optional[int] = None
) -> List[dict]:
    """Process text for grammar checking.
    
    Args:
        text: Input text to process.
        language: Language code (default: en-US).
        max_length: Maximum text length (optional).
        
    Returns:
        List of processed text chunks.
        
    Raises:
        ValueError: If text is empty or too long.
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    # Implementation
    return []
```

## Testing Guidelines

### Test Organization

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── load/          # Load tests
```

### Writing Tests

```python
import pytest
from api.src.main import app


class TestGrammarAPI:
    """Test suite for grammar API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_check_endpoint(self, client):
        """Test grammar check endpoint."""
        response = client.post("/check", json={
            "text": "Test text",
            "language": "en-US"
        })
        assert response.status_code == 200
        assert "matches" in response.json()
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov

# Specific test file
pytest tests/unit/test_api.py

# Specific test
pytest tests/unit/test_api.py::test_check_endpoint

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Documentation

### Code Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings
- Include type hints

### Project Documentation

When adding features, update:
- README.md (if it affects usage)
- Relevant docs in `docs/`
- API documentation
- Examples

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep documentation up-to-date

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with main

### PR Description

Include:
- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Changes**: List of changes made
- **Testing**: How was this tested?
- **Screenshots**: For UI changes

Template:
```markdown
## Summary
Brief description of changes

## Motivation
Why are these changes needed?

## Changes
- Change 1
- Change 2

## Testing
How were these changes tested?

## Related Issues
Closes #issue_number
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address review comments
4. Maintain a clean commit history
5. Squash commits if necessary

### After Merge

- Delete your branch
- Close related issues
- Update your fork

```bash
git checkout main
git pull upstream main
git push origin main
git branch -d feature/your-feature-name
```

## Development Environment

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed setup instructions.

## Questions?

- Check existing documentation
- Search closed issues
- Open a discussion
- Ask in pull request

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project website (if applicable)

Thank you for contributing to Grammared Language! 🎉
