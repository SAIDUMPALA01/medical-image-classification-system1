# Contributing to Pneumonia Detection CNN

Thank you for your interest in contributing to this project! We welcome contributions from everyone. This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We are committed to providing a welcoming environment for all contributors.

## Ways to Contribute

### 1. Report Bugs
- Use the GitHub Issues page
- Provide a clear title and description
- Include steps to reproduce the issue
- Specify your environment (Python version, OS, etc.)
- Attach any relevant error messages or screenshots

### 2. Suggest Enhancements
- Check existing issues to avoid duplicates
- Clearly describe the enhancement and its benefits
- Provide examples or mockups if applicable
- Explain how this aligns with the project goals

### 3. Submit Code Changes
- Fork the repository
- Create a feature branch
- Make your changes with clear commits
- Submit a pull request with detailed description

### 4. Improve Documentation
- Fix typos and clarity issues
- Add missing sections or examples
- Update outdated information
- Translate documentation to other languages

### 5. Improve Tests
- Add unit tests for new features
- Improve test coverage
- Add integration tests
- Fix failing tests

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv or conda)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/your-bug-fix-name
```

### 2. Make Your Changes
- Write clean, readable code
- Add docstrings to functions and classes
- Add comments for complex logic
- Follow PEP 8 style guidelines

### 3. Write Tests
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_model.py::test_model_creation
```

### 4. Lint Your Code
```bash
# Format code with black
black src/ tests/

# Check for linting issues
flake8 src/ tests/
pylint src/

# Run all checks
make lint
```

### 5. Commit Your Changes
```bash
# Add changes
git add .

# Commit with clear message
git commit -m "feat: add new preprocessing method

- Add rotation augmentation
- Add zoom augmentation
- Update tests
"
```

### Commit Message Guidelines
- Use conventional commits: `type: description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Keep first line under 50 characters
- Provide detailed explanation in body

## Pull Request Process

### Before Submitting
1. Ensure all tests pass: `pytest`
2. Ensure code is formatted: `black`
3. Check for linting issues: `flake8`
4. Update documentation if needed
5. Add tests for new features
6. Rebase on main branch

### Creating a PR
1. Push your feature branch to your fork
2. Create a Pull Request on GitHub
3. Fill in the PR template completely
4. Link any related issues
5. Add descriptive title and description

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Closes #(issue number)

## Changes Made
- Detailed change 1
- Detailed change 2

## Testing
- Describe tests added
- Describe how to test changes

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] No new warnings generated
```

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints for function signatures

### Example
```python
from typing import Tuple, Optional

def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Preprocess medical image for model input.
    
    Args:
        image_path: Path to image file
        target_size: Output image dimensions
        normalize: Whether to normalize pixel values
        
    Returns:
        Processed image array or None if error
        
    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image cannot be processed
    """
    pass
```

### Documentation Style
- Use docstrings for all public functions
- Use Google-style docstrings
- Include examples where helpful
- Document parameters, returns, and exceptions

## Testing Guidelines

### Test Organization
```python
# tests/test_preprocessing.py
import pytest
from src.preprocessing import preprocess_image

class TestPreprocessing:
    """Test preprocessing functions."""
    
    def test_image_resizing(self):
        """Test image resizing functionality."""
        # Arrange
        input_size = (500, 500)
        expected_size = (224, 224)
        
        # Act
        result = preprocess_image("test.jpg", target_size=expected_size)
        
        # Assert
        assert result.shape == (expected_size[0], expected_size[1], 3)
```

### Test Coverage
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use pytest fixtures for setup
- Mock external dependencies

## Documentation

### When to Update Docs
- New features require documentation
- API changes need updates
- Bug fixes may need clarification
- Add examples for complex features

### Documentation Format
- Use Markdown for README and guides
- Use Google-style docstrings in code
- Include code examples
- Add diagrams for complex concepts

## Reporting Security Issues

**Do not** create public issues for security vulnerabilities.

Email security concerns to: security@example.com

Please include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

## Attribution

Contributors will be recognized in:
- README contributors section
- Release notes
- GitHub contributors page

## Questions?

- Check existing documentation
- Review similar issues and PRs
- Ask in issue discussions
- Contact maintainers

## Recognition

### Contributors Hall of Fame
We recognize all contributors! Your work helps improve this project for everyone.

### Acknowledgment Tiers
- **Code Contributors**: Feature implementations, bug fixes
- **Documentation Contributors**: Docs, guides, examples
- **Community Contributors**: Bug reports, feedback, advocacy
- **Maintainers**: Project maintenance and leadership

## Resources

- [Git Documentation](https://git-scm.com/doc)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Guides](https://guides.github.com/)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to pneumonia detection research and healthcare AI!** 🎉

---

*Last Updated: February 2025*
