# Contributing to Flaschen Depot

Thank you for your interest in contributing to Flaschen Depot! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/flaschen_depot.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/flaschen_depot
isort src/flaschen_depot

# Check linting
flake8 src/flaschen_depot
```

## Code Style

- Follow PEP 8 style guide
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Write docstrings for all functions and classes
- Add type hints where appropriate

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Maintain or improve code coverage
- Use pytest for testing

## Pull Request Process

1. Update README.md with details of changes if needed
2. Update documentation if adding new features
3. Ensure all tests pass
4. Request review from maintainers
5. Address any feedback from reviewers

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## Questions?

If you have questions, please open an issue for discussion.
