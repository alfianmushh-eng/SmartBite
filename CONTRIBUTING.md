# Contributing to SmartBite

We welcome contributions!

## Development Setup

git clone https://github.com/alfianmushh-eng/SmartBite.git
cd SmartBite
pip install -e ".[dev]"
pre-commit install

## Code Style

- Python 3.11+ type hints everywhere
- Black formatting (line length 100)
- Flake8 linting
- Pre-commit hooks enforced

## Testing

pytest tests/ -v --cov=smartbite

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Commit with conventional commit messages
4. Push and open a PR

By contributing, you agree that your contributions will be licensed under the MIT License.

