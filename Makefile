# Makefile for turn-taking-pipeline

.PHONY: help install install-dev test clean setup-venv

# Default target
help:
	@echo "Turn-Taking Pipeline - Available commands:"
	@echo ""
	@echo "  make setup-venv     Create Python virtual environment"
	@echo "  make install        Install package and dependencies"
	@echo "  make install-dev    Install with development tools"
	@echo "  make test           Run tests (when implemented)"
	@echo "  make clean          Remove build artifacts and cache"
	@echo "  make format         Format code with black"
	@echo "  make lint           Run flake8 linter"
	@echo ""

# Create virtual environment
setup-venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Virtual environment created!"
	@echo ""
	@echo "To activate:"
	@echo "  source venv/bin/activate"
	@echo ""
	@echo "Then run: make install"

# Install package
install:
	pip install --upgrade pip
	pip install -e .

# Install with dev dependencies
install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"

# Install with all optional dependencies
install-all:
	pip install --upgrade pip
	pip install -e ".[dev,llm,modeling]"

# Run tests
test:
	pytest tests/ -v

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Format code
format:
	black src/ scripts/ tests/

# Lint code
lint:
	flake8 src/ scripts/ tests/ --max-line-length=100

# Run pipeline on test subject
test-pipeline:
	@echo "Testing pipeline on sub-01, run-01..."
	python scripts/run_pipeline.py --config config/config.yaml --subjects sub-01 --runs 1 --dry-run
