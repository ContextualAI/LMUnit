# Makefile for LMUnit-Develop linting and formatting
.PHONY: help lint lint-ruff lint-flake8 format format-ruff format-black check install-dev clean

# Default target
help:
	@echo "Available targets:"
	@echo "  help           - Show this help message"
	@echo "  install-dev    - Install development dependencies"
	@echo "  lint           - Run all linting checks (ruff + flake8)"
	@echo "  lint-ruff      - Run ruff linting only"
	@echo "  lint-flake8    - Run flake8 linting only"
	@echo "  format         - Auto-format code with ruff and black"
	@echo "  format-ruff    - Auto-format code with ruff only"
	@echo "  format-black   - Auto-format code with black only"
	@echo "  check          - Run both linting and verify formatting"
	@echo "  clean          - Clean up cache files"

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt

# Comprehensive linting - runs both ruff and flake8
lint: lint-ruff lint-flake8
	@echo "✅ All linting checks completed"

# Ruff linting only
lint-ruff:
	@echo "Running ruff linting..."
	ruff check . --exclude nbs/

# Flake8 linting only  
lint-flake8:
	@echo "Running flake8 linting..."
	flake8 . --exclude=nbs/

# Auto-formatting with both ruff and black
format: format-ruff format-black
	@echo "✅ Code formatting completed"

# Ruff formatting only (recommended - handles both linting and formatting)
format-ruff:
	@echo "Running ruff formatting..."
	ruff check . --fix --exclude nbs/
	ruff format . --exclude nbs/

# Black formatting only
format-black:
	@echo "Running black formatting..."
	black . --exclude='nbs/'
	isort . --skip=nbs

# Check everything - lint and verify formatting
check:
	@echo "Running comprehensive checks..."
	@echo "1. Checking code formatting..."
	ruff format . --check --exclude nbs/
	black . --check --exclude='nbs/'
	@echo "2. Running linting..."
	$(MAKE) lint
	@echo "✅ All checks passed!"

# Clean up cache and temporary files
clean:
	@echo "Cleaning up cache files..."
	find . -type d -name "__pycache__" -not -path "./nbs/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -not -path "./nbs/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -not -path "./nbs/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./nbs/*" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Advanced targets for more granular control
lint-strict:
	@echo "Running strict linting (no ignored rules)..."
	ruff check . --exclude nbs/ --select ALL

type-check:
	@echo "Running type checking with mypy..."
	mypy . --ignore-missing-imports

# Quick format for development
quick-format:
	@echo "Quick formatting with ruff..."
	ruff check . --fix --exclude nbs/ --select I,F401,F841
	ruff format . --exclude nbs/

# Pre-commit style check (comprehensive check)
pre-commit: clean check
	@echo "✅ Pre-commit checks completed successfully!" 