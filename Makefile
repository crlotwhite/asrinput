# ASRInput Makefile (uv/uvx workflow)
# Requires: uv (https://docs.astral.sh/uv/)

UV := uv
PYTHON := $(UV) run python
APP_NAME := asrinput
MAIN_FILE := main.py

.PHONY: run run-uvx deps clean help all

# Default target runs the app using the synced virtualenv
all: run

# Install project dependencies into the managed .venv
deps:
	$(UV) sync

# Run the application using the local environment
run: deps
	$(PYTHON) $(MAIN_FILE)

# Run without a local venv using uvx (ephemeral env)
run-uvx:
	uvx --from . python $(MAIN_FILE)

# Remove caches and old build artifacts
clean:
	- rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	- rm -rf dist build *.spec

# Display available targets
help:
	@echo "ASRInput (uv/uvx)"
	@echo "Targets:"
	@echo "  run        - Sync deps (if needed) and run the app"
	@echo "  run-uvx    - Run via uvx in an ephemeral environment"
	@echo "  deps       - Sync project dependencies"
	@echo "  clean      - Remove caches and build artifacts"
	@echo "  help       - Show this help message"
