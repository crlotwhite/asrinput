# ASRInput Build Makefile
# Requires: uv, pyinstaller

# Variables
PYTHON := uv run python
PIP := uv pip
APP_NAME := asrinput
MAIN_FILE := main.py
DIST_DIR := dist
BUILD_DIR := build
SPEC_FILE := $(APP_NAME).spec

# PyInstaller options
PYINSTALLER := uv run pyinstaller
PYINSTALLER_OPTS := --onefile --windowed --name $(APP_NAME)
PYINSTALLER_OPTS += --add-data "$(shell uv run python -c "import customtkinter; print(customtkinter.__path__[0])");customtkinter"

# Default target
.PHONY: all
all: build

# Install dependencies
.PHONY: deps
deps:
	uv sync
	uv add pyinstaller

# Build executable
.PHONY: build
build: deps
	$(PYINSTALLER) $(PYINSTALLER_OPTS) $(MAIN_FILE)
	@echo "Build complete: $(DIST_DIR)/$(APP_NAME).exe"

# Build with console (for debugging)
.PHONY: build-debug
build-debug: deps
	uv run pyinstaller --onefile --console --name $(APP_NAME)-debug $(MAIN_FILE)
	@echo "Debug build complete: $(DIST_DIR)/$(APP_NAME)-debug.exe"

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR) $(SPEC_FILE) *.spec __pycache__
	@echo "Cleaned build artifacts"

# Run the application
.PHONY: run
run:
	$(PYTHON) $(MAIN_FILE)

# Show help
.PHONY: help
help:
	@echo "ASRInput Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build the executable (default)"
	@echo "  build       - Build the executable"
	@echo "  build-debug - Build with console for debugging"
	@echo "  deps        - Install dependencies"
	@echo "  clean       - Remove build artifacts"
	@echo "  run         - Run the application"
	@echo "  help        - Show this help message"
