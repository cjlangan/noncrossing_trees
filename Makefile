VENV_PATH := venv
PYTHON := $(VENV_PATH)/bin/python3
PIP := $(VENV_PATH)/bin/pip
PROJECT_NAME := noncrossing_trees
PROJECT_ROOT := $(shell pwd)/..
SCRIPTS_DIR := .

.DEFAULT_GOAL := run

# Run main.py as a module (recommended)
run: check-venv
	@echo "Running $(PROJECT_NAME).main as module with virtual environment..."
	cd .. && $(PROJECT_NAME)/$(PYTHON) -m $(PROJECT_NAME).main

# Alternative: Run main.py directly with PYTHONPATH
run-direct: check-venv
	@echo "Running Scripts/main.py directly with virtual environment..."
	PYTHONPATH=$(PROJECT_ROOT) $(PYTHON) main.py

# Run from Scripts directory (if using relative imports)
run-local: check-venv
	@echo "Running main.py from Scripts directory with virtual environment..."
	PYTHONPATH=$(PROJECT_ROOT) $(PYTHON) main.py

# Run the demo script
demo: check-venv
	@echo "Running demo script with virtual environment..."
	cd .. && PYTHONPATH=$(PROJECT_ROOT) Scripts/$(PYTHON) -m Scripts.examples.demo

# Install in development mode
install-dev: check-venv
	@echo "Installing package in development mode..."
	cd .. && Scripts/$(PIP) install -e .

# Check if virtual environment exists
check-venv:
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "Virtual environment not found at $(VENV_PATH)"; \
		echo "Creating one..."; \
		python3 -m venv $(VENV_PATH); \
		$(MAKE) install-deps; \
	else \
		echo "Virtual environment found at $(VENV_PATH)"; \
		if [ ! -f "$(PYTHON)" ]; then \
			echo "Python executable not found in virtual environment, please recreate it."; \
			exit 1; \
		fi; \
		echo "Using Python executable: $(PYTHON)"; \
		echo "Using PIP executable: $(PIP)"; \
		$(PIP) install --upgrade pip setuptools wheel; \
	fi

# Install requirements if requirements.txt exists
install-deps:
	@if [ -f requirements.txt ]; then \
		echo "Installing dependencies from requirements.txt..."; \
		$(PIP) install -r requirements.txt; \
	elif [ -f ../requirements.txt ]; then \
		echo "Installing dependencies from ../requirements.txt..."; \
		$(PIP) install -r ../requirements.txt; \
	else \
		echo "No requirements.txt found"; \
	fi

# Activate virtual environment (for manual use)
activate:
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_PATH)/bin/activate"

clean:
	@echo "Cleaning Python cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleaning images..."
	find . -type f -name "*.png" -delete

debug: check-venv
	@echo "Project root: $(PROJECT_ROOT)"
	@echo "Scripts directory: $(shell pwd)"
	@echo "Virtual environment: $(VENV_PATH)"
	@echo "Python path: $(PYTHON)"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "PYTHONPATH will be set to: $(PROJECT_ROOT)"
	@echo ""
	@echo "Project structure:"
	@find . -name "*.py" | head -10

help:
	@echo "Available targets:"
	@echo "  run         - Run Scripts.main as module (recommended)"
	@echo "  run-direct  - Run Scripts/main.py directly with PYTHONPATH"
	@echo "  run-local   - Run from Scripts directory"
	@echo "  demo        - Run the demo script"
	@echo "  install-dev - Install package in development mode"
	@echo "  install-deps- Install dependencies from requirements.txt"
	@echo "  check-venv  - Check if virtual environment exists"
	@echo "  activate    - Show command to activate virtual environment"
	@echo "  clean       - Clean Python cache files"
	@echo "  debug       - Show project info and structure"
	@echo "  help        - Show this help message"

.PHONY: run run-direct run-local demo install-dev install-deps check-venv activate clean debug help
