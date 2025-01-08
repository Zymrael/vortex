.PHONY: setup clean dev lint format submodules

PYTHON := python3.11
UV := uv
VENV_BIN := $(CURDIR)/.venv/bin

setup: submodules
	$(UV) venv
	$(UV) pip install -e .
	@echo "Run this in your shell: export PATH=$(VENV_BIN):$$PATH"

submodules:
	git submodule update --init --recursive

clean:
	rm -rf .venv
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

rebuild: clean setup 