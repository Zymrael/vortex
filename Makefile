.PHONY: setup clean dev lint format submodules

PYTHON := python3.11
UV := uv
VENV_BIN := $(CURDIR)/.venv/bin
CONDA_ENV_NAME := vortex

PYTHON_VERSION := $(shell $(PYTHON) -c 'import sys; print(sys.version_info[0])')
TARGET_PYTHON_VERSION := 3.11
CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE_PATH := $(CUDA_PATH)/include
CUDA_LIB_PATH := $(CUDA_PATH)/lib64
CPATH := $(CUDA_INCLUDE_PATH):/usr/local/cuda/include
CUDACXX := /usr/local/cuda/bin/nvcc
CUDA_HOME := $(CUDA_PATH)

_detect_cuda_path:
ifndef CUDA_PATH
    NVCC_BIN := $(shell which nvcc 2>/dev/null)
    ifneq ($(NVCC_BIN),)
        # Derive CUDA_PATH by going one directory up from nvcc's bin directory
        CUDA_PATH := $(dir $(NVCC_BIN))..
        CUDA_PATH := $(realpath $(CUDA_PATH))
    endif
endif

_check_env_enabled:
ifneq ($(VIRTUAL_ENV),)
    $(info Detected active Python venv at $(VIRTUAL_ENV))
    ENV_ACTIVE := UV        # Using a UV/venv environment
else ifdef CONDA_PREFIX
    $(info Detected Conda environment "$(CONDA_DEFAULT_ENV)")
	ifeq ($(CONDA_DEFAULT_ENV), base)
		$(info Base Conda environment is active; please create a new environment for vortex)
		exit 1
	endif
    ENV_ACTIVE := CONDA     # Using a Conda environment
else
    ENV_ACTIVE := NONE      # No environment active
endif

_setup_missing_env:
ifeq ($(ENV_ACTIVE), NONE)
    ifneq ($(shell which uv 2>/dev/null),)
        $(info No env active; creating a new uv virtual environment)
        uv venv .venv --python=$(TARGET_PYTHON_VERSION)
        source .venv/bin/activate    # activate it for the remainder of the make
    else ifneq ($(shell which conda 2>/dev/null),)
        $(info No env active; creating a new Conda environment)
        conda env create -n $(CONDA_ENV_NAME) -f environment.yml 
        conda activate $(CONDA_ENV_NAME)
    else
        $(error Neither uv nor Conda is available for environment creation)
    endif
endif


setup-full: submodules
	pip install ninja cmake pybind11 numpy psutil
	pip install -e . --config-settings=build-script=local_setup.py
	pip install transformer_engine[pytorch]==1.13.0 --no-build-isolation 
	cd vortex/ops/attn && MAX_JOBS=32 pip install -v -e  . --no-build-isolation

setup-vortex-ops: submodules _check_env_enabled _setup_missing_env
	pip install ninja cmake pybind11 numpy psutil
	pip install -e . --config-settings=build-script=local_setup.py

setup-vortex-ops-hyenax: _check_env_enabled _setup_missing_env
	cd vortex/ops/hyenax && pip install -e .

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
	black -l 120 .

rebuild: clean setup 
