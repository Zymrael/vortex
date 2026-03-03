<div align="center">

# Vortex

</div>


This repository contains implementations of computational primitives for convolutional multi-hybrid models and layers: Hyena-[SE, MR, LI], StripedHyena 2, Evo 2.

For training, please refer to the [savanna](https://github.com/Zymrael/savanna/) project.

## Interface

There are two main ways to interface with `vortex`:

1. Use `vortex` as the inference engine for pre-trained multi-hybrids such as [Evo 2 40B](configs/evo2-40b-1m.yml). In this case, we recommend installing `vortex` in a new environment (see below).
2. Import from `vortex` specific classes, kernels or utilities to work with custom convolutional multi-hybrids. For example,sourcing utilities from `hyena_ops.interface`.

## 1. Pip install

The simplest way to install `vortex` is from PyPi or github.

### Requirements
Vortex requires PyTorch and Transformer Engine, and it is strongly recommended to also use Flash Attention. For detailed instructions and compatibility information, please refer to their respective GitHub repositories. Note TransformerEngine requires python 3.12 and has these additional [system requirements](https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#system-requirements).

*   **[PyTorch with CUDA](https://github.com/pytorch/pytorch):** Ensure you have a CUDA-enabled PyTorch installation compatible with your NVIDIA drivers.
*   **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine):** NVIDIA's Transformer Engine.
*   **[Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main):** For optimized attention operations.

Example of installing prerequisites. We recommended using `conda` for easy installation of Transformer Engine:
```bash
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch==2.3.0
pip install flash-attn==2.8.0.post2
```

### Installing vortex
After installing the requirements, you can install vortex:
```bash
pip install vtx
```
or you can install vortex after cloning the repository:
```bash
pip install .
```

### Installing without Transformer Engine (Evo 2 7b only)

If your environment cannot build Transformer Engine (e.g., incompatible CUDA/Python version),
you can skip it. The Evo 2 7b models (8k, 262k, 1m context) work without TE in pure bfloat16:

```bash
pip install vtx
```

Note: Evo 2 40b, 20b, and 1b models require Transformer Engine for FP8 support and will
not load without it.

## 2. Quick install for vortex ops

```bash
make setup-vortex-ops
```
Note that this does not install all dependencies required to run autoregressive inference with larger pre-trained models.

## 3. Running in a Docker environment

Docker is one of the easiest ways to get started with Vortex (and Evo 2). The
Docker environment does not depend on the currently installed CUDA version and
ensures that major dependencies (such as PyTorch and Transformer Engine) are
pinned to specific versions, which is beneficial for reproducibility.

To run Evo 2 40B generation sample, simply run `./run`.

To run Evo 2 7B generation sample: `sz=7 ./run`.

To run tests: `./run ./run_tests`.

To interactively execute commands in docker environment: `./run bash`.

### Generation quickstart

```bash
python3 generate.py \
    --config_path <PATH_TO_CONFIG> \
    --checkpoint_path <PATH_TO_CHECKPOINT> \
    --input_file <PATH_TO_INPUT_FILE> \
    --cached_generation
```
`--cached_generation` activates KV-caching and custom caching for different variants of Hyena layers, reducing peak memory usage and latency.


## Acknowledgements

Vortex was developed by Michael Poli ([Zymrael](https://github.com/Zymrael)) and Garyk Brixi ([garykbrixi](https://github.com/garykbrixi)). Vortex maintainers include Michael Poli ([Zymrael](https://github.com/Zymrael)), Garyk Brixi ([garykbrixi](https://github.com/garykbrixi)), Anton Vorontsov ([antonvnv](https://github.com/antonvnv)) with contributions from Amy Lu ([amyxlu](https://github.com/amyxlu)), Jerome Ku ([jeromeku](https://github.com/jeromeku)).

## Cite

If you find this project useful, consider citing the following [references](CITE.md).
