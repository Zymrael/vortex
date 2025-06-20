<div align="center">

# Vortex

</div>


This repository contains implementations of computational primitives for convolutional multi-hybrid models and layers: Hyena-[SE, MR, LI], StripedHyena 2, Evo 2.

For training, please refer to the [savanna](https://github.com/Zymrael/savanna/) project.

## Interface

There are two main ways to interface with `vortex`:

1. Use `vortex` as the inference engine for pre-trained multi-hybrids such as [Evo 2 40B](configs/evo2-40b-1m.yml). In this case, we recommend installing `vortex` in a new environment (see below).
2. Import from `vortex` specific classes, kernels or utilities to work with custom convolutional multi-hybrids. For example,sourcing utilities from `hyena_ops.interface`.


## 1. Pip install (easiest)

The simplest way to install `vortex` is from PyPi. This requires you to have dependencies already installed.

```bash
pip install vtx
```
or you can install Vortex after cloning the repository:
```bash
pip install .
```
Note this will take a few minutes to compile, which can be sped up by being on more CPUs.

## 2. Quick install for vortex ops

```bash
make setup-vortex-ops
```
Note that this does not install all dependencies required to run autoregressive inference with larger pre-trained models.

## 3. Building a custom development environment

### Using conda, venv or uv

To run e2e installation in a uv environment, use the following command:
```bash
make setup-full
```
Note that the `setup-full` step will compile various CUDA kernels, which usually takes at most several minutes. It may be necessary to customize CUDA header and library paths in `Makefile`.  

## 4. Running in a Docker environment

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


