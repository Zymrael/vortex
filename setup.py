import os
import sys
import platform
import subprocess
import warnings
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

def get_cuda_bare_metal_version(cuda_dir: Path):
    """Return raw nvcc output and the parsed bare metal version."""
    nvcc_path = cuda_dir / "bin" / "nvcc"
    raw_output = subprocess.check_output([str(nvcc_path), "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version

if CUDA_HOME is None:
    raise RuntimeError("CUDA_HOME is not set; NVCC may not be available. Please set it to the path of your CUDA installation and make sure PyTorch is built with CUDA support.")
else:
    # Check that the installed CUDA is supported (>= 11.7)
    _, bare_metal_version = get_cuda_bare_metal_version(Path(CUDA_HOME))
    if bare_metal_version < Version("11.7"):
        raise RuntimeError("FlashAttention requires CUDA 11.7 or higher. Check nvcc -V.")
    
# Set NVCC extra compile arguments.
nvcc_args = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-gencode", "arch=compute_80,code=sm_80"
]
if CUDA_HOME is not None and bare_metal_version >= Version("11.8"):
    nvcc_args += ["-gencode", "arch=compute_90,code=sm_90"]

setup_dir = Path(__file__).parent.resolve()
os.chdir(setup_dir)

flash_attn_sources = [
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/flash_api.cpp"),
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu"),
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu"),
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu"),
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu"),
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu"),
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu"),
]

if 'CUDNN_HOME' not in os.environ:
    os.environ['CUDNN_HOME'] = '/usr/local/cuda'  # Default location

include_dirs = [
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn"),
    str(setup_dir / "vortex/ops/attn/csrc/flash_attn/src"),
    str(setup_dir / "vortex/ops/cutlass/include/"),
    str(setup_dir / "vortex/ops/cutlass/include/cute"),
    str(setup_dir / "vortex/ops/cutlass/include/cutlass"),
    str(os.environ.get('CUDNN_HOME', '')),
]

ext_modules = [
    CUDAExtension(
        name="flash_attn_2_cuda",
        sources=flash_attn_sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_args,
        },
        include_dirs=include_dirs,
    )
]

setup(
    name="vtx",
    version="1.0.6",
    description="Inference and utilities for convolutional multi-hybrid models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Michael Poli",
    url="http://github.com/zymrael/vortex",
    packages=find_packages(include=['vortex', 'vortex.*']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
    zip_safe=False,
    include_package_data=False,
)
