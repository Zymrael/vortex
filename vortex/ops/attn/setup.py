# modified from https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
# Copyright (c) 2023, Tri Dao.

import sys
import warnings
import os
import shutil
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
)


this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(this_dir))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
FORCE_BUILD = True


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip("-").replace("-", "+"))


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(f"{global_option} was requested, but hipcc was not found.")


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")


def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx940", "gfx941", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


cmdclass = {}
ext_modules = []

subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

check_if_cuda_home_none("flash_attn")
# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
print(f"CUDA_HOME = {CUDA_HOME}")
if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.7"):
        raise RuntimeError(
            "FlashAttention is only supported on CUDA 11.7 and above.  "
            "Note: make sure nvcc has a supported version by running nvcc -V."
        )
# cc_flag.append("-gencode")
# cc_flag.append("arch=compute_75,code=sm_75")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
if CUDA_HOME is not None:
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")
ext_modules.append(
    CUDAExtension(
        name="flash_attn_2_cuda",
        sources=[
            "csrc/flash_attn/flash_api.cpp",
            "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    # "--ptxas-options=-v",
                    # "--ptxas-options=-O2",
                    # "-lineinfo",
                    # "-DFLASHATTENTION_DISABLE_BACKWARD",
                    "-DFLASHATTENTION_DISABLE_DROPOUT",
                    "-DFLASHATTENTION_DISABLE_ALIBI",
                    # "-DFLASHATTENTION_DISABLE_SOFTCAP",
                    # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                    # "-DFLASHATTENTION_DISABLE_LOCAL",
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "flash_attn",
            Path(this_dir) / "csrc" / "flash_attn" / "src",
            Path(this_dir).parent / "cutlass" / "include",
        ],
    )
)


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        return super().run()


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024**3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    name="local_flash_attn",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    ext_modules=ext_modules,
    cmdclass=(
        {
            "bdist_wheel": CachedWheelsCommand,
            "build_ext": NinjaBuildExtension,
        }
        if ext_modules
        else {
            "bdist_wheel": CachedWheelsCommand,
        }
    ),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
    zip_safe=False,
    include_package_data=True,
)
