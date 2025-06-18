import os
import subprocess
from packaging.version import parse, Version
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


CUDA_HOME = os.environ.get("CUDA_HOME")
print(f"CUDA HOME = {CUDA_HOME}")
cc_flag = ["-gencode"]

if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    print(f"Bare metal version = {bare_metal_version}")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("arch=compute_90,code=sm_90")


ext_modules = [
    CUDAExtension(
        name="local_causal_conv1d_cuda",
        sources=[
            # "csrc/causal_conv1d_common.h",
            "csrc/causal_conv1d_bwd.cu",
            "csrc/causal_conv1d_fwd.cu",
            "csrc/causal_conv1d_update.cu",
            "csrc/causal_conv1d.cpp",
            # "csrc/causal_conv1d.h",
            # "csrc/static_switch.h",
        ],
        extra_compile_args={
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ]
        },
    )
]

(
    setup(
        name="hyena_ops",
        version="0.1.1",
        packages=find_packages(),
        python_requires=">=3.10,<3.13",
        package_dir={"": "."},
    ),
)


setup(
    name="local_causal_conv1d",
    version="0.1.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.10,<3.13",
    package_dir={"": "."},
    install_requires=[
        "torch",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
