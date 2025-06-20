from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="vortex.ops.hyenax",
    version="1.0.0",
    description="Kernels for Hyena-X",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Michael Poli",
    url="http://github.com/zymrael/vortex",
    license="Apache-2.0",
    packages=find_packages(where="vortex/ops/hyenax"),
)
