from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="vtx",
    version="1.0.6",
    description="Inference and utilities for convolutional multi-hybrid models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Michael Poli",
    url="http://github.com/zymrael/vortex",
    license="Apache-2.0",
    packages=find_packages(where="vortex"),
)