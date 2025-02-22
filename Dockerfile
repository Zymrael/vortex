from nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as base
run apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

arg REQUIREMENTS=requirements.txt
copy ${REQUIREMENTS} .
run --mount=type=cache,target=/root/.cache \
    pip install -r ${REQUIREMENTS}

copy vortex/ops /usr/src/vortex-ops
run --mount=type=cache,target=/root/.cache \
    cd /usr/src/vortex-ops/attn && MAX_JOBS=32 pip install -v -e  . --no-build-isolation

workdir /workdir
