from nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as base
run apt-get update && apt-get install -y git python3-pip python3-tomli && rm -rf /var/lib/apt/lists/*

# Extract dependencies from pyproject: can't use pip-tools' pip-compile as it
# requires whole tree to be copied to Docker context, which makes it impossible
# to develop in docker envrionment.
copy pyproject.toml .
run python3 -c 'import tomli;\
                p = tomli.load(open("pyproject.toml", "rb"))["project"];\
                print("\n".join(p["dependencies"] + p.get("optional-dependencies", {"special": []})["special"]))'\
    > requirements.txt
# Must install torch first, as transformer engine build process will need it
run pip install `cat requirements.txt | grep ^torch`
run pip install -r requirements.txt

copy vortex/ops /usr/src/vortex-ops
run cd /usr/src/vortex-ops/attn && MAX_JOBS=32 pip install -v -e  . --no-build-isolation

workdir /workdir
