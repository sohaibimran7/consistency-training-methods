# Training image for Vast.ai (x86 CUDA): vLLM base + this repo's deps.
# The vllm-openai image ships matched torch+vllm+CUDA; we add uv and the repo.
#
# Build & push (or let Vast pull a public tag):
#   docker build -t <you>/ctm-train:latest -f infra/vastai/Dockerfile .
#
# The repo itself is cloned at runtime by provision.sh (keeps the image generic
# across branches); only dependencies are baked here.
FROM vllm/vllm-openai:latest

# The base image's entrypoint launches the API server — this is a shell box.
ENTRYPOINT []
CMD ["/bin/bash"]

RUN apt-get update && apt-get install -y --no-install-recommends git tmux rsync \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Pre-bake the heavy Python deps so instance start is fast. requirements.txt is
# copied alone first to keep this layer cacheable across code changes.
WORKDIR /opt/ctm-deps
COPY requirements.txt .
RUN grep -v '^grugstream' requirements.txt | pip install --no-cache-dir -r /dev/stdin \
    && pip install --no-cache-dir inspect-ai peft

WORKDIR /workspace
