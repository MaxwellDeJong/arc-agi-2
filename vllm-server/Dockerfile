# llama.cpp model server for ARC-AGI solver
#
# Serves a 4-bit GGUF quantised Qwen3.5-35B-A3B model via an OpenAI-compatible
# API on port 8000.  Uses llama.cpp instead of vLLM because Qwen3.5's novel
# architecture (Gated DeltaNet + sparse MoE, February 2026) is not yet
# supported by vLLM's GGUF loading path.
#
# The model file (~21 GB) is downloaded from HuggingFace at first startup and
# cached in the /models volume so subsequent starts are fast.
#
# Build:
#   docker build -t arc-llm-server ./vllm-server
#
# Run (single GPU, 24 GB VRAM):
#   docker run --gpus all --rm \
#     -p 8000:8000 \
#     -v ~/.cache/huggingface:/models \
#     arc-llm-server
#
# Override the quantisation variant or context length:
#   docker run --gpus all --rm \
#     -p 8000:8000 \
#     -v ~/.cache/huggingface:/models \
#     -e GGUF_FILE=Qwen3.5-35B-A3B-IQ4_XS.gguf \
#     -e CTX_SIZE=65536 \
#     arc-llm-server
#
# CPU-only (slow but functional):
#   docker run --rm \
#     -p 8000:8000 \
#     -v ~/.cache/huggingface:/models \
#     -e N_GPU_LAYERS=0 \
#     arc-llm-server
#
# Base image docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Install Python for HuggingFace Hub model download
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir \
        "huggingface_hub[hf_transfer]>=0.24"

# HF transfer accelerates downloads via the Xet protocol (5–10× faster)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# HuggingFace cache directory — mount a host directory here to persist the
# model across container restarts and avoid re-downloading on every run
ENV HF_HOME=/models

# ── Model configuration ────────────────────────────────────────────────────
# HuggingFace repo containing the GGUF files
ENV MODEL_REPO=unsloth/Qwen3.5-35B-A3B-GGUF

# Which 3-bit GGUF variant to use.  Options (smallest → largest / lowest → highest quality):
#   Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf  14.7 GB  — Unsloth Dynamic, smallest 3-bit
#   Qwen3.5-35B-A3B-Q3_K_S.gguf      15.2 GB  — standard 3-bit K-quant small
#   Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf  15.8 GB  — Unsloth Dynamic 2.0 (better accuracy)
#   Qwen3.5-35B-A3B-Q3_K_M.gguf      16.8 GB  — standard 3-bit K-quant medium (default)
ENV GGUF_FILE=Qwen3.5-35B-A3B-Q3_K_M.gguf

# Model alias exposed via the /v1/models endpoint and used in API requests.
# Must match the value of LOCAL_LLM_MODEL in the solver environment.
ENV SERVED_MODEL_NAME=unsloth/Qwen3.5-35B-A3B-GGUF

# Maximum context length in tokens.
ENV CTX_SIZE=65536

# Number of model layers to offload to GPU.  99 = all layers (recommended).
# Set to 0 for CPU-only inference.
ENV N_GPU_LAYERS=99

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
