# llama.cpp model server for ARC-AGI solver
#
# Serves a 6-bit GGUF quantised Qwen3.5-9B model via an OpenAI-compatible
# API on port 8000.  Uses llama.cpp instead of vLLM because Qwen3.5's novel
# architecture (Gated DeltaNet + sparse MoE, February 2026) is not yet
# supported by vLLM's GGUF loading path.
#
# The model file (~7.5 GB) is downloaded from HuggingFace at first startup and
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
# Override the quantisation variant, context length, or parallel slots:
#   docker run --gpus all --rm \
#     -p 8000:8000 \
#     -v ~/.cache/huggingface:/models \
#     -e GGUF_FILE=Qwen3.5-9B-UD-Q6_K_XL.gguf \
#     -e CTX_SIZE=262144 \
#     -e PARALLEL_REQUESTS=8 \
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
ENV MODEL_REPO=unsloth/Qwen3.5-9B-GGUF

# Which 6-bit GGUF variant to use.  Options (smallest → largest / lowest → highest quality):
#   Qwen3.5-9B-Q6_K.gguf       7.46 GB  — standard 6-bit K-quant (default)
#   Qwen3.5-9B-UD-Q6_K_XL.gguf 8.76 GB  — Unsloth Dynamic 2.0 (superior accuracy)
ENV GGUF_FILE=Qwen3.5-9B-Q6_K.gguf

# Model alias exposed via the /v1/models endpoint and used in API requests.
# Must match the value of LOCAL_LLM_MODEL in the solver environment.
ENV SERVED_MODEL_NAME=unsloth/Qwen3.5-9B-GGUF

# Maximum context length in tokens.  Qwen3.5-9B natively supports 262,144 tokens.
# Values above 262,144 automatically engage YaRN RoPE scaling in entrypoint.sh.
ENV CTX_SIZE=400000

# Number of model layers to offload to GPU.  99 = all layers (recommended).
# Set to 0 for CPU-only inference.
ENV N_GPU_LAYERS=99

# Number of request slots for batched parallel inference.  Each additional slot
# reserves an extra KV-cache region in VRAM (~ctx_size × kv_dim bytes), so
# raise this until VRAM is exhausted or GPU utilisation saturates.
ENV PARALLEL_REQUESTS=2

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
