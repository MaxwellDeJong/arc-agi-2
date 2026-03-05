# SGLang model server for ARC-AGI solver
#
# Serves Qwen/Qwen3.5-9B via an OpenAI-compatible API on port 8000.
# SGLang's RadixAttention gives native prefix-caching, which is especially
# valuable for Qwen3.5's hybrid Gated DeltaNet + Gated Attention layers.
#
# FP8 weight quantization (~9 GB) is used instead of GGUF: the `qwen35`
# GGUF architecture is not yet supported by transformers, so GGUF files can
# only be loaded by llama.cpp.  FP8 is comparable to Q8_0 GGUF in size
# while being a first-class format in SGLang.
#
# Build:
#   docker build -t arc-llm-server ./vllm-server
#
# Run (single GPU, 24 GB VRAM):
#   docker run --gpus all --rm \
#     -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     arc-llm-server
#
# Override context length, memory fraction, or disable quantization:
#   docker run --gpus all --rm \
#     -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -e MAX_MODEL_LEN=262144 \
#     -e QUANTIZATION=fp8 \
#     arc-llm-server
#
# Base image docs: https://docs.sglang.ai/start/install.html

FROM lmsysorg/sglang:latest

# Qwen3.5 requires SGLang from the main branch and transformers from source.
# Two separate pip invocations are needed: SGLang pins transformers==4.57.1,
# so we let it install that first, then force-upgrade transformers from git.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        "sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python" \
    && pip install --no-cache-dir --force-reinstall \
        "transformers @ git+https://github.com/huggingface/transformers.git" \
        "nvidia-cudnn-cu12==9.16.0.29"

# ── Model configuration ────────────────────────────────────────────────────
# HuggingFace model ID to serve.
ENV MODEL_NAME=Qwen/Qwen3.5-9B

# Model alias exposed via the /v1/models endpoint and used in API requests.
# Must match the value of LOCAL_LLM_MODEL in the solver environment.
ENV SERVED_MODEL_NAME=Qwen/Qwen3.5-9B

# Quantization — fp8 requires Hopper (H100+) and is disabled by default.
# Leave empty to run in full BF16 (~18 GB).
ENV QUANTIZATION=

# Maximum context length in tokens.  Qwen3.5-9B natively supports 262,144.
# BF16 weights consume ~18 GB on a 24 GB card, leaving ~6 GB for KV cache.
# 16384 fits comfortably; raise if running on a larger or quantized model.
ENV MAX_MODEL_LEN=16384

# Fraction of GPU memory reserved for static KV cache allocation.
# 0.95 gives the 18 GB BF16 model enough headroom on a 24 GB card:
# 0.95 * 24 GB = 22.8 GB static pool → ~4.8 GB available for KV cache.
ENV MEM_FRACTION_STATIC=0.95

# Tensor parallel degree — 1 for a single GPU, 2+ for multi-GPU.
ENV TENSOR_PARALLEL_SIZE=1

# Bypass flashinfer AOT cache / SGLang version-check mismatches that can
# occur when installing from the main branch over the base image.
ENV FLASHINFER_DISABLE_VERSION_CHECK=1
ENV SGLANG_DISABLE_CUDNN_CHECK=1

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
