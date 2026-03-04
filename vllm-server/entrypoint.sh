#!/usr/bin/env bash
# Download the GGUF model file (if not already cached) then launch llama-server.
set -euo pipefail

echo "[llama-server] Model repo : ${MODEL_REPO}"
echo "[llama-server] GGUF file  : ${GGUF_FILE}"
echo "[llama-server] Model alias: ${SERVED_MODEL_NAME}"
echo "[llama-server] Context len: ${CTX_SIZE}"
echo "[llama-server] GPU layers : ${N_GPU_LAYERS}"
echo "[llama-server] Parallel   : ${PARALLEL_REQUESTS}"
echo "[llama-server] HF cache   : ${HF_HOME}"
echo ""

# Download the GGUF file from HuggingFace (hf_hub_download is idempotent —
# returns the cached path immediately if the file is already present).
echo "[llama-server] Resolving model file from HuggingFace Hub..."
GGUF_PATH=$(python3 - <<'EOF'
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=os.environ["MODEL_REPO"],
    filename=os.environ["GGUF_FILE"],
    token=os.environ.get("HF_TOKEN"),   # optional; model is public
)
print(path)
EOF
)
echo "[llama-server] Model file  : ${GGUF_PATH}"
echo ""

# llama-server is the binary name in recent llama.cpp releases.
# The --jinja flag enables the Jinja2 chat template embedded in the GGUF file,
# which is required for correct tool-call formatting with Qwen3.5.
#
# YaRN RoPE scaling is required when CTX_SIZE exceeds the model's native context
# length (262,144 for Qwen3.5-9B).  The scale factor is ctx/native; enabling YaRN
# on shorter contexts degrades performance, so we gate it conditionally.
NATIVE_CTX=262144
YARN_ARGS=()
if (( CTX_SIZE > NATIVE_CTX )); then
    ROPE_SCALE=$(python3 -c "print(${CTX_SIZE} / ${NATIVE_CTX})")
    echo "[llama-server] CTX_SIZE (${CTX_SIZE}) > native (${NATIVE_CTX}): enabling YaRN (scale=${ROPE_SCALE})"
    YARN_ARGS=(
        --rope-scaling  yarn
        --rope-scale    "${ROPE_SCALE}"
        --yarn-orig-ctx "${NATIVE_CTX}"
    )
fi

exec /app/llama-server \
    --model          "${GGUF_PATH}" \
    --alias          "${SERVED_MODEL_NAME}" \
    --host           0.0.0.0 \
    --port           8000 \
    --ctx-size       "${CTX_SIZE}" \
    --n-gpu-layers   "${N_GPU_LAYERS}" \
    --parallel       "${PARALLEL_REQUESTS}" \
    "${YARN_ARGS[@]}" \
    --jinja
