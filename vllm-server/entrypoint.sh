#!/usr/bin/env bash
# Launch SGLang to serve Qwen/Qwen3.5-9B with an OpenAI-compatible API.
# SGLang's RadixAttention (prefix caching) is enabled by default.
set -euo pipefail

echo "[sglang] Model name    : ${MODEL_NAME}"
echo "[sglang] Model alias   : ${SERVED_MODEL_NAME}"
echo "[sglang] Quantization  : ${QUANTIZATION:-none}"
echo "[sglang] Max context   : ${MAX_MODEL_LEN}"
echo "[sglang] Mem fraction  : ${MEM_FRACTION_STATIC}"
echo "[sglang] Tensor parallel: ${TENSOR_PARALLEL_SIZE}"
echo ""

# Optional quantization flag — omitted entirely when QUANTIZATION is unset
# or empty so SGLang defaults to BF16.
QUANT_ARGS=()
if [[ -n "${QUANTIZATION:-}" ]]; then
    QUANT_ARGS=(--quantization "${QUANTIZATION}")
fi

# YaRN RoPE scaling is required when MAX_MODEL_LEN exceeds Qwen3.5-9B's
# native context of 262,144.  Scale factor = requested / native.
NATIVE_CTX=262144
OVERRIDE_ARGS=()
if (( MAX_MODEL_LEN > NATIVE_CTX )); then
    FACTOR=$(python3 -c "print(${MAX_MODEL_LEN} / ${NATIVE_CTX})")
    echo "[sglang] MAX_MODEL_LEN (${MAX_MODEL_LEN}) > native (${NATIVE_CTX}): enabling YaRN (factor=${FACTOR})"
    OVERRIDE_ARGS=(
        --json-model-override-args
        "{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true, \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"factor\": ${FACTOR}, \"original_max_position_embeddings\": ${NATIVE_CTX}}}}"
    )
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
fi

exec python -m sglang.launch_server \
    --model-path             "${MODEL_NAME}" \
    --served-model-name      "${SERVED_MODEL_NAME}" \
    --host                   0.0.0.0 \
    --port                   8000 \
    --tp-size                "${TENSOR_PARALLEL_SIZE}" \
    --mem-fraction-static    "${MEM_FRACTION_STATIC}" \
    --context-length         "${MAX_MODEL_LEN}" \
    --reasoning-parser       qwen3 \
    --tool-call-parser       qwen3_coder \
    "${QUANT_ARGS[@]}" \
    "${OVERRIDE_ARGS[@]}"
