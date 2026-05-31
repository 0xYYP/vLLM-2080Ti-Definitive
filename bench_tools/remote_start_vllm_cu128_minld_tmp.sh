#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR required}"
: "${SERVED_NAME:?SERVED_NAME required}"
: "${MAX_MODEL_LEN:=65536}"
: "${GPU_UTIL:=0.90}"
: "${MTP_K:=0}"
: "${SPECULATIVE_CONFIG:=}"
: "${PORT:=19206}"
: "${MODEL_FAMILY:=qwen}"
: "${MAX_BATCHED_TOKENS:=8192}"
: "${MAX_NUM_SEQS:=1}"
: "${CHAT_TEMPLATE_FILE:=}"
: "${QUANTIZATION:=}"
: "${COMPILATION_CONFIG_JSON:=}"
: "${ATTENTION_BACKEND:=}"
: "${REASONING_PARSER:=}"
: "${DEFAULT_CHAT_TEMPLATE_KWARGS:=}"
: "${VLLM_SM75_TURING_FA_PREFILL:=0}"
: "${ENFORCE_EAGER:=0}"
: "${KV_CACHE_DTYPE:=}"
: "${NO_ASYNC_SCHEDULING:=0}"
: "${DISABLE_HYBRID_KV_CACHE_MANAGER:=0}"
: "${DISABLE_PREFIX_CACHING:=0}"
: "${LANGUAGE_MODEL_ONLY:=0}"
: "${SKIP_MM_PROFILING:=0}"
: "${VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK:=0}"
: "${VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE:=0}"

EXP=/data/stable/vllm-sm75-tp2-cu128
FLASHQLA=/data/stable/FlashQLA-SM70-SM75
FLASH_ATTN_TURING=/data/stable/flash-attention-turing-sm75-compare-20260520/src
FLASH_ATTN_TURING_BUILD=$FLASH_ATTN_TURING/build/lib.linux-x86_64-cpython-311
STAMP=$(date +%Y%m%d-%H%M%S)
SAFE_NAME=$(printf '%s' "$SERVED_NAME" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
LOG="$EXP/vllm-${SAFE_NAME}-${STAMP}.log"
CURRENT="$EXP/vllm-${SAFE_NAME}.current"

systemctl stop miniclaw-minit1-dense-qwen27-proxy.service miniclaw-minit1-dense-qwen27.service \
  miniclaw-minit2-dense-gemma31-proxy.service miniclaw-minit2-dense-gemma31.service || true
pkill -9 -u dietpi -f "VLLM::|vllm|ptxas|triton|sglang.launch_server|sglang::scheduler|sglang::detokenizer" || true
sleep 2

common_args=(
  --host 127.0.0.1
  --port "$PORT"
  --model "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --dtype half
  --tensor-parallel-size 2
  --generation-config vllm
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --enable-chunked-prefill
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
)

if [[ -n "$QUANTIZATION" ]]; then
  common_args+=(--quantization "$QUANTIZATION")
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  common_args+=(--enforce-eager)
fi

if [[ -n "$KV_CACHE_DTYPE" ]]; then
  common_args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if [[ "$NO_ASYNC_SCHEDULING" == "1" ]]; then
  common_args+=(--no-async-scheduling)
fi

if [[ "$DISABLE_HYBRID_KV_CACHE_MANAGER" == "1" ]]; then
  common_args+=(--disable-hybrid-kv-cache-manager)
fi

if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
  common_args+=(--no-enable-prefix-caching)
fi

if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
  common_args+=(--language-model-only)
fi

if [[ "$SKIP_MM_PROFILING" == "1" ]]; then
  common_args+=(--skip-mm-profiling)
fi

if [[ -n "$ATTENTION_BACKEND" ]]; then
  common_args+=(--attention-backend "$ATTENTION_BACKEND")
fi

if [[ -n "$REASONING_PARSER" ]]; then
  common_args+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]]; then
  common_args+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
fi

if [[ "$MODEL_FAMILY" == qwen* ]]; then
  common_args+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  if [[ -z "$CHAT_TEMPLATE_FILE" && -s "$EXP/chat_template_no_think_ragent6.jinja" ]]; then
    CHAT_TEMPLATE_FILE="$EXP/chat_template_no_think_ragent6.jinja"
  fi
elif [[ "$MODEL_FAMILY" == gemma* ]]; then
  # Keep Gemma route text-only to avoid multimodal warmup stalls/noise during
  # performance-focused kernels validation.
  common_args+=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
fi

if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
  common_args+=(--chat-template "$CHAT_TEMPLATE_FILE")
fi

capture=$((MTP_K + 1))
if [[ -n "$SPECULATIVE_CONFIG" ]]; then
  common_args+=(--speculative-config "$SPECULATIVE_CONFIG")
elif (( MTP_K > 0 )); then
  common_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
fi

if [[ -n "$COMPILATION_CONFIG_JSON" ]]; then
  common_args+=(--compilation-config "$COMPILATION_CONFIG_JSON")
elif [[ -n "$SPECULATIVE_CONFIG" || "$MTP_K" -gt 0 ]]; then
  common_args+=(
    --compilation-config "{\"cudagraph_capture_sizes\":[${capture}],\"max_cudagraph_capture_size\":${capture}}"
  )
else
  common_args+=(--compilation-config '{"cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}')
fi

printf '%s\n' "$LOG" > "$CURRENT"
touch "$LOG"
chown dietpi:dietpi "$CURRENT" "$LOG"

args_file=$(mktemp)
printf '%q ' "${common_args[@]}" > "$args_file"
args_text=$(cat "$args_file")
rm -f "$args_file"

runuser -u dietpi --preserve-environment -- bash -c "
set -euo pipefail
cd '$EXP'
export HOME='/home/dietpi'
source .venv/bin/activate
# Do not inherit root-only PATH entries (e.g. /root/.local/bin), which can
# trigger subprocess PermissionError in torch/inductor repro helpers.
export CUDA_HOME='/usr/local/cuda-12.8'
export CUDA_PATH="\$CUDA_HOME"
export CUDACXX="\$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST='7.5'
export VENV_SITE='$EXP/.venv/lib/python3.11/site-packages'
export PATH='$EXP/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export CUDA_VISIBLE_DEVICES=1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$EXP:$FLASHQLA:$FLASH_ATTN_TURING:$FLASH_ATTN_TURING_BUILD"
export LD_LIBRARY_PATH="\$VENV_SITE/torch/lib:\${LD_LIBRARY_PATH:-}"
export FLASHINFER_ENABLE_AOT=1
export TORCHINDUCTOR_CACHE_DIR='$EXP/torchinductor-cache'
export TRITON_CACHE_DIR='$EXP/triton-cache'
export PYTHONUNBUFFERED=1
export VLLM_SM75_TURING_FA_PREFILL="$VLLM_SM75_TURING_FA_PREFILL"
for _v in \
  VLLM_GEMMA4_DEBUG_KV_PAGE_SIZES \
  VLLM_GEMMA4_TEXT_ONLY_FLASHINFER \
  VLLM_GEMMA4_SM75_TRITON_TILE16 \
  VLLM_TURBOQUANT_FLASHINFER_BACKEND \
  VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL \
  VLLM_TURBOQUANT_USE_TURING_PREFILL \
  VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH \
  VLLM_GEMMA4_TQ4NC_MTP_KV_SHARING_FIX \
  VLLM_GEMMA4_TQ4NC_GROUP_UNIFORM_TYPES \
  VLLM_GEMMA4_TQ4NC_SHARED_FI_WORKSPACE \
  VLLM_GEMMA4_TQ4NC_MIXED_METADATA_COMPAT \
  VLLM_GEMMA4_TQ4NC_HEAD512_PREFILL_SDPA_FALLBACK \
  VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK \
  VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE \
  VLLM_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK \
  VLLM_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK \
  VLLM_QWOPUS_MTP_BF16_DRAFT \
  VLLM_DISABLE_COMPILE_CACHE; do
  if [[ -n "\${!_v:-}" ]]; then
    export "\$_v=\${!_v}"
  fi
done
nohup python3 -m vllm.entrypoints.openai.api_server $args_text > '$LOG' 2>&1 &
echo \$! > '$EXP/vllm-${SAFE_NAME}.pid'
"
chown dietpi:dietpi "$LOG" "$EXP/vllm-${SAFE_NAME}.pid"

deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "READY log=$LOG"
    exit 0
  fi
  if grep -E "RuntimeError|ValueError|CUDA error|OutOfMemoryError|No supported config format|EngineCore encountered a fatal error|EngineDeadError" "$LOG" >/dev/null 2>&1; then
    echo "FAILED log=$LOG"
    tail -n 120 "$LOG" || true
    exit 1
  fi
  sleep 5
done

echo "TIMEOUT log=$LOG"
tail -n 160 "$LOG" || true
exit 1
