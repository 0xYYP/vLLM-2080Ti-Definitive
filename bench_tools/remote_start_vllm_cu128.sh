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
: "${ENFORCE_EAGER:=0}"
: "${KV_CACHE_DTYPE:=}"
: "${NO_ASYNC_SCHEDULING:=0}"
: "${DISABLE_HYBRID_KV_CACHE_MANAGER:=0}"
: "${DISABLE_PREFIX_CACHING:=0}"
: "${LANGUAGE_MODEL_ONLY:=0}"
: "${SKIP_MM_PROFILING:=0}"
: "${MM_LIMIT_JSON:=}"
: "${VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK:=0}"
: "${VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE:=0}"
: "${DISABLE_LOG_STATS:=0}"
: "${FLASHINFER_EXPERIMENT_ROOT:=}"

: "${STABLE_ROOT:=/opt/vllm-2080ti}"
: "${FLASHQLA_ROOT:=/data/stable/FlashQLA-SM70-SM75}"
: "${RUN_USER:=vllm}"
: "${RUN_HOME:=/var/lib/vllm}"
: "${CUDA_HOME:=/usr/local/cuda-12.8}"
: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${CUDA_DEVICE_ORDER:=PCI_BUS_ID}"
: "${SERVICE_STOP_LIST:=}"
: "${KILL_PROCESS_PATTERN:=VLLM::|vllm|ptxas|triton|sglang.launch_server|sglang::scheduler|sglang::detokenizer}"
: "${STOP_EXISTING_PROCESSES:=1}"

STAMP=$(date +%Y%m%d-%H%M%S)
SAFE_NAME=$(printf '%s' "$SERVED_NAME" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
LOG="${STABLE_ROOT}/vllm-${SAFE_NAME}-${STAMP}.log"
CURRENT="${STABLE_ROOT}/vllm-${SAFE_NAME}.current"

if [[ -n "$SERVICE_STOP_LIST" ]]; then
  # shellcheck disable=SC2086
  systemctl stop $SERVICE_STOP_LIST || true
fi

if [[ "$STOP_EXISTING_PROCESSES" == "1" ]]; then
  pkill -9 -u "$RUN_USER" -f "$KILL_PROCESS_PATTERN" || true
fi
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

if [[ "$DISABLE_LOG_STATS" == "1" ]]; then
  common_args+=(--disable-log-stats)
fi

if [[ -n "$MM_LIMIT_JSON" ]]; then
  common_args+=(--limit-mm-per-prompt "$MM_LIMIT_JSON")
elif [[ "$MODEL_FAMILY" == qwen* ]]; then
  common_args+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
elif [[ "$MODEL_FAMILY" == gemma* ]]; then
  common_args+=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
fi

if [[ "$MODEL_FAMILY" == qwen* && -n "$MM_LIMIT_JSON" ]]; then
  common_args+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
fi

if [[ "$MODEL_FAMILY" == qwen* && -z "$CHAT_TEMPLATE_FILE" && -s "$STABLE_ROOT/chat_template_no_think_ragent6.jinja" ]]; then
  CHAT_TEMPLATE_FILE="$STABLE_ROOT/chat_template_no_think_ragent6.jinja"
elif [[ "$MODEL_FAMILY" == qwen* && -z "$CHAT_TEMPLATE_FILE" && -s "$STABLE_ROOT/chat_template_no_think.jinja" ]]; then
  CHAT_TEMPLATE_FILE="$STABLE_ROOT/chat_template_no_think.jinja"
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
chown "$RUN_USER:$RUN_USER" "$CURRENT" "$LOG"

args_file=$(mktemp)
printf '%q ' "${common_args[@]}" > "$args_file"
args_text=$(cat "$args_file")
rm -f "$args_file"

runuser -u "$RUN_USER" --preserve-environment -- bash -c "
set -euo pipefail
cd '$STABLE_ROOT'
export HOME='$RUN_HOME'
source .venv/bin/activate
export CUDA_HOME='$CUDA_HOME'
export CUDA_PATH=\"\$CUDA_HOME\"
export CUDACXX=\"\$CUDA_HOME/bin/nvcc\"
export TORCH_CUDA_ARCH_LIST='7.5'
export VENV_SITE='$STABLE_ROOT/.venv/lib/python3.11/site-packages'
export PATH='$STABLE_ROOT/.venv/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
export CUDA_DEVICE_ORDER='$CUDA_DEVICE_ORDER'
if [[ -n '$FLASHINFER_EXPERIMENT_ROOT' ]]; then
  export PYTHONPATH='$FLASHINFER_EXPERIMENT_ROOT:$STABLE_ROOT:$FLASHQLA_ROOT'
else
  export PYTHONPATH=\"$STABLE_ROOT:$FLASHQLA_ROOT\"
fi
export LD_LIBRARY_PATH=\"\$VENV_SITE/torch/lib:\${LD_LIBRARY_PATH:-}\"
export FLASHINFER_ENABLE_AOT=1
export TORCHINDUCTOR_CACHE_DIR='$STABLE_ROOT/torchinductor-cache'
export TRITON_CACHE_DIR='$STABLE_ROOT/triton-cache'
export PYTHONUNBUFFERED=1
for _v in \
  VLLM_GEMMA4_DEBUG_KV_PAGE_SIZES \
  VLLM_GEMMA4_TEXT_ONLY_FLASHINFER \
  VLLM_GEMMA4_SM75_TRITON_TILE16 \
  VLLM_TURBOQUANT_FLASHINFER_BACKEND \
  VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL \
  VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE \
  VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH \
  VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS \
  VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK \
  VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS \
  VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM \
  VLLM_SM75_SPEC_SYNC_MODE \
  VLLM_USE_FLASHINFER_SAMPLER \
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
  VLLM_INT8KV_FA_PREFILL \
  VLLM_INT8KV_FA_CONTINUATION_DEQUANT \
  VLLM_INT8KV_FA_CASCADE_DEQUANT \
  VLLM_INT8KV_FA_CASCADE_TILE_TOKENS \
  VLLM_INT8KV_FA_DIRECT_PAGED \
  VLLM_DISABLE_COMPILE_CACHE; do
  if [[ -n \"\${!_v:-}\" ]]; then
    export \"\$_v=\${!_v}\"
  fi
done
nohup python3 -m vllm.entrypoints.openai.api_server $args_text > '$LOG' 2>&1 &
echo \$! > '$STABLE_ROOT/vllm-${SAFE_NAME}.pid'
"
chown "$RUN_USER:$RUN_USER" "$LOG" "$STABLE_ROOT/vllm-${SAFE_NAME}.pid"

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
