#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/opt/vllm}
PROFILE_DIR=${PROFILE_DIR:-"$ROOT/profiles"}
TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
MODEL_DIR=${MODEL_DIR:-/models/Qwen3.6-27B-GPTQ-Int4}
PROFILE=${PROFILE:-qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env}
MODE=${MODE:-fast}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export FLASHINFER_ENABLE_AOT="${FLASHINFER_ENABLE_AOT:-1}"
export FLASHQLA_DIR="${FLASHQLA_DIR:-$ROOT/FlashQLA-SM70-SM75}"
export PYTHONPATH="$ROOT:$FLASHQLA_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export PYTHONUNBUFFERED=1

read_profile_value() {
    local file=$1
    local key=$2
    awk -F= -v key="$key" '
        $1 == key {
            value = substr($0, index($0, "=") + 1)
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            gsub(/^'\''|'\''$/, "", value)
            gsub(/^"|"$/, "", value)
            print value
            exit
        }
    ' "$file"
}

profile_key_is_global() {
    case "$1" in
        MODEL_DIR|PROFILE_DIR|PROFILE|MODE|PORT|SERVICE_SCOPE|GPU_DEVICES|TP_SIZE|\
        CHAT_TEMPLATE_FILE|CHAT_TEMPLATE_PRESET|TEMPLATE_DIR|REASONING_PARSER|\
        DEFAULT_CHAT_TEMPLATE_KWARGS|REASONING_MODE|REASONING_BUDGET|\
        ENABLE_AUTO_TOOL_CHOICE|TOOL_CALL_PARSER|TOOL_PARSER_PLUGIN|\
        ENABLE_PREFIX_CACHING|ENABLE_PROMPT_TOKENS_DETAILS|DISABLE_PREFIX_CACHING|\
        VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH|VLLM_ENFORCE_STRICT_TOOL_CALLING)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_profile_file() {
    if [[ -n "${PROFILE_FILE:-}" ]]; then
        printf '%s\n' "$PROFILE_FILE"
        return 0
    fi
    if [[ -f "$PROFILE_DIR/$PROFILE" ]]; then
        printf '%s\n' "$PROFILE_DIR/$PROFILE"
        return 0
    fi
    if [[ -f "$PROFILE_DIR/${PROFILE%.env}.env" ]]; then
        printf '%s\n' "$PROFILE_DIR/${PROFILE%.env}.env"
        return 0
    fi
    return 1
}

apply_profile_overrides() {
    local file=$1
    local key value

    while IFS= read -r key; do
        [[ -n "$key" ]] || continue
        profile_key_is_global "$key" && continue
        [[ -n "${!key+x}" ]] && continue
        value=$(read_profile_value "$file" "$key")
        printf -v "$key" '%s' "$value"
        export "$key"
    done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)
}

normalize_bool() {
    case "${1,,}" in
        1|yes|y|true|on) echo 1 ;;
        *) echo 0 ;;
    esac
}

set_mode_defaults() {
    case "$MODE" in
        normal)
            export ENFORCE_EAGER=${ENFORCE_EAGER:-0}
            export DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
            export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
            export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}
            ;;
        fast)
            export ENFORCE_EAGER=${ENFORCE_EAGER:-0}
            export DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
            export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
            export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-1}
            ;;
        aggressive)
            export ENFORCE_EAGER=${ENFORCE_EAGER:-0}
            export DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
            export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-nosync}
            export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-1}
            ;;
        safe)
            export ENFORCE_EAGER=${ENFORCE_EAGER:-1}
            export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
            export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}
            ;;
        *)
            echo "ERROR: MODE must be safe, normal, fast, or aggressive; got $MODE" >&2
            exit 1
            ;;
    esac
}

set_route_defaults() {
    local model_dir_l=${MODEL_DIR,,}

    MODEL_FAMILY=${MODEL_FAMILY:-qwen}
    SERVED_NAME=${SERVED_NAME:-${MODEL_DIR##*/}}
    if [[ -z "${QUANTIZATION:-}" ]]; then
        case "$model_dir_l" in
            *gptq*|*int4*|*4bit*) QUANTIZATION=gptq_marlin ;;
            *awq*) QUANTIZATION=awq_marlin ;;
            *fp8*) QUANTIZATION=fp8 ;;
            *quark*) QUANTIZATION=quark ;;
        esac
    fi
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
    GPU_UTIL=${GPU_UTIL:-0.90}
    MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    MTP_K=${MTP_K:-0}
    TP_SIZE=${TP_SIZE:-2}
    ENABLE_PREFIX_CACHING=$(normalize_bool "${ENABLE_PREFIX_CACHING:-1}")
    ENABLE_PROMPT_TOKENS_DETAILS=$(normalize_bool "${ENABLE_PROMPT_TOKENS_DETAILS:-1}")
    DISABLE_PREFIX_CACHING=$(normalize_bool "${DISABLE_PREFIX_CACHING:-0}")

    if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
        ENABLE_PREFIX_CACHING=0
    elif [[ "$ENABLE_PREFIX_CACHING" == "1" && "$MODEL_FAMILY" == qwen* && -z "${MAMBA_CACHE_MODE:-}" ]]; then
        MAMBA_CACHE_MODE=align
    fi

    if [[ "$MODEL_FAMILY" == qwen* && -z "${REASONING_PARSER:-}" ]]; then
        REASONING_PARSER=qwen3
    fi

    if [[ -z "${MM_LIMIT_JSON:-}" ]]; then
        LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-1}
        SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-1}
    fi
}

set_turboquant_defaults() {
    [[ "${KV_CACHE_DTYPE:-}" == turboquant_* ]] || return 0

    local reserve=${VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS:-}
    if [[ -z "$reserve" ]]; then
        reserve=65536
        if [[ "$ENABLE_PREFIX_CACHING" == "1" && "$MAX_MODEL_LEN" =~ ^[0-9]+$ && "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]] \
            && (( MAX_MODEL_LEN >= 240000 )) && (( MAX_NUM_SEQS <= 1 )); then
            reserve=262144
        fi
    fi

    export VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL=${VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL:-1}
    export VLLM_TURBOQUANT_FLASHINFER_BACKEND=${VLLM_TURBOQUANT_FLASHINFER_BACKEND:-fa2}
    export VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=$reserve
    export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=${VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE:-1}
    export VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK=${VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK:-512}
    export VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS=${VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS:-16777216}
    export VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=${VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH:-1}
}

if [[ $# -gt 0 ]]; then
    exec python -m vllm.entrypoints.openai.api_server "$@"
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR" >&2
    exit 1
fi

profile_file=$(resolve_profile_file) || {
    echo "ERROR: Profile not found: ${PROFILE_FILE:-$PROFILE} under $PROFILE_DIR" >&2
    exit 1
}

apply_profile_overrides "$profile_file"
set_mode_defaults
set_route_defaults
set_turboquant_defaults

args=(
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL_DIR"
    --served-model-name "$SERVED_NAME"
    --dtype half
    --tensor-parallel-size "$TP_SIZE"
    --generation-config vllm
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
)

[[ -n "${QUANTIZATION:-}" ]] && args+=(--quantization "$QUANTIZATION")
[[ -n "${KV_CACHE_DTYPE:-}" ]] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${MAMBA_CACHE_MODE:-}" ]] && args+=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
[[ "${ENFORCE_EAGER:-0}" == "1" ]] && args+=(--enforce-eager)
if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    args+=(--enable-prefix-caching)
else
    args+=(--no-enable-prefix-caching)
fi
[[ "$ENABLE_PROMPT_TOKENS_DETAILS" == "1" ]] && args+=(--enable-prompt-tokens-details)
[[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]] && args+=(--language-model-only)
[[ "${SKIP_MM_PROFILING:-0}" == "1" ]] && args+=(--skip-mm-profiling)
[[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]] && args+=(--disable-custom-all-reduce)
[[ "${DISABLE_LOG_STATS:-0}" == "1" ]] && args+=(--disable-log-stats)
[[ -n "${REASONING_PARSER:-}" && "${REASONING_PARSER:-}" != "off" ]] && args+=(--reasoning-parser "$REASONING_PARSER")
[[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]] && args+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
[[ -n "${TOOL_PARSER_PLUGIN:-}" ]] && args+=(--tool-parser-plugin "$TOOL_PARSER_PLUGIN")
[[ -n "${TOOL_CALL_PARSER:-}" ]] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
[[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]] && args+=(--enable-auto-tool-choice)
[[ -n "${HF_OVERRIDES_JSON:-}" ]] && args+=(--hf-overrides "$HF_OVERRIDES_JSON")
[[ -n "${MM_LIMIT_JSON:-}" ]] && args+=(--limit-mm-per-prompt "$MM_LIMIT_JSON")

if [[ -n "${ADDITIONAL_CONFIG_JSON:-}" ]]; then
    args+=(--additional-config "$ADDITIONAL_CONFIG_JSON")
elif [[ "$MODEL_FAMILY" == qwen* ]]; then
    args+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
fi

if [[ -n "${CHAT_TEMPLATE_PRESET:-}" && -f "$TEMPLATE_DIR/$CHAT_TEMPLATE_PRESET" ]]; then
    CHAT_TEMPLATE_FILE="$TEMPLATE_DIR/$CHAT_TEMPLATE_PRESET"
fi
[[ -n "${CHAT_TEMPLATE_FILE:-}" ]] && args+=(--chat-template "$CHAT_TEMPLATE_FILE")

capture=$((MTP_K + 1))
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
    args+=(--speculative-config "$SPECULATIVE_CONFIG")
elif (( MTP_K > 0 )); then
    args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
fi

if [[ -n "${COMPILATION_CONFIG_JSON:-}" ]]; then
    args+=(--compilation-config "$COMPILATION_CONFIG_JSON")
elif (( MTP_K > 0 )); then
    args+=(--compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${capture}],\"max_cudagraph_capture_size\":${capture}}")
else
    args+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}')
fi

echo "=========================================="
echo " vLLM 2080Ti Definitive Docker"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader || true
echo "Model Dir: $MODEL_DIR"
echo "Profile: $profile_file"
echo "Mode: $MODE"
echo "Served Name: $SERVED_NAME"
echo "KV: ${KV_CACHE_DTYPE:-fp16}"
echo "Context: $MAX_MODEL_LEN"
echo "MTP: $MTP_K"
echo "Port: $PORT"
printf 'Command: python -m vllm.entrypoints.openai.api_server'
printf ' %q' "${args[@]}"
echo
echo "=========================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

exec python -m vllm.entrypoints.openai.api_server "${args[@]}"
