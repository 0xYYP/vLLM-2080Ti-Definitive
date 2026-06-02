#!/usr/bin/env bash
set -euo pipefail

ROOT=${STABLE_ROOT:-/opt/vllm-2080ti}
START="$ROOT/bench_tools/remote_start_vllm_cu128.sh"
: "${MODEL_ROOT:=/models}"
: "${PROFILE:?PROFILE required: qwen27-awq-mtp3 | qwen27-awq-mtp3-peak | qwen27-awq-mtp3-tq4nc-fi | qwen27-awq-mtp3-int8kv | qwen27-awq-mm-nomtp | qwen27-awq-mm-mtp3 | qwen27-awq-mm-tq4nc-nomtp | qwen27-awq-mm-tq4nc-mtp3 | heretic27-gptq-mm-nomtp | heretic27-gptq-mm-tq4nc-mtp3 | gemma4-gptq-tq4nc-mtp3 | gemma4-gptq-tq4nc-nomtp | gemma4-gptq-mm-nomtp | gemma4-gptq-mm-mtp3}"
: "${PORT:=19266}"

case "$PROFILE" in
  qwen27-awq-mtp3)
    export VLLM_QWOPUS_MTP_BF16_DRAFT=1
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mtp3-cu128-stable}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.90}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    ;;
  qwen27-awq-mtp3-peak)
    export VLLM_QWOPUS_MTP_BF16_DRAFT=1
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mtp3-peak-cu128}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.86}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-nosync}
    export DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
    ;;
  qwen27-awq-mtp3-tq4nc-fi)
    export VLLM_QWOPUS_MTP_BF16_DRAFT=1
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mtp3-tq4nc-fi-cu128-stable}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export KV_CACHE_DTYPE=turboquant_4bit_nc
    # Agent-style structured workloads can exceed 8192 when the harness reserves
    # a large output budget. Keep the real-use TQ profile default at 16k.
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
    export GPU_UTIL=${GPU_UTIL:-0.90}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export NO_ASYNC_SCHEDULING=1
    export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
    export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1
    export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
    export LANGUAGE_MODEL_ONLY=1
    export SKIP_MM_PROFILING=1
    # Qwen-family head_dim=256 regresses badly if SM75 TQ prefill is forced
    # away from FlashInfer/FA2. Keep this scoped to the Qwen-family TQ profile.
    export VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM=0
    # Keep speculative continuation on the decode-style path for q_len>1
    # CUDAGraph verification chunks.
    export VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1
    ;;
  qwen27-awq-mtp3-int8kv)
    export VLLM_QWOPUS_MTP_BF16_DRAFT=1
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mtp3-int8kv-cu128-stable}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export KV_CACHE_DTYPE=int8_per_token_head
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
    export GPU_UTIL=${GPU_UTIL:-0.90}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export VLLM_INT8KV_FA_PREFILL=${VLLM_INT8KV_FA_PREFILL:-1}
    export VLLM_INT8KV_FA_CONTINUATION_DEQUANT=${VLLM_INT8KV_FA_CONTINUATION_DEQUANT:-1}
    export VLLM_INT8KV_FA_CASCADE_DEQUANT=${VLLM_INT8KV_FA_CASCADE_DEQUANT:-1}
    export VLLM_INT8KV_FA_CASCADE_TILE_TOKENS=${VLLM_INT8KV_FA_CASCADE_TILE_TOKENS:-65536}
    ;;
  qwen27-awq-mm-nomtp)
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mm-nomtp-cu128-experimental}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=0
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    ;;
  qwen27-awq-mm-mtp3)
    export VLLM_QWOPUS_MTP_BF16_DRAFT=1
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mm-mtp3-cu128-experimental}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    ;;
  qwen27-awq-mm-tq4nc-nomtp)
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mm-tq4nc-nomtp-cu128-experimental}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export KV_CACHE_DTYPE=turboquant_4bit_nc
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=0
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
    export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1
    export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
    export VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM=0
    export VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1
    ;;
  qwen27-awq-mm-tq4nc-mtp3)
    export VLLM_QWOPUS_MTP_BF16_DRAFT=1
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-awq}
    export SERVED_NAME=${SERVED_NAME:-qwen27-awq-mm-tq4nc-mtp3-cu128-experimental}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=awq_marlin
    export KV_CACHE_DTYPE=turboquant_4bit_nc
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
    export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1
    export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
    export VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM=0
    export VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1
    ;;
  heretic27-gptq-mm-nomtp)
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-heretic-gptq}
    export SERVED_NAME=${SERVED_NAME:-heretic27-gptq-mm-nomtp-cu128-experimental}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=gptq_marlin
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=0
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    ;;
  heretic27-gptq-mm-tq4nc-mtp3)
    export VLLM_QWOPUS_MTP_BF16_DRAFT=1
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/qwen-family-27b-heretic-gptq}
    export SERVED_NAME=${SERVED_NAME:-heretic27-gptq-mm-tq4nc-mtp3-cu128-experimental}
    export MODEL_FAMILY=qwen
    export QUANTIZATION=gptq_marlin
    export KV_CACHE_DTYPE=turboquant_4bit_nc
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
    export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1
    export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
    export VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM=0
    export VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1
    ;;
  gemma4-gptq-tq4nc-mtp3)
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/gemma4-family-31b-gptq}
    export SERVED_NAME=${SERVED_NAME:-gemma4-gptq-tq4nc-mtp3-cu128-stable}
    export MODEL_FAMILY=gemma4
    export QUANTIZATION=compressed-tensors
    export KV_CACHE_DTYPE=turboquant_4bit_nc
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export GEMMA4_ASSISTANT_MODEL=${GEMMA4_ASSISTANT_MODEL:-$MODEL_ROOT/gemma4-family-assistant}
    export SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-"{\"method\":\"mtp\",\"model\":\"$GEMMA4_ASSISTANT_MODEL\",\"num_speculative_tokens\":3,\"draft_tensor_parallel_size\":1,\"disable_padded_drafter_batch\":true}"}
    export NO_ASYNC_SCHEDULING=1
    export DISABLE_HYBRID_KV_CACHE_MANAGER=1
    export DISABLE_PREFIX_CACHING=1
    export LANGUAGE_MODEL_ONLY=1
    export SKIP_MM_PROFILING=1
    export VLLM_DISABLE_COMPILE_CACHE=1
    export VLLM_GEMMA4_TEXT_ONLY_FLASHINFER=1
    export VLLM_GEMMA4_SM75_TRITON_TILE16=1
    export VLLM_GEMMA4_TQ4NC_MTP_KV_SHARING_FIX=1
    export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
    export VLLM_GEMMA4_TQ4NC_GROUP_UNIFORM_TYPES=1
    export VLLM_GEMMA4_TQ4NC_SHARED_FI_WORKSPACE=1
    export VLLM_GEMMA4_TQ4NC_MIXED_METADATA_COMPAT=1
    export VLLM_GEMMA4_TQ4NC_HEAD512_PREFILL_SDPA_FALLBACK=1
    export VLLM_GEMMA4_SM75_SDPA_PREFILL512=${VLLM_GEMMA4_SM75_SDPA_PREFILL512:-1}
    export VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM=${VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM:-0}
    export VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK=1
    export VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE=1
    export VLLM_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK=${VLLM_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK:-0}
    export VLLM_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK=${VLLM_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK:-0}
    ;;
  gemma4-gptq-tq4nc-nomtp)
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/gemma4-family-31b-gptq}
    export SERVED_NAME=${SERVED_NAME:-gemma4-gptq-tq4nc-nomtp-cu128-stable}
    export MODEL_FAMILY=gemma4
    export QUANTIZATION=compressed-tensors
    export KV_CACHE_DTYPE=turboquant_4bit_nc
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=0
    export NO_ASYNC_SCHEDULING=1
    export DISABLE_HYBRID_KV_CACHE_MANAGER=1
    export DISABLE_PREFIX_CACHING=1
    export LANGUAGE_MODEL_ONLY=1
    export SKIP_MM_PROFILING=1
    export VLLM_DISABLE_COMPILE_CACHE=1
    export VLLM_GEMMA4_TEXT_ONLY_FLASHINFER=1
    export VLLM_GEMMA4_SM75_TRITON_TILE16=1
    export VLLM_GEMMA4_TQ4NC_MTP_KV_SHARING_FIX=1
    export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
    export VLLM_GEMMA4_TQ4NC_GROUP_UNIFORM_TYPES=1
    export VLLM_GEMMA4_TQ4NC_SHARED_FI_WORKSPACE=1
    export VLLM_GEMMA4_TQ4NC_MIXED_METADATA_COMPAT=1
    export VLLM_GEMMA4_TQ4NC_HEAD512_PREFILL_SDPA_FALLBACK=1
    export VLLM_GEMMA4_SM75_SDPA_PREFILL512=${VLLM_GEMMA4_SM75_SDPA_PREFILL512:-1}
    export VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM=${VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM:-0}
    export VLLM_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK=${VLLM_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK:-0}
    export VLLM_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK=${VLLM_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK:-0}
    ;;
  gemma4-gptq-mm-nomtp)
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/gemma4-family-31b-gptq}
    export SERVED_NAME=${SERVED_NAME:-gemma4-gptq-mm-nomtp-cu128-experimental}
    export MODEL_FAMILY=gemma4
    export QUANTIZATION=compressed-tensors
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=0
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    ;;
  gemma4-gptq-mm-mtp3)
    export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/gemma4-family-31b-gptq}
    export SERVED_NAME=${SERVED_NAME:-gemma4-gptq-mm-mtp3-cu128-experimental}
    export MODEL_FAMILY=gemma4
    export QUANTIZATION=compressed-tensors
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export GPU_UTIL=${GPU_UTIL:-0.82}
    export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    export MTP_K=${MTP_K:-3}
    export GEMMA4_ASSISTANT_MODEL=${GEMMA4_ASSISTANT_MODEL:-$MODEL_ROOT/gemma4-family-assistant}
    export SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-"{\"method\":\"mtp\",\"model\":\"$GEMMA4_ASSISTANT_MODEL\",\"num_speculative_tokens\":3,\"draft_tensor_parallel_size\":1,\"disable_padded_drafter_batch\":true}"}
    export MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    export VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK=1
    export VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE=1
    ;;
  *)
    echo "unknown PROFILE=$PROFILE" >&2
    exit 2
    ;;
esac

export PORT
exec "$START"
