#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/opt/vllm}
export RUNTIME_ROOT=${RUNTIME_ROOT:-"$ROOT"}
export PROFILE_DIR=${PROFILE_DIR:-"$ROOT/profiles"}
export TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
export LOG_DIR=${LOG_DIR:-"$ROOT/run-logs"}
export STATE_FILE=${STATE_FILE:-"$LOG_DIR/start-manager.state"}
export MODEL_DIR=${MODEL_DIR:-/models/Qwen3.6-27B-GPTQ-Int4}
export PROFILE=${PROFILE:-qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env}
export MODE=${MODE:-fast}
export PORT=${PORT:-8000}
export SERVICE_SCOPE=${SERVICE_SCOPE:-lan}

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

pid_is_running() {
    local pid=${1:-}
    [[ -n "$pid" && -d "/proc/$pid" ]]
}

load_manager_state() {
    [[ -f "$STATE_FILE" ]] || return 1
    # shellcheck disable=SC1090
    source "$STATE_FILE"
}

stop_server() {
    local pid=${1:-}
    local pgid

    pid_is_running "$pid" || return 0
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)
    if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
    else
        kill -TERM "$pid" 2>/dev/null || true
    fi
}

wait_for_stop() {
    local pid=${1:-}
    local attempts=${2:-30}

    for _ in $(seq 1 "$attempts"); do
        pid_is_running "$pid" || return 0
        sleep 1
    done
    return 1
}

forward_shutdown() {
    local pid=${SERVER_PID:-}
    local tail_pid=${TAIL_PID:-}

    if [[ -n "$pid" ]]; then
        stop_server "$pid"
        wait_for_stop "$pid" || true
    fi

    if [[ -n "$tail_pid" ]]; then
        kill "$tail_pid" 2>/dev/null || true
        wait "$tail_pid" 2>/dev/null || true
    fi
}

handle_signal() {
    forward_shutdown
    exit 143
}

launcher_reads_only() {
    local arg

    [[ "${DRY_RUN:-0}" == "1" || "${PRINT_CONFIG:-0}" == "1" ]] && return 0
    for arg in "$@"; do
        case "$arg" in
            --print-config|--help|-h)
                return 0
                ;;
        esac
    done
    return 1
}

start_with_launcher() {
    mkdir -p "$LOG_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    cd "$ROOT"

    "$ROOT/launcher.sh" --non-interactive "$@"
    if launcher_reads_only "$@"; then
        return 0
    fi

    load_manager_state || {
        echo "ERROR: launcher did not write state file: $STATE_FILE" >&2
        exit 1
    }

    if [[ -z "${LAST_PID_FILE:-}" || ! -f "${LAST_PID_FILE:-}" ]]; then
        echo "ERROR: launcher did not record a pid file in $STATE_FILE" >&2
        exit 1
    fi
    if [[ -z "${LAST_LOG_FILE:-}" || ! -f "${LAST_LOG_FILE:-}" ]]; then
        echo "ERROR: launcher did not record a log file in $STATE_FILE" >&2
        exit 1
    fi

    SERVER_PID=$(cat "$LAST_PID_FILE")
    if ! pid_is_running "$SERVER_PID"; then
        echo "ERROR: launched server is not running: $SERVER_PID" >&2
        exit 1
    fi

    trap handle_signal TERM INT

    tail --pid="$SERVER_PID" -n +1 -F "$LAST_LOG_FILE" &
    TAIL_PID=$!

    while pid_is_running "$SERVER_PID"; do
        sleep 1
    done
    wait "$TAIL_PID" 2>/dev/null || true
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        launcher|launcher.sh)
            shift
            cd "$ROOT"
            exec "$ROOT/launcher.sh" "$@"
            ;;
        api-server)
            shift
            cd "$ROOT"
            exec "$ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "$@"
            ;;
        --*)
            start_with_launcher "$@"
            exit 0
            ;;
        *)
            exec "$@"
            ;;
    esac
fi

start_with_launcher
