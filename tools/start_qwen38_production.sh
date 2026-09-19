#!/usr/bin/env bash
# 生产常驻服务启动脚本：Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ（int4 / 0.2 栈）
#
# 本脚本是 cybros 上对外 vLLM 服务的权威启动入口，取代此前只存在于 tmpfs 的
# /tmp/start.PRODUCTION.sh（那份脚本不在版本库里，重启即丢，且 /tmp 另有 10 天
# 未访问清理规则）。脚本逐参数复现 2026-09-17 21:04 起跑、至今在线的那一个实例，
# 并在每次启动前把可追溯信息（git HEAD / 工作区改动数 / 模板 md5 / 完整 argv）
# 写进日志头部，便于事后比对而不是靠回忆。
#
# 路由参数：256K 上下文 / fp16 KV / MTP k=4 / 单流 max-num-seqs=1 / gpu-util 0.98
#          mamba-cache-mode align / 模板 qwen3.8-zh-compatible-v7.jinja
#
# 注意 served-model-name 里的 "mtp2" 是历史字样，实际 k=4；网关按这两个名字做
# 模型映射，**不得改动**（改了 New API 那边的映射会断）。
#
# 用法
#   bash start_qwen38_production.sh               # 后台启动，等 /v1/models 就绪后返回
#   bash start_qwen38_production.sh --print-args   # 只打印 argv（无副作用，供审计）
#   bash start_qwen38_production.sh --pids         # 只打印本服务进程 PID（供审计）
#   bash start_qwen38_production.sh --status       # 进程 / 端口 / 服务名
#   bash start_qwen38_production.sh --stop         # 优雅停止（加 --force 才升级 SIGKILL）
#   bash start_qwen38_production.sh --restart      # 停止后重启
#   bash start_qwen38_production.sh --foreground   # 前台运行（systemd 单元用的就是这个）
#
# 审计现役进程是否就是本脚本描述的配置（argv 逐行比对）：
#   pid=$(bash start_qwen38_production.sh --pids | head -1)
#   diff <(tr '\0' '\n' < /proc/$pid/cmdline | tail -n +4) <(bash start_qwen38_production.sh --print-args)
# （tail -n +4 是跳过 argv[0]=python、argv[1]=-m、argv[2]=模块名这三行）
#
# 所有可调项都是环境变量，默认值即生产值，逐条见下方分节。
#
# 与历史 /tmp/start.PRODUCTION.sh 的差异，全部有意且非行为性：
#   * PYTHONUNBUFFERED=1：日志实时落盘，只影响 Python stdout 缓冲；
#   * 日志头部增加追溯块（见上）；
#   * --port 仅在 PORT≠8000 时传给 vLLM，这样默认情况下 argv 可与现役逐行 diff。
set -euo pipefail

# ---------------------------------------------------------------- 路径（默认=cybros 现役布局）
# /data/models/.ai-stack-0.2 是指向 /opt/vllm-2080ti-definitive 的软链接，两种写法等价。
STACK_ROOT=${STACK_ROOT:-/data/models/.ai-stack-0.2}
VLLM_ROOT=${VLLM_ROOT:-$STACK_ROOT/vllm-2080ti-definitive-0.2}
PYTHON=${PYTHON:-$VLLM_ROOT/.venv/bin/python}
MODEL_DIR=${MODEL_DIR:-/data/models/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ}
LOG_DIR=${LOG_DIR:-$STACK_ROOT/run-logs}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$VLLM_ROOT/profiles/templates/qwen3.8-zh-compatible-v7.jinja}

# ---------------------------------------------------------------- 服务身份
SERVICE_TAG=${SERVICE_TAG:-qwen38-27b-uncensored-256K-mtp2-text-image-cu128}
SERVED_NAME_FULL=${SERVED_NAME_FULL:-"$SERVICE_TAG qwen3.8-27b-uncensored"}
PORT=${PORT:-8000}
# 留空=不导出 CUDA_VISIBLE_DEVICES（TP=2 用满两卡）。只跑单卡时 TP_SIZE 也要跟着改。
GPU_DEVICES=${GPU_DEVICES:-}

# ---------------------------------------------------------------- 路由参数
TP_SIZE=${TP_SIZE:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MTP_K=${MTP_K:-4}
GPU_UTIL=${GPU_UTIL:-0.98}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-float16}
QUANTIZATION=${QUANTIZATION:-compressed-tensors}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}
GDN_PREFILL_BACKEND=${GDN_PREFILL_BACKEND:-flashqla_legacy}

# ---------------------------------------------------------------- 运行时
WAIT_TIMEOUT=${WAIT_TIMEOUT:-600}   # 启动就绪等待上限（秒）；冷启动实测约 200s
STOP_TIMEOUT=${STOP_TIMEOUT:-60}    # 优雅停止等待上限（秒）

MODE=start
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    start | --start) MODE=start ;;
    -f | --foreground) MODE=foreground ;;
    --print-args) MODE=print-args ;;
    --pids) MODE=pids ;;
    --status) MODE=status ;;
    --stop) MODE=stop ;;
    --restart) MODE=restart ;;
    --force) FORCE=1 ;;
    -h | --help) MODE=help ;;
    *)
      echo "未知参数：$1（用 --help 看用法）" >&2
      exit 2
      ;;
  esac
  shift
done

die() {
  echo "错误：$*" >&2
  exit 1
}

file_md5() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    echo n/a
  fi
}

log_basename() {
  # 沿用历史命名 qwen38-02-fp16kv-mtp<K>-g<util>，便于和 run-logs 里旧日志连成一条线
  local util=${GPU_UTIL//./}
  printf 'qwen38-02-fp16kv-mtp%s-g%s-%s.log\n' "$MTP_K" "$util" "$(date +%Y%m%d-%H%M%S)"
}

build_args() {
  local -a names
  read -r -a names <<<"$SERVED_NAME_FULL"
  ARGS=(
    --model "$MODEL_DIR"
    --served-model-name "${names[@]}"
    --dtype half
    --tensor-parallel-size "$TP_SIZE"
    --generation-config vllm
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --quantization "$QUANTIZATION"
    --gpu-memory-utilization "$GPU_UTIL"
    --mamba-cache-mode align
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --reasoning-parser "$REASONING_PARSER"
    --tool-call-parser "$TOOL_CALL_PARSER"
    --enable-auto-tool-choice
    --additional-config "{\"gdn_prefill_backend\":\"$GDN_PREFILL_BACKEND\"}"
    --chat-template "$CHAT_TEMPLATE"
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}"
    --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[$((MTP_K + 1))],\"max_cudagraph_capture_size\":$((MTP_K + 1))}"
  )
  # 端口等于 vLLM 默认值时不显式传，保持 argv 与现役进程可逐行 diff
  if [[ "$PORT" != "8000" ]]; then
    ARGS=(--port "$PORT" "${ARGS[@]}")
  fi
  return 0
}

apply_env() {
  # 0.2 栈的四个必需导出；FLASHQLA_ROOT 是 flashqla_legacy 的运行时硬依赖，缺了直接 ImportError
  export VLLM_CACHE_ROOT=$STACK_ROOT/cache/vllm
  export TORCHINDUCTOR_CACHE_DIR=$STACK_ROOT/cache/torch-inductor
  export FLASHQLA_ROOT=$VLLM_ROOT/.deps/FlashQLA-SM70-SM75
  export PYTHONPATH=$VLLM_ROOT:$FLASHQLA_ROOT${PYTHONPATH:+:$PYTHONPATH}
  export PYTHONUNBUFFERED=1
  if [[ -n "$GPU_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICES
  fi
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$LOG_DIR"
}

provenance() {
  # git 缺失或 VLLM_ROOT 不是 git 仓库时（例如只部署了 venv 的机器）不能因此启动失败
  local head dirty
  head=$(git -C "$VLLM_ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)
  dirty=$(git -C "$VLLM_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ' || echo n/a)
  echo "=== qwen38-production 启动于 $(date '+%Y-%m-%d %H:%M:%S %z')，主机 $(hostname)，用户 $(id -un) ==="
  echo "git: HEAD=$head 工作区改动文件数=$dirty"
  echo "model: $MODEL_DIR"
  echo "chat-template: $CHAT_TEMPLATE (md5=$(file_md5 "$CHAT_TEMPLATE"))"
  echo "argv:"
  printf '  %s\n' "${ARGS[@]}"
  echo "==========================================================="
}

require_files() {
  [[ -x "$PYTHON" ]] || die "$PYTHON 不存在或不可执行（0.2 栈 venv 是否还在？）"
  [[ -d "$MODEL_DIR" ]] || die "模型目录不存在：$MODEL_DIR"
  [[ -f "$CHAT_TEMPLATE" ]] || die "chat template 不存在：$CHAT_TEMPLATE"
  [[ "$MTP_K" =~ ^[0-9]+$ ]] || die "MTP_K 必须是整数：$MTP_K"
  [[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT 必须是整数：$PORT"
}

collect_pids() {
  # 只按 served-name 匹配会误伤客户端：本机上 fastllm 实验室的 `ftllm webui` 就把
  # 这个名字写在 --api_model 里（PID 固定存活数天）。所以必须同时要求该进程就是
  # vLLM 服务本体（argv 含 `-m vllm.entrypoints.openai.api_server`）。
  local p argv
  for p in $(pgrep -f 'vllm\.entrypoints\.openai\.api_server' 2>/dev/null || true); do
    [[ "$p" == "$$" ]] && continue
    argv=$(tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null) || continue
    [[ "$argv" == *"-m vllm.entrypoints.openai.api_server"* ]] || continue
    [[ "$argv" == *"$SERVICE_TAG"* ]] || continue
    echo "$p"
  done
}

port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -q ":$PORT "
  else
    curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/v1/models"
  fi
}

wait_ready() {
  local i
  for ((i = 1; i <= WAIT_TIMEOUT; i++)); do
    if curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
      echo "服务就绪（约 ${i}s）：http://127.0.0.1:$PORT/v1/models"
      echo "注意：MTP 冷启动预热完成前测出的吞吐不是真数。"
      return 0
    fi
    sleep 1
  done
  echo "⚠️ 等待 ${WAIT_TIMEOUT}s 仍未就绪，查日志尾部：" >&2
  tail -n 20 "$(ls -t "$LOG_DIR"/qwen38-02-*.log 2>/dev/null | head -1)" >&2 || true
  return 1
}

do_stop() {
  local pids
  pids=$(collect_pids)
  if [[ -z "$pids" ]]; then
    echo "没有匹配到运行中的实例（匹配串：$SERVICE_TAG）"
    return 0
  fi
  echo "停止：$(echo "$pids" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  local i
  for ((i = 1; i <= STOP_TIMEOUT; i++)); do
    if [[ -z "$(collect_pids)" ]]; then
      echo "已退出（约 ${i}s）"
      return 0
    fi
    sleep 1
  done
  if [[ "$FORCE" == "1" ]]; then
    echo "超过 ${STOP_TIMEOUT}s 未退出，SIGKILL"
    # shellcheck disable=SC2046
    kill -9 $(collect_pids) 2>/dev/null || true
    sleep 2
    return 0
  fi
  echo "⚠️ 超过 ${STOP_TIMEOUT}s 仍未退出；确认后加 --force 升级 SIGKILL" >&2
  return 1
}

do_status() {
  local pids
  pids=$(collect_pids)
  if [[ -n "$pids" ]]; then
    echo "进程：$(echo "$pids" | tr '\n' ' ')"
    local -a pid_list
    read -r -a pid_list <<<"$pids"
    ps -o pid,etime,rss --no-headers -p "${pid_list[@]}" 2>/dev/null || true
  else
    echo "进程：未运行"
  fi
  echo "端口 $PORT：$(port_in_use && echo 在监听 || echo 未监听)"
  echo "服务名：$SERVED_NAME_FULL"
  local latest
  latest=$(ls -t "$LOG_DIR"/qwen38-02-*.log 2>/dev/null | head -1 || true)
  echo "日志：${latest:-$LOG_DIR 下没有 qwen38-02-*.log}"
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" |
    grep -o '"id":"[^"]*"' | grep -v modelperm | sed 's/^/  /' 2>/dev/null || true
}

do_start() {
  require_files
  if [[ -n "$(collect_pids)" ]]; then
    die "已有实例在跑（pgrep -f $SERVICE_TAG）。要先停请用 --restart，或 --stop。"
  fi
  if port_in_use; then
    die "端口 $PORT 已被占用，但不是本服务。确认占用者后再启动。"
  fi

  apply_env
  build_args
  local log="$LOG_DIR/$(log_basename)"

  if [[ "$MODE" == "foreground" ]]; then
    # systemd 场景：前台运行，日志同时进 journald 和 run-logs
    exec > >(tee -a "$log") 2>&1
    provenance
    cd "$VLLM_ROOT"
    exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
  fi

  provenance >"$log"
  cd "$VLLM_ROOT"
  nohup "$PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >>"$log" 2>&1 &
  local pid=$!
  echo "已启动 pid=$pid"
  echo "日志：$log"
  wait_ready
}

case "$MODE" in
  help)
    # 打印文件头部的注释块（第 1 行是 shebang，到第一个非注释行为止）
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    ;;
  print-args)
    build_args
    printf '%s\n' "${ARGS[@]}"
    ;;
  pids) collect_pids ;;
  status) do_status ;;
  stop) do_stop ;;
  restart)
    do_stop
    do_start
    ;;
  start | foreground) do_start ;;
esac
