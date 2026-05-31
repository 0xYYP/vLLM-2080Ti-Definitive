#!/usr/bin/env bash
set -euo pipefail

: "${RUN_USER:=vllm}"
: "${SERVICE_RESTART_LIST:=}"
: "${HEALTH_URLS:=}"
: "${KILL_PROCESS_PATTERN:=VLLM::|vllm|ptxas|triton|sglang.launch_server|sglang::scheduler|sglang::detokenizer}"

pkill -9 -u "$RUN_USER" -f "$KILL_PROCESS_PATTERN" || true

if [[ -n "$SERVICE_RESTART_LIST" ]]; then
  # shellcheck disable=SC2086
  systemctl restart $SERVICE_RESTART_LIST
  sleep 8
fi

if [[ -n "$HEALTH_URLS" ]]; then
  for url in $HEALTH_URLS; do
    curl -fsS "$url"
    echo
  done
fi

if [[ -n "$SERVICE_RESTART_LIST" ]]; then
  # shellcheck disable=SC2086
  systemctl is-active $SERVICE_RESTART_LIST
fi

failed=$(systemctl --failed --no-legend || true)
if [[ -n "$failed" ]]; then
  printf '%s\n' "$failed"
  exit 1
fi
