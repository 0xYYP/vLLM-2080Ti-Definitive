#!/usr/bin/env bash
set -euo pipefail

pkill -9 -u dietpi -f "VLLM::|vllm|ptxas|triton|sglang.launch_server|sglang::scheduler|sglang::detokenizer" || true
systemctl restart miniclaw-minit1-dense-qwen27.service miniclaw-minit1-dense-qwen27-proxy.service \
  miniclaw-minit2-dense-gemma31.service miniclaw-minit2-dense-gemma31-proxy.service
sleep 8
curl -fsS http://127.0.0.1:18100/health
echo
curl -fsS http://127.0.0.1:18110/health
echo
systemctl is-active miniclaw-minit1-dense-qwen27.service miniclaw-minit1-dense-qwen27-proxy.service \
  miniclaw-minit2-dense-gemma31.service miniclaw-minit2-dense-gemma31-proxy.service
failed=$(systemctl --failed --no-legend || true)
if [[ -n "$failed" ]]; then
  printf '%s\n' "$failed"
  exit 1
fi
