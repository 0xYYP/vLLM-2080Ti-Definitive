#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper kept for old local command lines. New deployments should
# call remote_start_vllm_cu128.sh directly.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/remote_start_vllm_cu128.sh" "$@"
