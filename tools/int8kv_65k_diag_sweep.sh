#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_DIR=${MODEL_DIR:-}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
PORT=${PORT:-8000}
START_TIMEOUT=${START_TIMEOUT:-900}
RUNS=${RUNS:-3}
ALLOW_STOP_EXISTING=${ALLOW_STOP_EXISTING:-0}
KEEP_LAST_SERVICE=${KEEP_LAST_SERVICE:-0}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
OUT_DIR=${OUT_DIR:-/tmp/int8kv-65k-diag-$STAMP}
OUT=${OUT:-$OUT_DIR/profile.jsonl}

PROFILES=(
  qwen27b/experimental/fp8/int8kv-65K-mtp3-text-only.env
  qwen27b/experimental/fp8/int8kv-65K-fastgraph-mtp3-text-only.env
  qwen27b/experimental/fp8/int8kv-65K-fast2560-mtp3-text-only.env
  qwen27b/experimental/fp8/int8kv-65K-fastaligned-mtp3-text-only.env
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

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

pid_is_running() {
  local pid=${1:-}
  [[ -n "$pid" && -d "/proc/$pid" ]]
}

stop_pid_tree() {
  local pid=$1
  local mode=${2:-term}
  local child
  pid_is_running "$pid" || return 0
  while IFS= read -r child; do
    [[ -n "$child" && "$child" != "$pid" ]] || continue
    stop_pid_tree "$child" "$mode" || true
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  if [[ "$mode" == force ]]; then
    kill -KILL "$pid" 2>/dev/null || true
  else
    kill "$pid" 2>/dev/null || true
  fi
}

managed_pid_files() {
  find "$ROOT/run-logs" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort
}

managed_service_count() {
  local pid_file pid count=0
  while IFS= read -r pid_file; do
    pid=$(cat "$pid_file" 2>/dev/null || true)
    pid_is_running "$pid" && count=$((count + 1))
  done < <(managed_pid_files)
  echo "$count"
}

stop_managed_services() {
  local pid_file pid
  while IFS= read -r pid_file; do
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if pid_is_running "$pid"; then
      echo "Stopping managed service pid=$pid file=$pid_file"
      stop_pid_tree "$pid" term || true
      sleep 2
      if pid_is_running "$pid"; then
        stop_pid_tree "$pid" force || true
      fi
    fi
    rm -f "$pid_file"
  done < <(managed_pid_files)
}

profile_case_name() {
  local profile=$1
  basename "$profile" .env | sed -E 's/-mtp3-text-only$//'
}

run_profile_request() {
  local served=$1
  local label=$2
  local prompt_tokens=$3
  local gen_tokens=$4
  "$ROOT/.venv/bin/python" "$ROOT/tools/profile_request.py" \
    --model-dir "$MODEL_DIR" \
    --served-name "$served" \
    --base-url "$BASE_URL" \
    --endpoint completions \
    --prompt-tokens "$prompt_tokens" \
    --gen-tokens "$gen_tokens" \
    --label "$label" \
    --out "$OUT" \
    --ignore-eos \
    --pure-filler
}

summarize_jsonl() {
  "$ROOT/.venv/bin/python" - "$OUT" <<'PY'
import json
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
rows = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            rows.append(json.loads(line))

groups = defaultdict(list)
for row in rows:
    label = row.get("label", "")
    if "warmup" in label:
        continue
    parts = label.rsplit("_", 1)
    case_stage = parts[0] if len(parts) == 2 and parts[1].startswith("run") else label
    groups[case_stage].append(row)

print("case_stage\tcount\tprefill_median\tdecode_median\tdecode_values")
for key in sorted(groups):
    vals = groups[key]
    decodes = [float(v["decode_tok_s"]) for v in vals if v.get("decode_tok_s") is not None]
    prefills = [float(v["prefill_tok_s"]) for v in vals if v.get("prefill_tok_s") is not None]
    if not decodes:
        continue
    print(
        f"{key}\t{len(vals)}\t"
        f"{statistics.median(prefills):.2f}\t"
        f"{statistics.median(decodes):.2f}\t"
        f"{','.join(f'{x:.2f}' for x in decodes)}"
    )
PY
}

[[ -n "$MODEL_DIR" ]] || die "MODEL_DIR is required"
[[ -d "$MODEL_DIR" ]] || die "MODEL_DIR does not exist: $MODEL_DIR"
[[ -x "$ROOT/.venv/bin/python" ]] || die "missing runtime python: $ROOT/.venv/bin/python"

mkdir -p "$OUT_DIR"
echo "Output JSONL: $OUT"

if (( $(managed_service_count) > 0 )) && [[ "$ALLOW_STOP_EXISTING" != "1" ]]; then
  die "managed vLLM service is already running; set ALLOW_STOP_EXISTING=1 to let this sweep restart it"
fi

for profile in "${PROFILES[@]}"; do
  profile_file="$ROOT/profiles/$profile"
  [[ -f "$profile_file" ]] || die "profile not found: $profile"
  served=$(read_profile_value "$profile_file" SERVED_NAME)
  mode=$(read_profile_value "$profile_file" COMPATIBLE_MODES)
  mode=${mode%%,*}
  case_name=$(profile_case_name "$profile")

  stop_managed_services
  echo
  echo "=== Launching $case_name mode=$mode served=$served ==="
  MODEL_DIR="$MODEL_DIR" \
    PROFILE="$profile" \
    MODE="$mode" \
    PORT="$PORT" \
    NON_INTERACTIVE=1 \
    SKIP_STARTUP_SMOKE=1 \
    START_TIMEOUT="$START_TIMEOUT" \
    "$ROOT/launcher.sh" --non-interactive

  echo "=== Warmup $case_name ==="
  run_profile_request "$served" "${case_name}_4k_warmup" 4096 128
  run_profile_request "$served" "${case_name}_65k_warmup" 65536 512

  for i in $(seq 1 "$RUNS"); do
    echo "=== Measured $case_name run $i/$RUNS ==="
    run_profile_request "$served" "${case_name}_4k_run$i" 4096 128
    run_profile_request "$served" "${case_name}_65k_run$i" 65536 512
  done
done

if [[ "$KEEP_LAST_SERVICE" != "1" ]]; then
  stop_managed_services
fi

echo
echo "Summary:"
summarize_jsonl | tee "$OUT_DIR/summary.tsv"
echo
echo "JSONL: $OUT"
echo "Summary: $OUT_DIR/summary.tsv"
