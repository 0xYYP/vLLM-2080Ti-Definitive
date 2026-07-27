#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
REMOTE_HOST=${REMOTE_HOST:-root@192.168.1.40}
REMOTE_ROOT=${REMOTE_ROOT:-/data/stable/vllm-sm75-tp2-cu128}
REMOTE_USER=${REMOTE_USER:-dietpi}
GPU_DEVICES=${GPU_DEVICES:-1,2}
PORT=${PORT:-19447}
PROMPT_TOKENS=${PROMPT_TOKENS:-4096}
GEN_TOKENS=${GEN_TOKENS:-128}
WARMUPS=${WARMUPS:-1}
MEASURED_RUNS=${MEASURED_RUNS:-3}
FAST_MARGIN=${FAST_MARGIN:-1.0}
FAST_ONLY=${FAST_ONLY:-1}
BASELINE_RESULT_DIR=${BASELINE_RESULT_DIR:-}
CASE_FILTER=${CASE_FILTER:-}
SPEC_SYNC_OVERRIDE=${SPEC_SYNC_OVERRIDE:-}
MTP_OVERRIDE=${MTP_OVERRIDE:-}
FAST_COMPARE_REQUIRED=${FAST_COMPARE_REQUIRED:-1}
BENCHMARK_TIMEOUT_SEC=${BENCHMARK_TIMEOUT_SEC:-1800}
EVAL_LAUNCHER=${EVAL_LAUNCHER:-./launcher.sh}
RESULT_STAMP=${RESULT_STAMP:-$(date +%Y%m%d-%H%M%S)}
REMOTE_RESULT_DIR=${REMOTE_RESULT_DIR:-$REMOTE_ROOT/results/fast_mode_evaluator_$RESULT_STAMP}
SOURCE_CLOSURE_MANIFEST=$(mktemp "${TMPDIR:-/tmp}/fast-mode-source-closure.XXXXXX")

FP8_MODEL_DIR=${FP8_MODEL_DIR:-/data/models/vllm/qwen-family-27b-fp8}
INT4_MODEL_DIR=${INT4_MODEL_DIR:-/data/models/vllm/qwen-family-27b-gptq-int4}
FP8_TOKENIZER_DIR=${FP8_TOKENIZER_DIR:-$FP8_MODEL_DIR}
INT4_TOKENIZER_DIR=${INT4_TOKENIZER_DIR:-$INT4_MODEL_DIR}
EVAL_GIT_HEAD=${EVAL_GIT_HEAD:-$(git -C "$ROOT" rev-parse HEAD)}

if [[ "$(id -u)" == "0" ]]; then
  SSH=(runuser -u max -- ssh)
  RSYNC=(runuser -u max -- rsync)
else
  SSH=(ssh)
  RSYNC=(rsync)
fi

run_ssh() {
  "${SSH[@]}" -o ConnectTimeout=10 "$REMOTE_HOST" "$@"
}

cleanup_local() {
  rm -f "$SOURCE_CLOSURE_MANIFEST"
}

trap cleanup_local EXIT

SOURCE_CLOSURE_FILES=(
  launcher.sh
  build.sh
  vllm/config/compilation.py
  vllm/envs.py
  vllm/sm75_attention_trace.py
  vllm/v1/attention/ops/triton_unified_attention.py
  vllm/v1/attention/ops/triton_turboquant_decode.py
  vllm/v1/attention/backends/triton_attn.py
  vllm/v1/attention/backends/turboquant_attn.py
  vllm/v1/attention/sm75_attention_planner.py
  vllm/v1/attention/sm75_attention_planner_types.py
  vllm/v1/worker/gpu_model_runner.py
  tools/evaluate_fast_modes.sh
  tools/profile_request.py
  tools/validate_benchmark_manifest.py
)

ROOT="$ROOT" SOURCE_CLOSURE_MANIFEST="$SOURCE_CLOSURE_MANIFEST" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
files = [
    "launcher.sh",
    "build.sh",
    "vllm/config/compilation.py",
    "vllm/envs.py",
    "vllm/sm75_attention_trace.py",
    "vllm/v1/attention/ops/triton_unified_attention.py",
    "vllm/v1/attention/ops/triton_turboquant_decode.py",
    "vllm/v1/attention/backends/triton_attn.py",
    "vllm/v1/attention/backends/turboquant_attn.py",
    "vllm/v1/attention/sm75_attention_planner.py",
    "vllm/v1/attention/sm75_attention_planner_types.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "tools/evaluate_fast_modes.sh",
    "tools/profile_request.py",
    "tools/validate_benchmark_manifest.py",
]
payload = {"schema_version": 1, "files": []}
for relative in files:
    path = root / relative
    payload["files"].append(
        {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
Path(os.environ["SOURCE_CLOSURE_MANIFEST"]).write_text(
    json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
)
PY

sync_runtime() {
  if [[ "${EVAL_SYNC:-1}" != "1" ]]; then
    printf 'EVAL_SYNC=0 is rejected: source closure cannot be verified.\n' >&2
    return 1
  fi

  "${RSYNC[@]}" -a --delete "$ROOT/profiles/" "$REMOTE_HOST:$REMOTE_ROOT/profiles/"
  "${RSYNC[@]}" -a --delete "$ROOT/tools/" "$REMOTE_HOST:$REMOTE_ROOT/tools/"
  (
    cd "$ROOT"
    "${RSYNC[@]}" -aR "${SOURCE_CLOSURE_FILES[@]}" "$REMOTE_HOST:$REMOTE_ROOT/"
  )
  "${RSYNC[@]}" -a "$SOURCE_CLOSURE_MANIFEST" "$REMOTE_HOST:$REMOTE_ROOT/source-closure-manifest.json"

  run_ssh "chown -R $REMOTE_USER:$REMOTE_USER '$REMOTE_ROOT/profiles' '$REMOTE_ROOT/tools' '$REMOTE_ROOT/launcher.sh' '$REMOTE_ROOT/build.sh' '$REMOTE_ROOT/vllm' '$REMOTE_ROOT/source-closure-manifest.json'"

  run_ssh "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_USER='$REMOTE_USER' bash -s" <<'REMOTE_SYNC'
set -euo pipefail
site_packages=$(runuser -u "$REMOTE_USER" -- "$REMOTE_ROOT/.venv/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)
site_vllm="$site_packages/vllm"
if [[ -d "$site_vllm" ]]; then
  for relative in \
    vllm/config/compilation.py \
    vllm/envs.py \
    vllm/sm75_attention_trace.py \
    vllm/v1/attention/ops/triton_unified_attention.py \
    vllm/v1/attention/ops/triton_turboquant_decode.py \
    vllm/v1/attention/backends/triton_attn.py \
    vllm/v1/attention/backends/turboquant_attn.py \
    vllm/v1/attention/sm75_attention_planner.py \
    vllm/v1/attention/sm75_attention_planner_types.py \
    vllm/v1/worker/gpu_model_runner.py; do
    install -D -o "$REMOTE_USER" -g "$REMOTE_USER" -m 0644 \
      "$REMOTE_ROOT/$relative" "$site_vllm/${relative#vllm/}"
  done
fi
REMOTE_SYNC

  run_ssh "runuser -u '$REMOTE_USER' -- '$REMOTE_ROOT/.venv/bin/python' '$REMOTE_ROOT/tools/validate_benchmark_manifest.py' --source-closure '$REMOTE_ROOT/source-closure-manifest.json' --source-root '$REMOTE_ROOT'"
}

sync_runtime

run_ssh \
  "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_USER='$REMOTE_USER' GPU_DEVICES='$GPU_DEVICES' PORT='$PORT' PROMPT_TOKENS='$PROMPT_TOKENS' GEN_TOKENS='$GEN_TOKENS' WARMUPS='$WARMUPS' MEASURED_RUNS='$MEASURED_RUNS' FAST_MARGIN='$FAST_MARGIN' FAST_ONLY='$FAST_ONLY' BASELINE_RESULT_DIR='$BASELINE_RESULT_DIR' CASE_FILTER='$CASE_FILTER' SPEC_SYNC_OVERRIDE='$SPEC_SYNC_OVERRIDE' MTP_OVERRIDE='$MTP_OVERRIDE' FAST_COMPARE_REQUIRED='$FAST_COMPARE_REQUIRED' BENCHMARK_TIMEOUT_SEC='$BENCHMARK_TIMEOUT_SEC' EVAL_LAUNCHER='$EVAL_LAUNCHER' REMOTE_RESULT_DIR='$REMOTE_RESULT_DIR' FP8_MODEL_DIR='$FP8_MODEL_DIR' INT4_MODEL_DIR='$INT4_MODEL_DIR' FP8_TOKENIZER_DIR='$FP8_TOKENIZER_DIR' INT4_TOKENIZER_DIR='$INT4_TOKENIZER_DIR' EVAL_GIT_HEAD='$EVAL_GIT_HEAD' bash -s" <<'REMOTE'
set -euo pipefail

cd "$REMOTE_ROOT"
mkdir -p "$REMOTE_RESULT_DIR"
chown -R "$REMOTE_USER:$REMOTE_USER" "$REMOTE_RESULT_DIR"

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

sanitize() {
  printf '%s' "$1" | tr '/ ' '__' | tr -c 'A-Za-z0-9_.-' '_'
}

stop_runtime_vllm() {
  local pid_file pid cmd
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
      if [[ "$cmd" == *"$REMOTE_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server"* ]]; then
        kill "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pid_file"
  done < <(find "$REMOTE_ROOT/run-logs" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)

  sleep 2
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done < <(pgrep -f "$REMOTE_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server" || true)
}

trap stop_runtime_vllm EXIT

make_eval_profile() {
  local rel=$1
  local mode=$2
  local out=$3
  cp "$REMOTE_ROOT/profiles/$rel" "$out"
  if grep -q '^COMPATIBLE_MODES=' "$out"; then
    sed -i 's/^COMPATIBLE_MODES=.*/COMPATIBLE_MODES=safe,normal,fast/' "$out"
  else
    printf '\nCOMPATIBLE_MODES=safe,normal,fast\n' >>"$out"
  fi
  if [[ -n "$MTP_OVERRIDE" ]]; then
    if grep -q '^MTP_K=' "$out"; then
      sed -i "s/^MTP_K=.*/MTP_K=$MTP_OVERRIDE/" "$out"
    else
      printf 'MTP_K=%s\n' "$MTP_OVERRIDE" >>"$out"
    fi
  fi
  if [[ "$mode" == "safe" ]]; then
    case "$(read_profile_value "$out" KV_CACHE_DTYPE)" in
      ""|fp16|default|auto) ;;
      *) sed -i 's/^MTP_K=.*/MTP_K=0/' "$out" ;;
    esac
  fi
}

write_rejected_manifest() {
  local path=$1
  local reason=$2
  runuser -u "$REMOTE_USER" -- "$REMOTE_ROOT/.venv/bin/python" - "$path" "$reason" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "schema_version": 2,
            "status": "rejected",
            "decision": "excluded",
            "artifacts": [],
            "reason": sys.argv[2],
        },
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
}

run_one_case() {
  local group=$1
  local profile=$2
  local model_dir=$3
  local tokenizer_dir=$4
  local mode=$5
  local role=$6
  local threshold=$7

  local case_id case_dir tmp_profile served_name semantic_jsonl bench_jsonl csv_name launch_out launch_err label i closure_name
  case_id=$(sanitize "${group}_${mode}")
  case_dir="$REMOTE_RESULT_DIR/$case_id"
  if [[ -e "$case_dir" ]]; then
    write_rejected_manifest "$REMOTE_RESULT_DIR/${case_id}.rejected.json" stale_case_directory
    printf 'stale case directory: %s\n' "$case_dir" >&2
    return 1
  fi
  mkdir -p "$case_dir"
  chown -R "$REMOTE_USER:$REMOTE_USER" "$case_dir"
  tmp_profile="$case_dir/profile.env"
  if [[ ! -f "$REMOTE_ROOT/profiles/$profile" ]]; then
    write_rejected_manifest "$case_dir/artifact-manifest.json" missing_profile
    printf 'rejected missing profile: %s\n' "$profile" >&2
    return 1
  fi
  make_eval_profile "$profile" "$mode" "$tmp_profile"
  chown "$REMOTE_USER:$REMOTE_USER" "$tmp_profile"
  served_name=$(read_profile_value "$tmp_profile" SERVED_NAME)
  if [[ -z "$served_name" ]]; then
    write_rejected_manifest "$case_dir/artifact-manifest.json" missing_served_name
    printf 'rejected missing SERVED_NAME: %s\n' "$profile" >&2
    return 1
  fi
  semantic_jsonl="$case_dir/pp${PROMPT_TOKENS}_tg${GEN_TOKENS}.jsonl"
  bench_jsonl="$case_dir/bench-results.jsonl"
  csv_name=results.csv
  closure_name=source-closure-manifest.json
  cp "$REMOTE_ROOT/source-closure-manifest.json" "$case_dir/$closure_name"
  launch_out="$case_dir/launch.out"
  launch_err="$case_dir/launch.err"
  : >"$launch_out"
  : >"$launch_err"
  : >"$case_dir/bench.out"
  : >"$case_dir/bench.err"
  chown "$REMOTE_USER:$REMOTE_USER" \
    "$launch_out" "$launch_err" "$case_dir/bench.out" "$case_dir/bench.err"

  write_artifact_manifest() {
    local artifact_status=$1
    local reason=${2:-}
    runuser -u "$REMOTE_USER" -- "$REMOTE_ROOT/.venv/bin/python" - \
      "$case_dir" "$artifact_status" "$(basename "$semantic_jsonl")" "$(basename "$bench_jsonl")" "$csv_name" "$reason" \
      "$case_id" "$group" "$profile" "$mode" "$role" "$threshold" "$model_dir" "$tokenizer_dir" "$served_name" \
      "$PROMPT_TOKENS" "$GEN_TOKENS" "$WARMUPS" "$MEASURED_RUNS" "$EVAL_GIT_HEAD" "$closure_name" <<'PY'
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
status = sys.argv[2]
semantic_name = sys.argv[3]
bench_name = sys.argv[4]
csv_name = sys.argv[5]
reason = sys.argv[6]
case_id = sys.argv[7]
group = sys.argv[8]
profile_path = sys.argv[9]
mode = sys.argv[10]
role = sys.argv[11]
threshold = float(sys.argv[12])
checkpoint = sys.argv[13]
tokenizer = sys.argv[14]
served_alias = sys.argv[15]
prompt_tokens = int(sys.argv[16])
generation_tokens = int(sys.argv[17])
warmups = int(sys.argv[18])
measured_runs = int(sys.argv[19])
git_head = sys.argv[20]
closure_name = sys.argv[21]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_values(path: Path) -> dict[str, str]:
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


profile = profile_values(case_dir / "profile.env")
closure = case_dir / closure_name
try:
    vllm_version = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    vllm_version = "source-checkout"
payload = {
    "schema_version": 2,
    "status": status,
    "decision": "accepted" if status == "captured" else "excluded",
    "reason": None if status == "captured" else (reason or "not_started"),
    "case": {
        "id": case_id,
        "group": group,
        "mode": mode,
        "role": role,
        "threshold": threshold,
    },
    "profile": {
        "path": profile_path,
        "model_family": profile.get("MODEL_FAMILY", ""),
        "quantization": profile.get("QUANTIZATION", ""),
        "kv_cache_dtype": profile.get("KV_CACHE_DTYPE") or "fp16",
        "mtp_k": int(profile.get("MTP_K", "0")),
        "compatible_modes": sorted(
            item.strip()
            for item in profile.get("COMPATIBLE_MODES", "").split(",")
            if item.strip()
        ),
    },
    "workload": {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "warmups": warmups,
        "measured_runs": measured_runs,
    },
    "model": {
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "served_alias": served_alias,
    },
    "provenance": {
        "git_head": git_head,
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "vllm_version": vllm_version,
        },
        "build": {
            "source_closure_sha256": sha256(closure),
            "source_closure_schema_version": 1,
        },
    },
    "artifacts": [],
}
if status == "captured":
    semantic = case_dir / semantic_name
    bench = case_dir / bench_name
    rows = []
    for line in semantic.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
    rows = [
        row
        for row in rows
        if isinstance(row.get("label"), str)
        and re.fullmatch(r".+-run[1-9][0-9]*", row["label"])
    ]
    if not rows:
        raise SystemExit("captured benchmark has no measured semantic records")
    values = {
        "prefill_tok_s": [float(row["prefill_tok_s"]) for row in rows],
        "decode_tok_s": [float(row["decode_tok_s"]) for row in rows],
        "chunks": [float(row["chunks"]) for row in rows],
    }
    filler_valid = all(
        " the the the the" in str(row.get("content_sample", "")).lower()
        and "climate change" not in str(row.get("content_sample", "")).lower()
        and "introduction" not in str(row.get("content_sample", "")).lower()
        for row in rows
    )
    csv_path = case_dir / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prefill_tok_s", "decode_tok_s", "chunks", "filler_valid"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "prefill_tok_s": row["prefill_tok_s"],
                    "decode_tok_s": row["decode_tok_s"],
                    "chunks": row["chunks"],
                    "filler_valid": filler_valid,
                }
            )
    payload["artifacts"] = [
        {"role": "semantic_jsonl", "path": semantic_name, "sha256": sha256(semantic)},
        {"role": "bench_jsonl", "path": bench_name, "sha256": sha256(bench)},
        {"role": "summary_csv", "path": csv_name, "sha256": sha256(csv_path)},
        {"role": "profile_snapshot", "path": "profile.env", "sha256": sha256(case_dir / "profile.env")},
        {"role": "source_closure", "path": closure_name, "sha256": sha256(closure)},
        {"role": "metadata_tsv", "path": "meta.tsv", "sha256": sha256(case_dir / "meta.tsv")},
    ]
    payload["metrics"] = {
        "prefill_median": statistics.median(values["prefill_tok_s"]),
        "decode_median": statistics.median(values["decode_tok_s"]),
        "chunks_median": statistics.median(values["chunks"]),
        "filler_valid": filler_valid,
    }
(case_dir / "artifact-manifest.json").write_text(
    json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  }

  write_artifact_manifest not-run not_started

  stop_runtime_vllm

  {
    printf 'group\t%s\nprofile\t%s\nmode\t%s\nrole\t%s\nthreshold\t%s\nmodel_dir\t%s\ntokenizer_dir\t%s\nserved_name\t%s\nprompt_tokens\t%s\ngeneration_tokens\t%s\nwarmups\t%s\nmeasured_runs\t%s\ngit_head\t%s\n' \
      "$group" "$profile" "$mode" "$role" "$threshold" "$model_dir" "$tokenizer_dir" "$served_name" "$PROMPT_TOKENS" "$GEN_TOKENS" "$WARMUPS" "$MEASURED_RUNS" "$EVAL_GIT_HEAD"
  } >"$case_dir/meta.tsv"
  chown "$REMOTE_USER:$REMOTE_USER" "$case_dir/meta.tsv"

  if ! runuser -u "$REMOTE_USER" -- bash -lc \
    "cd '$REMOTE_ROOT' && MODEL_DIR='$model_dir' PROFILE_FILE='$tmp_profile' MODE='$mode' VLLM_SM75_SPEC_SYNC_MODE_OVERRIDE='$SPEC_SYNC_OVERRIDE' GPU_DEVICES='$GPU_DEVICES' PORT='$PORT' SERVICE_SCOPE=local NON_INTERACTIVE=1 START_TIMEOUT=900 '$EVAL_LAUNCHER' --non-interactive" \
    >>"$launch_out" 2>>"$launch_err"; then
    printf 'launch_failed\n' >"$case_dir/status"
    write_artifact_manifest failed launch_failed
    chown "$REMOTE_USER:$REMOTE_USER" "$case_dir/status"
    return 0
  fi

  for ((i = 0; i < WARMUPS + MEASURED_RUNS; i += 1)); do
    if (( i < WARMUPS )); then
      label="${case_id}-warmup$((i + 1))"
    else
      label="${case_id}-run$((i - WARMUPS + 1))"
    fi
    if ! runuser -u "$REMOTE_USER" -- timeout --signal=TERM --kill-after=10 "$BENCHMARK_TIMEOUT_SEC" \
      "$REMOTE_ROOT/.venv/bin/python" "$REMOTE_ROOT/tools/profile_request.py" \
      --model-dir "$model_dir" --served-name "$served_name" \
      --base-url "http://127.0.0.1:$PORT/v1" --endpoint completions \
      --prompt-tokens "$PROMPT_TOKENS" --gen-tokens "$GEN_TOKENS" \
      --label "$label" --out "$semantic_jsonl" --ignore-eos --pure-filler \
      >>"$case_dir/bench.out" 2>>"$case_dir/bench.err"; then
      printf 'benchmark_failed\n' >"$case_dir/status"
      write_artifact_manifest failed benchmark_timeout_or_failure
      stop_runtime_vllm
      return 0
    fi
  done

  if ! runuser -u "$REMOTE_USER" -- timeout --signal=TERM --kill-after=10 "$BENCHMARK_TIMEOUT_SEC" \
    "$REMOTE_ROOT/.venv/bin/python" -m vllm bench serve \
    --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model "$model_dir" --tokenizer "$tokenizer_dir" --served-model-name "$served_name" \
    --dataset-name random --input-len "$PROMPT_TOKENS" --output-len "$GEN_TOKENS" \
    --num-prompts 1 --num-warmups "$WARMUPS" --label "$case_id-bench" --ignore-eos \
    --save-result --save-detailed --append-result --result-dir "$case_dir" \
    --result-filename "$(basename "$bench_jsonl")" \
    --metadata group="$group" mode="$mode" model_dir="$model_dir" tokenizer_dir="$tokenizer_dir" served_model_name="$served_name" \
    >>"$case_dir/bench.out" 2>>"$case_dir/bench.err"; then
    printf 'benchmark_failed\n' >"$case_dir/status"
    write_artifact_manifest failed benchmark_timeout_or_failure
    stop_runtime_vllm
    return 0
  fi

  write_artifact_manifest captured
  printf 'captured\n' >"$case_dir/status"
  chown -R "$REMOTE_USER:$REMOTE_USER" "$case_dir"
  stop_runtime_vllm
}

all_cases_file="$REMOTE_RESULT_DIR/all_cases.tsv"
cases_file="$REMOTE_RESULT_DIR/cases.tsv"
cat >"$all_cases_file" <<CASES
group	profile	model_dir	tokenizer_dir	mode	role	threshold
fp8_fp16kv	qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env	$FP8_MODEL_DIR	$FP8_TOKENIZER_DIR	safe	fp16_compare	0
fp8_fp16kv	qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env	$FP8_MODEL_DIR	$FP8_TOKENIZER_DIR	normal	fp16_compare	0
fp8_fp16kv	qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env	$FP8_MODEL_DIR	$FP8_TOKENIZER_DIR	fast	fp16_compare	0
int4_fp16kv	qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env	$INT4_MODEL_DIR	$INT4_TOKENIZER_DIR	safe	fp16_compare	0
int4_fp16kv	qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env	$INT4_MODEL_DIR	$INT4_TOKENIZER_DIR	normal	fp16_compare	0
int4_fp16kv	qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env	$INT4_MODEL_DIR	$INT4_TOKENIZER_DIR	fast	fp16_compare	0
fp8_int8kv	qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env	$FP8_MODEL_DIR	$FP8_TOKENIZER_DIR	fast	fast_guard	70
int4_int8kv	qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env	$INT4_MODEL_DIR	$INT4_TOKENIZER_DIR	fast	fast_guard	90
CASES

if [[ -f "$REMOTE_ROOT/profiles/qwen27b/experimental/fp8/tqk8v4-256K-mtp3-text-only.env" ]]; then
  printf 'fp8_tqk8v4\tqwen27b/experimental/fp8/tqk8v4-256K-mtp3-text-only.env\t%s\t%s\tfast\tfast_guard\t65\n' "$FP8_MODEL_DIR" "$FP8_TOKENIZER_DIR" >>"$all_cases_file"
fi

if [[ "$FAST_ONLY" == "1" ]]; then
  awk -F'\t' 'NR == 1 || $5 == "fast"' "$all_cases_file" >"$cases_file"
else
  cp "$all_cases_file" "$cases_file"
fi

if [[ -n "$CASE_FILTER" ]]; then
  awk -F'\t' -v pat="$CASE_FILTER" '
    NR == 1 {
      print
      next
    }
    ($1 "\t" $2 "\t" $5) ~ pat
  ' "$cases_file" >"$cases_file.filtered"
  mv "$cases_file.filtered" "$cases_file"
fi

tail -n +2 "$cases_file" | while IFS=$'\t' read -r group profile model_dir tokenizer_dir mode role threshold; do
  run_one_case "$group" "$profile" "$model_dir" "$tokenizer_dir" "$mode" "$role" "$threshold"
done

summary_args=(
  --summarize "$REMOTE_RESULT_DIR"
  --expected-runs "$MEASURED_RUNS"
  --fast-margin "$FAST_MARGIN"
)
if [[ "$FAST_ONLY" == "1" ]]; then
  summary_args+=(--fast-only)
fi
if [[ -n "$BASELINE_RESULT_DIR" ]]; then
  summary_args+=(--baseline-result-dir "$BASELINE_RESULT_DIR")
fi
if [[ "$FAST_COMPARE_REQUIRED" == "1" ]]; then
  summary_args+=(--fast-compare-required)
fi
runuser -u "$REMOTE_USER" -- "$REMOTE_ROOT/.venv/bin/python" \
  "$REMOTE_ROOT/tools/validate_benchmark_manifest.py" "${summary_args[@]}"
REMOTE

echo "Remote result: $REMOTE_RESULT_DIR"
