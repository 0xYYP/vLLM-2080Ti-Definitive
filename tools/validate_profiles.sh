#!/usr/bin/env bash
set -euo pipefail

ROOT=${STABLE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
PROFILE_DIR=${PROFILE_DIR:-"$ROOT/profiles"}

if [[ ! -d "$PROFILE_DIR" ]]; then
  echo "profile directory not found: $PROFILE_DIR" >&2
  exit 2
fi

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

total=0
errors=0

while IFS= read -r -d '' file; do
  ((total += 1))
  base=$(basename "$file")
  mode=$(read_profile_value "$file" MODE)
  kv=$(read_profile_value "$file" KV_CACHE_DTYPE)

  case "$base" in
    stable-*.env)
      expected=stable
      ;;
    speed-*.env)
      expected=speed
      ;;
    *)
      echo "ERROR $base: profile filename must start with stable- or speed-" >&2
      ((errors += 1))
      continue
      ;;
  esac

  if [[ "$mode" != "$expected" ]]; then
    echo "ERROR $base: MODE=$mode does not match filename prefix $expected" >&2
    ((errors += 1))
  fi

  if [[ "$mode" == "stable" ]]; then
    case "$kv" in
      ""|fp16|default|auto)
        ;;
      *)
        echo "ERROR $base: stable mode only supports FP16/default KV, got $kv" >&2
        ((errors += 1))
        ;;
    esac
  fi
done < <(find "$PROFILE_DIR" -maxdepth 1 -type f -name '*.env' -print0 | sort -z)

if ((errors > 0)); then
  echo "profile_validation_failed total=$total errors=$errors" >&2
  exit 1
fi

echo "profile_validation_ok total=$total"
