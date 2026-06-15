#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"

REMOTE_REPO=${REMOTE_REPO:-https://github.com/weicj/vLLM-2080Ti-Definitive.git}
REMOTE_RELEASE_API=${REMOTE_RELEASE_API:-https://api.github.com/repos/weicj/vLLM-2080Ti-Definitive/releases/latest}
UPDATE_PROMPT_TIMEOUT_SECONDS=${UPDATE_PROMPT_TIMEOUT_SECONDS:-10}
UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS=${UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS:-10}
UPDATE_DOWNLOAD_TIMEOUT_SECONDS=${UPDATE_DOWNLOAD_TIMEOUT_SECONDS:-60}

PRESERVE_PATHS=(
  ".venv"
  ".deps"
  "build-logs"
  "run-logs"
  "results"
  "torchinductor-cache"
  "triton-cache"
  ".omx"
  "backups"
  "profiles/qwen27b/user"
)

banner() {
  cat <<EOF
============================================================
 $PROJECT_NAME v$FORK_RELEASE
 Update helper
 Current tree: $ROOT
 Remote repo: $REMOTE_REPO
============================================================
EOF
}

prompt_yes_no_timeout() {
  local prompt=$1
  local timeout_seconds=${2:-10}
  local default_answer=${3:-y}
  local answer=""

  if [[ ! -t 0 ]]; then
    printf '%s\n' "$default_answer"
    return 0
  fi

  printf '%s ' "$prompt"
  if read -r -t "$timeout_seconds" answer; then
    case "$answer" in
      y|Y|yes|YES) printf 'y\n'; return 0 ;;
      n|N|no|NO) printf 'n\n'; return 0 ;;
      *) printf '%s\n' "$default_answer"; return 0 ;;
    esac
  fi
  printf '%s\n' "$default_answer"
}

latest_remote_version() {
  python3 - "$REMOTE_RELEASE_API" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    tag = data.get("tag_name", "").strip()
    if tag.startswith("v"):
        tag = tag[1:]
    if tag:
        print(tag)
except Exception:
    pass
PY
}

read_local_version() {
  tr -d '[:space:]' < "$ROOT/VERSION"
}

is_remote_newer() {
  local local_version=$1
  local remote_version=$2
  if [[ -z "$remote_version" || "$remote_version" == "$local_version" ]]; then
    return 1
  fi
  [[ "$(printf '%s\n%s\n' "$local_version" "$remote_version" | sort -V | tail -n 1)" == "$remote_version" ]]
}

download_archive() {
  local version=$1
  local tmpdir url archive_path strip_root
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT

  url="https://github.com/weicj/vLLM-2080Ti-Definitive/archive/refs/tags/v${version}.tar.gz"
  archive_path="$tmpdir/src.tar.gz"
  strip_root="vLLM-2080Ti-Definitive-${version}"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --max-time "$UPDATE_DOWNLOAD_TIMEOUT_SECONDS" -o "$archive_path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive_path" --timeout="$UPDATE_DOWNLOAD_TIMEOUT_SECONDS" "$url"
  else
    echo "Neither curl nor wget was found."
    exit 1
  fi

  tar -xzf "$archive_path" -C "$tmpdir"

  for path in "${PRESERVE_PATHS[@]}"; do
    [[ -e "$ROOT/$path" ]] || continue
    mkdir -p "$tmpdir/preserve/$(dirname -- "$path")"
    cp -a "$ROOT/$path" "$tmpdir/preserve/$path"
  done

  find "$ROOT" -mindepth 1 -maxdepth 1 \
    \( -name .git -o -name .venv -o -name .deps -o -name build-logs -o -name run-logs -o -name results -o -name torchinductor-cache -o -name triton-cache -o -name .omx -o -name backups -o -path "$ROOT/profiles/qwen27b/user" \) \
    -prune -o -exec rm -rf {} +

  cp -a "$tmpdir/$strip_root"/. "$ROOT"/

  for path in "${PRESERVE_PATHS[@]}"; do
    [[ -e "$tmpdir/preserve/$path" ]] || continue
    mkdir -p "$ROOT/$(dirname -- "$path")"
    cp -a "$tmpdir/preserve/$path" "$ROOT/$path"
  done
}

main() {
  banner

  local local_version remote_version
  local_version=$(read_local_version)
  remote_version=$(latest_remote_version || true)

  if [[ -n "$remote_version" ]]; then
    echo "Local version:  $local_version"
    echo "Remote version: $remote_version"
  else
    echo "Local version:  $local_version"
    echo "Remote version: unavailable"
  fi

  if ! is_remote_newer "$local_version" "$remote_version"; then
    echo "Already up to date."
    return 0
  fi

  echo "New release available: v$remote_version"
  local answer
  answer=$(prompt_yes_no_timeout "Update now? [Y/n]:" "$UPDATE_PROMPT_TIMEOUT_SECONDS" y)
  [[ "$answer" != "n" ]] || exit 0

  echo "Updating from release archive v$remote_version..."
  download_archive "$remote_version"

  echo "Update complete."
  echo "Current version: $(read_local_version)"

  answer=$(prompt_yes_no_timeout "Build now? [Y/n]:" "$UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS" y)
  if [[ "$answer" != "n" ]]; then
    exec bash "$ROOT/build.sh"
  fi
}

main "$@"
