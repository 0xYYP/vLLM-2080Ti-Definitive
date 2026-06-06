#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$(basename "$SCRIPT_DIR")" == "launcher" && -d "$SCRIPT_DIR/../profiles" ]]; then
  MANAGER_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
else
  MANAGER_ROOT="$SCRIPT_DIR"
fi
RUNTIME_ROOT=${RUNTIME_ROOT:-"$MANAGER_ROOT"}
PROFILE_DIR=${PROFILE_DIR:-"$MANAGER_ROOT/profiles"}
LOG_DIR=${LOG_DIR:-"$MANAGER_ROOT/run-logs"}
STATE_FILE=${STATE_FILE:-"$LOG_DIR/start-manager.state"}
STAMP=$(date +%Y%m%d-%H%M%S)

banner() {
  cat <<'EOF'
============================================================
 vLLM 2080 Ti Definitive Edition
 Service manager
 Author: github.com/weicj
============================================================
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_tty() {
  [[ -t 0 && -t 1 ]]
}

pause_enter() {
  is_tty || return 0
  read -r -p "Press Enter to continue..." _
}

normalize_bool() {
  case "${1,,}" in
    1|yes|y|true|on) echo 1 ;;
    *) echo 0 ;;
  esac
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

source_profile_defaults() {
  local file=$1
  [[ -f "$file" ]] || return 0

  local key value
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    if [[ -v "$key" ]]; then
      continue
    fi
    value=$(read_profile_value "$file" "$key")
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)
}

apply_profile_overrides() {
  local file=$1
  [[ -f "$file" ]] || return 0

  local key value
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    value=$(read_profile_value "$file" "$key")
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)
}

load_manager_state() {
  [[ -f "$STATE_FILE" ]] || return 0
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

save_manager_state() {
  mkdir -p "$LOG_DIR"
  {
    printf 'MODEL_DIR=%q\n' "${MODEL_DIR:-}"
    printf 'PROFILE_DIR=%q\n' "${PROFILE_DIR:-}"
    printf 'PROFILE=%q\n' "${PROFILE:-}"
    printf 'MODEL_FAMILY=%q\n' "${MODEL_FAMILY:-}"
    printf 'PROFILE_GROUP=%q\n' "${PROFILE_GROUP:-}"
    printf 'MODEL_VARIANT=%q\n' "${MODEL_VARIANT:-}"
    printf 'SERVED_NAME=%q\n' "${SERVED_NAME:-}"
    printf 'GPU_DEVICES=%q\n' "${GPU_DEVICES:-}"
    printf 'QUANTIZATION=%q\n' "${QUANTIZATION:-}"
    printf 'KV_CACHE_DTYPE=%q\n' "${KV_CACHE_DTYPE:-}"
    printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN:-}"
    printf 'GPU_UTIL=%q\n' "${GPU_UTIL:-}"
    printf 'MAX_BATCHED_TOKENS=%q\n' "${MAX_BATCHED_TOKENS:-}"
    printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS:-}"
    printf 'MTP_K=%q\n' "${MTP_K:-}"
    printf 'MESSAGE_TYPE=%q\n' "${MESSAGE_TYPE:-}"
    printf 'MM_LIMIT_JSON=%q\n' "${MM_LIMIT_JSON:-}"
    printf 'LANGUAGE_MODEL_ONLY=%q\n' "${LANGUAGE_MODEL_ONLY:-}"
    printf 'SKIP_MM_PROFILING=%q\n' "${SKIP_MM_PROFILING:-}"
    printf 'HF_OVERRIDES_JSON=%q\n' "${HF_OVERRIDES_JSON:-}"
    printf 'ADDITIONAL_CONFIG_JSON=%q\n' "${ADDITIONAL_CONFIG_JSON:-}"
    printf 'SPECULATIVE_CONFIG=%q\n' "${SPECULATIVE_CONFIG:-}"
    printf 'COMPILATION_CONFIG_JSON=%q\n' "${COMPILATION_CONFIG_JSON:-}"
    printf 'TP_SIZE=%q\n' "${TP_SIZE:-}"
    printf 'CHAT_TEMPLATE_FILE=%q\n' "${CHAT_TEMPLATE_FILE:-}"
    printf 'ATTENTION_BACKEND=%q\n' "${ATTENTION_BACKEND:-}"
    printf 'REASONING_PARSER=%q\n' "${REASONING_PARSER:-}"
    printf 'DEFAULT_CHAT_TEMPLATE_KWARGS=%q\n' "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
    printf 'ENFORCE_EAGER=%q\n' "${ENFORCE_EAGER:-}"
    printf 'NO_ASYNC_SCHEDULING=%q\n' "${NO_ASYNC_SCHEDULING:-}"
    printf 'DISABLE_HYBRID_KV_CACHE_MANAGER=%q\n' "${DISABLE_HYBRID_KV_CACHE_MANAGER:-}"
    printf 'DISABLE_PREFIX_CACHING=%q\n' "${DISABLE_PREFIX_CACHING:-}"
    printf 'DISABLE_CUSTOM_ALL_REDUCE=%q\n' "${DISABLE_CUSTOM_ALL_REDUCE:-}"
    printf 'DISABLE_LOG_STATS=%q\n' "${DISABLE_LOG_STATS:-}"
    printf 'MODE=%q\n' "${MODE:-stable}"
    printf 'PORT=%q\n' "${PORT:-8000}"
    printf 'SERVICE_SCOPE=%q\n' "${SERVICE_SCOPE:-local}"
  } > "$STATE_FILE"
}

list_profiles() {
  [[ -d "$PROFILE_DIR" ]] || return 0
  find "$PROFILE_DIR" -maxdepth 1 -type f -name '*.env' -printf '%f\n' | sort
}

list_profiles_for_model() {
  local family=$1
  local quantization=${2:-}
  local profile profile_file profile_family profile_variant

  while IFS= read -r profile; do
    [[ -n "$profile" ]] || continue
    profile_file="$PROFILE_DIR/$profile"
    profile_family=$(read_profile_value "$profile_file" MODEL_FAMILY)
    profile_variant=$(read_profile_value "$profile_file" MODEL_VARIANT)

    if [[ "$family" == qwen* && -n "$profile_family" && "$profile_family" != qwen* ]]; then
      continue
    fi
    if [[ "$family" == gemma* && -n "$profile_family" && "$profile_family" != gemma* ]]; then
      continue
    fi

    if [[ "$family" == qwen* && "$quantization" == fp8 && "$profile_variant" != fp8 ]]; then
      continue
    fi
    if [[ "$family" == qwen* && -n "$quantization" && "$quantization" != fp8 && "$profile_variant" == fp8 ]]; then
      continue
    fi

    printf '%s\n' "$profile"
  done < <(list_profiles)
}

gpu_device_count() {
  local devices=${1:-}
  local count=0 part
  devices=${devices// /}
  [[ -n "$devices" ]] || {
    echo 0
    return 0
  }
  IFS=',' read -r -a parts <<< "$devices"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && count=$((count + 1))
  done
  echo "$count"
}

list_nvidia_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null |
    awk -F, '
      {
        idx = $1
        name = substr($0, index($0, ",") + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        if (idx != "" && name != "") {
          print idx "\t" name
        }
      }
    '
}

profile_summary() {
  local profile_file=$1
  [[ -f "$profile_file" ]] || return 0

  local keys=(
    SERVED_NAME
    MODEL_FAMILY
    PROFILE_GROUP
    MODEL_VARIANT
    QUANTIZATION
    KV_CACHE_DTYPE
    MAX_MODEL_LEN
    GPU_UTIL
    MAX_BATCHED_TOKENS
    MAX_NUM_SEQS
    MTP_K
    MM_LIMIT_JSON
    HF_OVERRIDES_JSON
  )

  local key value
  for key in "${keys[@]}"; do
    value=$(read_profile_value "$profile_file" "$key")
    [[ -n "$value" ]] || value="-"
    printf '  %-22s %s\n' "$key" "$value"
  done
}

menu_select() {
  local title=$1
  local default=$2
  shift
  shift
  local options=("$@")
  local count=${#options[@]}
  local idx=0
  local key

  (( count > 0 )) || return 1
  for i in "${!options[@]}"; do
    if [[ "${options[$i]}" == "$default" ]]; then
      idx=$i
      break
    fi
  done

  if ! is_tty; then
    printf '%s\n' "${options[$idx]}"
    return 0
  fi

  while true; do
    clear >/dev/tty
    {
      banner
      echo "$title"
      echo
      for i in "${!options[@]}"; do
        if (( i == idx )); then
          printf ' > %s\n' "${options[$i]}"
        else
          printf '   %s\n' "${options[$i]}"
        fi
      done
      echo
      echo "Use ↑/↓, Enter to select."
    } >/dev/tty

    IFS= read -rsn1 key </dev/tty || true
    if [[ "$key" == $'\x1b' ]]; then
      read -rsn2 -t 0.1 key </dev/tty || true
      case "$key" in
        "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
        "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
      esac
    elif [[ "$key" == "" ]]; then
      printf '%s\n' "${options[$idx]}"
      return 0
    fi
  done
}

prompt_default() {
  local label=$1
  local default=$2
  local answer

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  read -r -p "$label [$default]: " answer </dev/tty
  if [[ -z "$answer" ]]; then
    printf '%s\n' "$default"
  else
    printf '%s\n' "$answer"
  fi
}

prompt_required_dir() {
  local label=$1
  local default=$2
  local answer path

  while true; do
    if [[ -n "$default" ]]; then
      read -r -p "$label [$default] (q to cancel): " answer </dev/tty
    else
      read -r -p "$label [required, q to cancel]: " answer </dev/tty
    fi
    case "${answer,,}" in
      q|quit|exit)
        return 1
        ;;
    esac
    [[ -z "$answer" && -n "$default" ]] && answer="$default"
    [[ -z "$answer" ]] && return 1

    path="$answer"
    if [[ "$path" == "~" ]]; then
      path="$HOME"
    elif [[ "$path" == "~/"* ]]; then
      path="$HOME/${path#~/}"
    fi
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
    echo "Directory does not exist: $path" >&2
    default="$path"
  done
}

prompt_checkpoint_dir() {
  local default=$1
  local answer path

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  while true; do
    if [[ -n "$default" ]]; then
      read -r -p "Checkpoint directory [$default] (q to quit): " answer </dev/tty
    else
      read -r -p "Checkpoint directory [required, q to quit]: " answer </dev/tty
    fi

    case "${answer,,}" in
      q|quit|exit)
        echo "Start cancelled." >&2
        return 1
        ;;
    esac

    if [[ -z "$answer" && -n "$default" ]]; then
      answer="$default"
    fi
    if [[ -z "$answer" ]]; then
      echo "No checkpoint directory selected. Start cancelled." >&2
      return 1
    fi

    path="$answer"
    if [[ "$path" == "~" ]]; then
      path="$HOME"
    elif [[ "$path" == "~/"* ]]; then
      path="$HOME/${path#~/}"
    fi

    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi

    echo "Checkpoint directory does not exist: $path" >&2
    default="$path"
  done
}

prompt_choice() {
  local title=$1
  local current=$2
  shift
  shift
  menu_select "$title" "$current" "$@"
}

prompt_segmented() {
  local title=$1
  local current=$2
  local left=$3
  local right=$4
  local answer

  while true; do
    if [[ "$current" == "$left" ]]; then
      printf '%s: [x] %s  [ ] %s\n' "$title" "$left" "$right" >/dev/tty
    else
      printf '%s: [ ] %s  [x] %s\n' "$title" "$left" "$right" >/dev/tty
    fi
    read -r -p "Choose left/right, 1/2, or Enter to keep: " answer </dev/tty
    case "${answer,,}" in
      "" )
        printf '%s\n' "$current"
        return 0
        ;;
      left|l|1)
        printf '%s\n' "$left"
        return 0
        ;;
      right|r|2)
        printf '%s\n' "$right"
        return 0
        ;;
      *)
        echo "Please choose left/right, 1/2, or Enter." >&2
        ;;
    esac
  done
}

confirm_start() {
  local answer

  if ! is_tty; then
    return 0
  fi

  while true; do
    read -r -p "Start server now? [y/N]: " answer </dev/tty
    case "$answer" in
      y|Y)
        return 0
        ;;
      n|N|"")
        echo "Start cancelled."
        return 1
        ;;
      *)
        echo "Please type y to start or n to exit."
        ;;
    esac
  done
}

show_help() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  cat <<'EOF'
This is the vLLM 2080 Ti Definitive service manager for a source checkout.

Main menu:
  1. Weight directory: choose the checkpoint directory.
  2. Profile: choose a profile directory, scan .env files, optionally apply one,
     then edit the filled runtime parameters.
  3. GPU / TP selection: select GPUs with Space; TP size follows GPU count.
  4. Launch mode: stable or speed.
  5. Port: default 8000.
  6. Service scope: local only or local + LAN.
  7. Help.
  8. Start service: launch vLLM, wait for /health, run a small smoke request,
     then print API URL, served model, PID file, and log file.
  9. Exit.

Profiles are optional. They are presets only; the current menu values are the
actual launch configuration.

Project:
  https://github.com/weicj/vLLM-2080Ti-Definitive

Notes:
  - stable mode is recommended for daily service.
  - speed mode explores higher throughput and may carry quality risk.
  - text+image requires a checkpoint that actually supports vision inputs.
  - --print-config prints the final launch summary and exits without starting.
EOF
  echo
  pause_enter
}

show_profiles() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Profile presets:"
  echo
  local profile profile_file family variant mode kv context mtp seqs
  if [[ ! -d "$PROFILE_DIR" ]]; then
    echo "No profile directory found: $PROFILE_DIR"
    echo
    pause_enter
    return 0
  fi
  while IFS= read -r profile; do
    [[ -n "$profile" ]] || continue
    profile_file="$PROFILE_DIR/$profile"
    family=$(read_profile_value "$profile_file" MODEL_FAMILY)
    variant=$(read_profile_value "$profile_file" MODEL_VARIANT)
    mode=$(read_profile_value "$profile_file" MODE)
    kv=$(read_profile_value "$profile_file" KV_CACHE_DTYPE)
    context=$(read_profile_value "$profile_file" MAX_MODEL_LEN)
    mtp=$(read_profile_value "$profile_file" MTP_K)
    seqs=$(read_profile_value "$profile_file" MAX_NUM_SEQS)
    printf '  %-50s mode=%-6s family=%-7s weight=%-6s kv=%-24s ctx=%-8s mtp=%-3s seqs=%s\n' \
      "$profile" "${mode:-auto}" "${family:-auto}" "${variant:-auto}" "${kv:-fp16}" "${context:-auto}" "${mtp:-0}" "${seqs:-1}"
  done < <(list_profiles)
  echo
  pause_enter
}

current_scope_label() {
  if [[ "${SERVICE_SCOPE:-local}" == "lan" ]]; then
    echo "local + LAN"
  else
    echo "local only"
  fi
}

current_profile_label() {
  if [[ -n "${PROFILE:-}" ]]; then
    echo "$PROFILE"
  else
    echo "none"
  fi
}

detect_default_gpu_devices() {
  local detected
  detected=$(
    list_nvidia_gpus 2>/dev/null |
      awk -F'\t' '
        BEGIN { sep = "" }
        tolower($2) ~ /2080[[:space:]]*ti/ {
          out = out sep $1
          sep = ","
          count++
          if (count == 2) {
            print out
            exit
          }
        }
      '
  ) || true
  if [[ -n "$detected" ]]; then
    printf '%s\n' "$detected"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
  else
    printf '0,1\n'
  fi
}

gpu_selected() {
  local idx=$1
  local devices=",${2// /},"
  [[ "$devices" == *",$idx,"* ]]
}

select_gpu_devices_menu() {
  local rows=() selected_devices idx=0 key count current_line gpu_idx gpu_name new_devices tp_count
  mapfile -t rows < <(list_nvidia_gpus || true)
  selected_devices=${GPU_DEVICES:-$(detect_default_gpu_devices)}

  if ((${#rows[@]} == 0)) || ! is_tty; then
    GPU_DEVICES=$(prompt_default "GPU devices / CUDA_VISIBLE_DEVICES" "$selected_devices")
    tp_count=$(gpu_device_count "$GPU_DEVICES")
    (( tp_count > 0 )) && TP_SIZE="$tp_count"
    save_manager_state
    return 0
  fi

  count=${#rows[@]}
  while true; do
    clear >/dev/tty
    {
      banner
      echo "GPU / TP selection"
      echo
      echo "Space toggles a GPU. Enter confirms. TP size follows selected GPU count."
      echo
      for i in "${!rows[@]}"; do
        current_line=${rows[$i]}
        gpu_idx=${current_line%%$'\t'*}
        gpu_name=${current_line#*$'\t'}
        if gpu_selected "$gpu_idx" "$selected_devices"; then
          mark="[x]"
        else
          mark="[ ]"
        fi
        if (( i == idx )); then
          printf ' > %s GPU %s  %s\n' "$mark" "$gpu_idx" "$gpu_name"
        else
          printf '   %s GPU %s  %s\n' "$mark" "$gpu_idx" "$gpu_name"
        fi
      done
      echo
      printf 'Selected: %s    TP_SIZE: %s\n' "${selected_devices:-none}" "$(gpu_device_count "$selected_devices")"
    } >/dev/tty

    IFS= read -rsn1 key </dev/tty || true
    if [[ "$key" == $'\x1b' ]]; then
      read -rsn2 -t 0.1 key </dev/tty || true
      case "$key" in
        "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
        "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
      esac
    elif [[ "$key" == " " ]]; then
      current_line=${rows[$idx]}
      gpu_idx=${current_line%%$'\t'*}
      if gpu_selected "$gpu_idx" "$selected_devices"; then
        new_devices=""
        IFS=',' read -r -a parts <<< "${selected_devices// /}"
        for part in "${parts[@]}"; do
          [[ -z "$part" || "$part" == "$gpu_idx" ]] && continue
          if [[ -n "$new_devices" ]]; then
            new_devices+=",$part"
          else
            new_devices="$part"
          fi
        done
        selected_devices="$new_devices"
      else
        if [[ -n "$selected_devices" ]]; then
          selected_devices+=",$gpu_idx"
        else
          selected_devices="$gpu_idx"
        fi
      fi
    elif [[ "$key" == "" ]]; then
      if [[ -z "$selected_devices" ]]; then
        echo "Select at least one GPU." >/dev/tty
        sleep 1
        continue
      fi
      GPU_DEVICES="$selected_devices"
      TP_SIZE=$(gpu_device_count "$GPU_DEVICES")
      save_manager_state
      return 0
    elif [[ "$key" == "q" || "$key" == "Q" ]]; then
      return 0
    fi
  done
}

select_weight_dir() {
  local selected
  selected=$(prompt_required_dir "Weight/checkpoint directory" "${MODEL_DIR:-}") || return 0
  MODEL_DIR="$selected"
  MODEL_FAMILY=$(guess_model_family "$MODEL_DIR")
  QUANTIZATION=$(guess_quantization "$MODEL_DIR")
  SERVED_NAME=$(basename "$MODEL_DIR")
  save_manager_state
}

select_profile_preset() {
  local selected_dir profiles=() selected profile_file choices=()
  selected_dir=$(prompt_required_dir "Profile directory" "${PROFILE_DIR:-$MANAGER_ROOT/profiles}") || return 0
  PROFILE_DIR="$selected_dir"
  mapfile -t profiles < <(list_profiles)
  if ((${#profiles[@]} == 0)); then
    echo "No .env profiles found under $PROFILE_DIR."
    echo
    if is_tty; then
      local answer
      read -r -p "Edit manual runtime parameters now? [y/N]: " answer </dev/tty
      case "$answer" in
        y|Y) edit_runtime_parameters ;;
      esac
    fi
    pause_enter
    return 0
  fi
  choices=("Return" "No profile" "Edit current parameters" "${profiles[@]}")
  selected=$(menu_select "Profile preset" "${PROFILE:-Return}" "${choices[@]}")
  case "$selected" in
    "Return")
      return 0
      ;;
    "No profile")
      PROFILE=""
      edit_runtime_parameters
      save_manager_state
      return 0
      ;;
    "Edit current parameters")
      edit_runtime_parameters
      save_manager_state
      return 0
      ;;
  esac
  PROFILE="$selected"
  profile_file="$PROFILE_DIR/$PROFILE"
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Applying profile preset: $PROFILE"
  echo
  profile_summary "$profile_file"
  echo
  echo "Profile only fills the editable runtime parameters below."
  echo
  apply_profile_overrides "$profile_file"
  pause_enter
  edit_runtime_parameters
  save_manager_state
}

prompt_optional() {
  local label=$1
  local default=${2:-}
  local answer

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  read -r -p "$label [${default:-empty}] (type '-' to clear): " answer </dev/tty
  if [[ "$answer" == "-" ]]; then
    printf '\n'
  elif [[ -z "$answer" ]]; then
    printf '%s\n' "$default"
  else
    printf '%s\n' "$answer"
  fi
}

prompt_toggle01() {
  local label=$1
  local default=${2:-0}
  local answer

  while true; do
    read -r -p "$label [$default] (0/1, Enter to keep): " answer </dev/tty
    case "$answer" in
      "")
        printf '%s\n' "$default"
        return 0
        ;;
      0|1)
        printf '%s\n' "$answer"
        return 0
        ;;
      *)
        echo "Please enter 0 or 1." >&2
        ;;
    esac
  done
}

edit_advanced_parameters() {
  local answer

  if ! is_tty; then
    return 0
  fi

  read -r -p "Edit advanced optional parameters? [y/N]: " answer </dev/tty
  case "$answer" in
    y|Y) ;;
    *) return 0 ;;
  esac

  echo
  CHAT_TEMPLATE_FILE=$(prompt_optional "Chat template file" "${CHAT_TEMPLATE_FILE:-}")
  ATTENTION_BACKEND=$(prompt_optional "Attention backend" "${ATTENTION_BACKEND:-}")
  REASONING_PARSER=$(prompt_optional "Reasoning parser" "${REASONING_PARSER:-}")
  DEFAULT_CHAT_TEMPLATE_KWARGS=$(prompt_optional "Default chat template kwargs JSON" "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}")
  HF_OVERRIDES_JSON=$(prompt_optional "HF overrides JSON" "${HF_OVERRIDES_JSON:-}")
  ADDITIONAL_CONFIG_JSON=$(prompt_optional "Additional config JSON" "${ADDITIONAL_CONFIG_JSON:-}")
  SPECULATIVE_CONFIG=$(prompt_optional "Speculative config JSON" "${SPECULATIVE_CONFIG:-}")
  COMPILATION_CONFIG_JSON=$(prompt_optional "Compilation config JSON" "${COMPILATION_CONFIG_JSON:-}")
  MM_LIMIT_JSON=$(prompt_optional "Multimodal limit JSON" "${MM_LIMIT_JSON:-}")
  if [[ -n "${MM_LIMIT_JSON:-}" && "${MESSAGE_TYPE:-text-only}" == "text-only" ]]; then
    MESSAGE_TYPE=text+image
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
  fi
  LANGUAGE_MODEL_ONLY=$(prompt_toggle01 "Language-model only" "${LANGUAGE_MODEL_ONLY:-1}")
  SKIP_MM_PROFILING=$(prompt_toggle01 "Skip multimodal profiling" "${SKIP_MM_PROFILING:-1}")
  ENFORCE_EAGER=$(prompt_toggle01 "Enforce eager" "${ENFORCE_EAGER:-0}")
  NO_ASYNC_SCHEDULING=$(prompt_toggle01 "No async scheduling" "${NO_ASYNC_SCHEDULING:-0}")
  DISABLE_HYBRID_KV_CACHE_MANAGER=$(prompt_toggle01 "Disable hybrid KV cache manager" "${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}")
  DISABLE_PREFIX_CACHING=$(prompt_toggle01 "Disable prefix caching" "${DISABLE_PREFIX_CACHING:-0}")
  DISABLE_CUSTOM_ALL_REDUCE=$(prompt_toggle01 "Disable custom all-reduce" "${DISABLE_CUSTOM_ALL_REDUCE:-0}")
  DISABLE_LOG_STATS=$(prompt_toggle01 "Disable log stats" "${DISABLE_LOG_STATS:-0}")
}

edit_runtime_parameters() {
  local answer kv_choice message_choice current_message_type

  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
    banner
    echo "Runtime parameters"
    echo
    echo "Press Enter to keep the current value. Type '-' on optional fields to clear."
    echo
  fi

  MODEL_FAMILY=$(prompt_default "Model family" "${MODEL_FAMILY:-$(guess_model_family "${MODEL_DIR:-}")}")
  PROFILE_GROUP=$(prompt_optional "Profile group" "${PROFILE_GROUP:-}")
  MODEL_VARIANT=$(prompt_optional "Weight precision/profile variant" "${MODEL_VARIANT:-}")
  SERVED_NAME=$(prompt_default "Served model name" "${SERVED_NAME:-${MODEL_DIR:+$(basename "$MODEL_DIR")}}")
  select_gpu_devices_menu
  QUANTIZATION=$(prompt_default "Weight quantization (empty/auto, fp8, gptq_marlin, awq_marlin, compressed-tensors)" "${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}")

  kv_choice=$(menu_select "KV precision" "${KV_CACHE_DTYPE:-fp16}" fp16 int8_per_token_head turboquant_k8v4 turboquant_4bit_nc)
  if [[ "$kv_choice" == "fp16" ]]; then
    KV_CACHE_DTYPE=""
  else
    KV_CACHE_DTYPE="$kv_choice"
  fi

  MAX_MODEL_LEN=$(prompt_default "Context tokens" "${MAX_MODEL_LEN:-$(default_context_tokens)}")
  GPU_UTIL=$(prompt_default "GPU memory utilization" "${GPU_UTIL:-$(default_gpu_util)}")
  MAX_BATCHED_TOKENS=$(prompt_default "Max batched tokens" "${MAX_BATCHED_TOKENS:-2048}")
  MAX_NUM_SEQS=$(prompt_default "Max concurrent sequences" "${MAX_NUM_SEQS:-1}")
  MTP_K=$(prompt_default "MTP speculative tokens" "${MTP_K:-0}")

  current_message_type=${MESSAGE_TYPE:-text-only}
  [[ -n "${MM_LIMIT_JSON:-}" ]] && current_message_type=text+image
  message_choice=$(menu_select "Message type" "$current_message_type" "text-only" "text+image")
  MESSAGE_TYPE="$message_choice"
  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
  edit_advanced_parameters

  save_manager_state
}

select_mode_menu() {
  MODE=$(prompt_segmented "Launch mode" "${MODE:-stable}" stable speed)
  save_manager_state
}

input_port_menu() {
  PORT=$(prompt_default "Port" "${PORT:-8000}")
  save_manager_state
}

select_scope_menu() {
  local selected
  selected=$(prompt_segmented "Service scope" "$(current_scope_label)" "local only" "local + LAN")
  if [[ "$selected" == "local + LAN" ]]; then
    SERVICE_SCOPE=lan
  else
    SERVICE_SCOPE=local
  fi
  save_manager_state
}

show_status() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Service status:"
  echo
  mkdir -p "$LOG_DIR"
  local found=0 pid_file pid name state cmd
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    found=1
    name=$(basename "$pid_file" .pid)
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      state="running"
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | sed 's/[[:space:]]*$//')
    else
      state="stale"
      cmd="-"
    fi
    printf '  %-36s pid=%-8s %s\n' "$name" "${pid:-unknown}" "$state"
    [[ "$cmd" != "-" ]] && printf '    %s\n' "$cmd"
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  if (( found == 0 )); then
    echo "  No pid files found under $LOG_DIR."
  fi
  echo
  pause_enter
}

show_launch_status() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Service status"
  echo
  echo "  Status:       START OK"
  echo "  Served model: ${SERVED_NAME:-unknown}"
  echo "  Model path:   ${MODEL_DIR:-unknown}"
  echo "  GPU devices:  ${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-unknown}}"
  echo "  Mode:         ${MODE:-stable}"
  echo "  Scope:        ${SERVICE_SCOPE:-local}"
  echo "  Local API:    ${LAST_API_LOCAL:-http://127.0.0.1:${PORT:-8000}/v1}"
  if [[ -n "${LAST_API_LAN:-}" ]]; then
    echo "  LAN API:      $LAST_API_LAN"
  fi
  echo "  PID file:     ${LAST_PID_FILE:-unknown}"
  echo "  Log file:     ${LAST_LOG_FILE:-unknown}"
  if [[ -n "${LAST_SMOKE_OUTPUT:-}" ]]; then
    echo "  Smoke:        $LAST_SMOKE_OUTPUT"
  fi
}

show_logs() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Recent logs:"
  echo
  mkdir -p "$LOG_DIR"
  local logs=() selected log_file
  mapfile -t logs < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' -printf '%T@ %f\n' 2>/dev/null | sort -nr | awk '{print $2}' | head -n 20)
  if ((${#logs[@]} == 0)); then
    echo "No logs found under $LOG_DIR."
    echo
    pause_enter
    return 0
  fi
  selected=$(menu_select "Log file" "${logs[0]}" "${logs[@]}")
  log_file="$LOG_DIR/$selected"
  clear >/dev/tty 2>/dev/null || true
  banner
  echo "Log: $log_file"
  echo
  tail -n "${LOG_TAIL_LINES:-120}" "$log_file" || true
  echo
  pause_enter
}

stop_service() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  mkdir -p "$LOG_DIR"
  local pid_files=() choices=() pid_file selected pid answer
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      pid_files+=("$pid_file")
      choices+=("$(basename "$pid_file" .pid) pid=$pid")
    fi
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  if ((${#choices[@]} == 0)); then
    echo "No running services found from $LOG_DIR pid files."
    echo
    pause_enter
    return 0
  fi
  selected=$(menu_select "Stop service" "${choices[0]}" "${choices[@]}")
  local index=-1
  for i in "${!choices[@]}"; do
    if [[ "${choices[$i]}" == "$selected" ]]; then
      index=$i
      break
    fi
  done
  (( index >= 0 )) || return 0
  pid_file="${pid_files[$index]}"
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if [[ -z "$pid" || ! -d "/proc/$pid" ]]; then
    echo "Service is no longer running."
    rm -f "$pid_file"
    pause_enter
    return 0
  fi
  read -r -p "Stop $selected? [y/N]: " answer </dev/tty
  case "$answer" in
    y|Y)
      kill "$pid" 2>/dev/null || true
      sleep 2
      if [[ -d "/proc/$pid" ]]; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
      rm -f "$pid_file"
      echo "Stop requested."
      ;;
    *)
      echo "Stop cancelled."
      ;;
  esac
  echo
  pause_enter
}

guess_model_family() {
  local dir=${1,,}
  if [[ "$dir" == *gemma* ]]; then
    echo gemma4
  else
    echo qwen
  fi
}

guess_quantization() {
  local dir=${1,,}
  if [[ "$dir" == *fp8* ]]; then
    echo fp8
  elif [[ "$dir" == *gptq* ]]; then
    echo gptq_marlin
  elif [[ "$dir" == *awq* ]]; then
    echo awq_marlin
  else
    echo ""
  fi
}

default_context_tokens() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 102400
  else
    echo 131072
  fi
}

default_gpu_util() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 0.92
  else
    echo 0.90
  fi
}

apply_mode() {
  case "$MODE" in
    speed)
      export DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
      if [[ "${MTP_K:-0}" =~ ^[0-9]+$ ]] && (( MTP_K > 0 )); then
        export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-nosync}
      fi
      ;;
    stable)
      if [[ "${MTP_K:-0}" =~ ^[0-9]+$ ]] && (( MTP_K > 0 )); then
        export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-safe}
      fi
      ;;
    *)
      die "MODE must be speed or stable."
      ;;
  esac
}

validate_mode_kv_policy() {
  local kv=${KV_CACHE_DTYPE:-}
  case "$MODE" in
    stable)
      case "$kv" in
        ""|fp16|default|auto)
          ;;
        *)
          echo "ERROR: stable mode only supports FP16/default KV." >&2
          echo "       Select a speed-* profile or set MODE=speed for quantized KV: $kv" >&2
          return 1
          ;;
      esac
      ;;
    speed)
      ;;
    *)
      echo "ERROR: MODE must be speed or stable." >&2
      return 1
      ;;
  esac
}

set_sm75_runtime_env() {
  local flashqla_candidate
  export STABLE_ROOT="$RUNTIME_ROOT"
  export HOME=${RUN_HOME:-"$HOME"}
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
  if [[ ! -x "$CUDA_HOME/bin/nvcc" && -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
  export CUDA_PATH="$CUDA_HOME"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-$(detect_default_gpu_devices)}}"
  export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
  if [[ -z "${FLASHQLA_ROOT:-}" ]]; then
    for flashqla_candidate in \
      "$RUNTIME_ROOT/FlashQLA-SM70-SM75" \
      "$MANAGER_ROOT/FlashQLA-SM70-SM75" \
      /opt/FlashQLA-SM70-SM75; do
      if [[ -d "$flashqla_candidate/flash_qla" ]]; then
        FLASHQLA_ROOT="$flashqla_candidate"
        break
      fi
    done
  fi
  export PYTHONPATH="$RUNTIME_ROOT${FLASHQLA_ROOT:+:$FLASHQLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
  export PATH="$RUNTIME_ROOT/.venv/bin:${CUDA_HOME}/bin:$PATH"
  export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
  # Keep generated kernels inside this runtime tree. Reusing cache dirs from
  # experiment worktrees can leave absolute paths to deleted environments.
  export TORCHINDUCTOR_CACHE_DIR="$MANAGER_ROOT/torchinductor-cache"
  export TRITON_CACHE_DIR="$MANAGER_ROOT/triton-cache"
  export PYTHONUNBUFFERED=1
}

build_args() {
  local host_arg=$1

  VLLM_ARGS=(
    --host "$host_arg"
    --port "$PORT"
    --model "$MODEL_DIR"
    --served-model-name "$SERVED_NAME"
    --dtype half
    --tensor-parallel-size "${TP_SIZE:-2}"
    --generation-config vllm
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  )

  [[ -n "${QUANTIZATION:-}" ]] && VLLM_ARGS+=(--quantization "$QUANTIZATION")
  [[ -n "${KV_CACHE_DTYPE:-}" ]] && VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  [[ "${ENFORCE_EAGER:-0}" == "1" ]] && VLLM_ARGS+=(--enforce-eager)
  [[ "${NO_ASYNC_SCHEDULING:-0}" == "1" ]] && VLLM_ARGS+=(--no-async-scheduling)
  [[ "${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}" == "1" ]] && VLLM_ARGS+=(--disable-hybrid-kv-cache-manager)
  [[ "${DISABLE_PREFIX_CACHING:-0}" == "1" ]] && VLLM_ARGS+=(--no-enable-prefix-caching)
  [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]] && VLLM_ARGS+=(--language-model-only)
  [[ "${SKIP_MM_PROFILING:-0}" == "1" ]] && VLLM_ARGS+=(--skip-mm-profiling)
  [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]] && VLLM_ARGS+=(--disable-custom-all-reduce)
  [[ "${DISABLE_LOG_STATS:-0}" == "1" ]] && VLLM_ARGS+=(--disable-log-stats)
  [[ -n "${ATTENTION_BACKEND:-}" ]] && VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
  [[ -n "${REASONING_PARSER:-}" ]] && VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
  [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]] && VLLM_ARGS+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
  [[ -n "${ADDITIONAL_CONFIG_JSON:-}" ]] && VLLM_ARGS+=(--additional-config "$ADDITIONAL_CONFIG_JSON")
  [[ -n "${HF_OVERRIDES_JSON:-}" ]] && VLLM_ARGS+=(--hf-overrides "$HF_OVERRIDES_JSON")

  if [[ -n "${MM_LIMIT_JSON:-}" ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt "$MM_LIMIT_JSON")
  elif [[ "$MODEL_FAMILY" == qwen* ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  elif [[ "$MODEL_FAMILY" == gemma* ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
  fi

  if [[ "$MODEL_FAMILY" == qwen* && -n "${MM_LIMIT_JSON:-}" && -z "${ADDITIONAL_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  fi

  if [[ "$MODEL_FAMILY" == qwen* && -z "${CHAT_TEMPLATE_FILE:-}" && -s "$MANAGER_ROOT/chat_template_no_think_ragent6.jinja" ]]; then
    CHAT_TEMPLATE_FILE="$MANAGER_ROOT/chat_template_no_think_ragent6.jinja"
  elif [[ "$MODEL_FAMILY" == qwen* && -z "${CHAT_TEMPLATE_FILE:-}" && -s "$RUNTIME_ROOT/chat_template_no_think_ragent6.jinja" ]]; then
    CHAT_TEMPLATE_FILE="$RUNTIME_ROOT/chat_template_no_think_ragent6.jinja"
  fi
  [[ -n "${CHAT_TEMPLATE_FILE:-}" ]] && VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE_FILE")

  local capture=$((MTP_K + 1))
  if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
    VLLM_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
  elif (( MTP_K > 0 )); then
    VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
  fi

  if [[ -n "${COMPILATION_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG_JSON")
  elif [[ -n "${SPECULATIVE_CONFIG:-}" || "$MTP_K" -gt 0 ]]; then
    VLLM_ARGS+=(--compilation-config "{\"cudagraph_capture_sizes\":[${capture}],\"max_cudagraph_capture_size\":${capture}}")
  else
    VLLM_ARGS+=(--compilation-config '{"cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}')
  fi
}

wait_for_ready() {
  local log_file=$1
  local url_host=$2
  local deadline=$((SECONDS + START_TIMEOUT))
  local fatal_regex='RuntimeError|ValueError|NotImplementedError|CUDA error|OutOfMemoryError|No supported config format|EngineCore encountered a fatal error|EngineDeadError'
  local pid=""
  if [[ -n "${CURRENT_SERVER_PID:-}" ]]; then
    pid="$CURRENT_SERVER_PID"
  fi

  while (( SECONDS < deadline )); do
    if curl -fsS "http://${url_host}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if grep -E "$fatal_regex" "$log_file" >/dev/null 2>&1; then
      return 1
    fi
    if [[ -n "$pid" && ! -d "/proc/$pid" ]]; then
      return 1
    fi
    sleep 3
  done
  return 2
}

smoke_test() {
  local url_host=$1
  local model_id model_output
  model_output=$("$RUNTIME_ROOT/.venv/bin/python" - "$url_host" "$PORT" <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=30) as resp:
    data = json.load(resp)
items = data.get("data") or []
if not items:
    raise SystemExit("no model returned")
print(items[0]["id"])
PY
)
  # Runtime sitecustomize hooks may print diagnostic lines on Python startup.
  model_id=$(printf '%s\n' "$model_output" | tail -n 1)

  "$RUNTIME_ROOT/.venv/bin/python" - "$url_host" "$PORT" "$model_id" <<'PY'
import json
import sys
import urllib.request

host, port, model_id = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 8,
    "temperature": 0,
}
req = urllib.request.Request(
    f"http://{host}:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.load(resp)
text = data["choices"][0]["message"].get("content", "")
if not text.strip():
    raise SystemExit("empty smoke response")
print(text.strip().replace("\n", " ")[:120])
PY
}

launch_server() {
  mkdir -p "$LOG_DIR"
  local safe_name log_file pid_file host_arg url_host args_text
  if [[ -z "${SERVED_NAME:-}" ]]; then
    SERVED_NAME=$(basename "$MODEL_DIR")
  fi
  GPU_DEVICES=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  TP_SIZE=${TP_SIZE:-$(gpu_device_count "$GPU_DEVICES")}
  if [[ -z "${SERVED_NAME:-}" || "$SERVED_NAME" == "." || "$SERVED_NAME" == "/" ]]; then
    echo "ERROR: Served model name is empty. Set SERVED_NAME or choose a valid checkpoint directory." >&2
    return 1
  fi
  safe_name=$(printf '%s' "$SERVED_NAME" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
  [[ -n "$safe_name" ]] || safe_name="vllm"
  log_file="$LOG_DIR/vllm-${safe_name}-${STAMP}.log"
  pid_file="$LOG_DIR/vllm-${safe_name}.pid"

  if [[ "$SERVICE_SCOPE" == "lan" ]]; then
    host_arg="0.0.0.0"
    url_host="127.0.0.1"
  else
    host_arg="127.0.0.1"
    url_host="127.0.0.1"
  fi

  build_args "$host_arg"
  printf -v args_text '%q ' "${VLLM_ARGS[@]}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo
    echo "DRY RUN"
    echo "Environment:"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "Command:"
    echo "  $RUNTIME_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server $args_text"
    return 0
  fi

  echo
  echo "Starting server..."
  echo "  Log: $log_file"
  echo "  Mode: $MODE"
  echo "  Served name: $SERVED_NAME"
  echo "  Model: $MODEL_DIR"
  echo "  Bind: $host_arg:$PORT"

  nohup "$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >"$log_file" 2>&1 &
  CURRENT_SERVER_PID=$!
  echo "$CURRENT_SERVER_PID" > "$pid_file"

  local ready_rc=0
  wait_for_ready "$log_file" "$url_host" || ready_rc=$?
  if [[ "$ready_rc" != "0" ]]; then
    echo
    echo "START FAILED"
    echo "Log: $log_file"
    tail -n 120 "$log_file" || true
    kill "$(cat "$pid_file" 2>/dev/null)" >/dev/null 2>&1 || true
    rm -f "$pid_file"
    return 1
  fi

  echo "Health check: OK"
  local smoke_output
  if ! smoke_output=$(smoke_test "$url_host" 2>&1); then
    echo
    echo "SMOKE FAILED"
    echo "$smoke_output"
    echo "Log: $log_file"
    kill "$(cat "$pid_file" 2>/dev/null)" >/dev/null 2>&1 || true
    rm -f "$pid_file"
    return 1
  fi

  local api_local="http://127.0.0.1:${PORT}/v1"
  local api_lan=""
  if [[ "$SERVICE_SCOPE" == "lan" ]]; then
    api_lan="http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/v1"
  fi
  LAST_PID_FILE="$pid_file"
  LAST_LOG_FILE="$log_file"
  LAST_API_LOCAL="$api_local"
  LAST_API_LAN="$api_lan"
  LAST_SMOKE_OUTPUT="$smoke_output"

  echo
  echo "START OK"
  echo "Smoke response: $smoke_output"
  echo "PID file: $pid_file"
  echo "Log: $log_file"
  echo "Local API: $api_local"
  if [[ -n "$api_lan" ]]; then
    echo "LAN API:   $api_lan"
  fi

  if is_tty; then
    show_launch_status
  fi
}

prepare_runtime_defaults() {
  if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "ERROR: MODEL_DIR is required. Choose item 1 first." >&2
    return 1
  fi
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory does not exist: $MODEL_DIR" >&2
    return 1
  fi
  MODEL_FAMILY=${MODEL_FAMILY:-$(guess_model_family "$MODEL_DIR")}
  SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL_DIR")}
  GPU_DEVICES=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  TP_SIZE=${TP_SIZE:-$(gpu_device_count "$GPU_DEVICES")}
  QUANTIZATION=${QUANTIZATION:-$(guess_quantization "$MODEL_DIR")}
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(default_context_tokens)}
  GPU_UTIL=${GPU_UTIL:-$(default_gpu_util)}
  MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  MTP_K=${MTP_K:-0}
  PORT=${PORT:-8000}
  MODE=${MODE:-stable}
  SERVICE_SCOPE=${SERVICE_SCOPE:-local}
  if [[ -z "${MESSAGE_TYPE:-}" && -n "${MM_LIMIT_JSON:-}" ]]; then
    MESSAGE_TYPE=text+image
  else
    MESSAGE_TYPE=${MESSAGE_TYPE:-text-only}
  fi
  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0}
    SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
  validate_mode_kv_policy
}

collect_config_env() {
  if [[ -n "${PROFILE_FILE:-}" ]]; then
    apply_profile_overrides "$PROFILE_FILE"
  elif [[ -n "${PROFILE:-}" && -f "$PROFILE_DIR/${PROFILE%.env}.env" ]]; then
    apply_profile_overrides "$PROFILE_DIR/${PROFILE%.env}.env"
  fi
  prepare_runtime_defaults || die "Invalid runtime configuration."
}

print_review() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  local message_type=text-only
  [[ -n "${MM_LIMIT_JSON:-}" ]] && message_type=text+image

  cat <<EOF
Launch summary:
  Model directory:      $MODEL_DIR
  Served name:          $SERVED_NAME
  Model family:         $MODEL_FAMILY
  Weight quantization:  ${QUANTIZATION:-auto}
  GPU devices:          ${GPU_DEVICES:-$(detect_default_gpu_devices)}
  KV precision:         ${KV_CACHE_DTYPE:-fp16}
  Context tokens:       $MAX_MODEL_LEN
  GPU util:             $GPU_UTIL
  Max batched tokens:   $MAX_BATCHED_TOKENS
  Max sequences:        $MAX_NUM_SEQS
  MTP tokens:           $MTP_K
  Message type:         $message_type
  Mode:                 $MODE
  Port:                 $PORT
  Scope:                $SERVICE_SCOPE
EOF
  echo
}

start_configured_service() {
  if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "Set item 1: weight/checkpoint directory first."
    pause_enter
    return 0
  fi

  if ! prepare_runtime_defaults; then
    pause_enter
    return 0
  fi

  apply_mode
  set_sm75_runtime_env
  START_TIMEOUT=${START_TIMEOUT:-900}
  print_review
  confirm_start || return 0

  if launch_server; then
    echo
    pause_enter
  else
    echo
    echo "Returned to main menu."
    pause_enter
  fi
}

menu_value() {
  local value=${1:-}
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf '<unset>'
  fi
}

render_main_menu() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Main menu"
  echo
  printf '  1. Weight directory: %s\n' "$(menu_value "${MODEL_DIR:-}")"
  printf '  2. Profile:          %s\n' "$(current_profile_label)"
  printf '     Profile dir:      %s\n' "$(menu_value "${PROFILE_DIR:-}")"
  printf '     Model family:     %s\n' "$(menu_value "${MODEL_FAMILY:-}")"
  printf '     Served name:      %s\n' "$(menu_value "${SERVED_NAME:-}")"
  printf '     GPU devices:      %s\n' "$(menu_value "${GPU_DEVICES:-$(detect_default_gpu_devices)}")"
  printf '     TP size:          %s\n' "$(menu_value "${TP_SIZE:-$(gpu_device_count "${GPU_DEVICES:-$(detect_default_gpu_devices)}")}")"
  printf '     Weight quant:     %s\n' "$(menu_value "${QUANTIZATION:-auto}")"
  printf '     KV precision:     %s\n' "$(menu_value "${KV_CACHE_DTYPE:-fp16}")"
  printf '     Context tokens:   %s\n' "$(menu_value "${MAX_MODEL_LEN:-$(default_context_tokens)}")"
  printf '     GPU util:         %s\n' "$(menu_value "${GPU_UTIL:-$(default_gpu_util)}")"
  printf '     Batch tokens:     %s\n' "$(menu_value "${MAX_BATCHED_TOKENS:-2048}")"
  printf '     Max sequences:    %s\n' "$(menu_value "${MAX_NUM_SEQS:-1}")"
  printf '     MTP tokens:       %s\n' "$(menu_value "${MTP_K:-0}")"
  printf '     Message type:     %s\n' "$(menu_value "${MESSAGE_TYPE:-text-only}")"
  echo "  3. GPU / TP selection"
  printf '  4. Launch mode:      %s\n' "${MODE:-stable}"
  printf '  5. Port:             %s\n' "${PORT:-8000}"
  printf '  6. Service scope:    %s\n' "$(current_scope_label)"
  echo "  7. Help"
  echo "  8. Start service"
  echo "  9. Exit"
  echo
  echo "Profile presets are optional. They only fill editable runtime parameters."
  echo
}

service_manager() {
  local choice
  load_manager_state
  MODE=${MODE:-stable}
  PORT=${PORT:-8000}
  SERVICE_SCOPE=${SERVICE_SCOPE:-local}

  while true; do
    render_main_menu
    read -r -p "Select [1-9]: " choice </dev/tty
    case "$choice" in
      1)
        select_weight_dir
        ;;
      2)
        select_profile_preset
        ;;
      3)
        select_gpu_devices_menu
        ;;
      4)
        select_mode_menu
        ;;
      5)
        input_port_menu
        ;;
      6)
        select_scope_menu
        ;;
      7)
        show_help
        ;;
      8)
        start_configured_service
        ;;
      9|q|Q|quit|exit)
        exit 0
        ;;
      "")
        ;;
      *)
        echo "Unknown choice: $choice"
        pause_enter
        ;;
    esac
  done
}

has_arg() {
  local want=$1
  shift || true
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$want" ]] && return 0
  done
  return 1
}

run_start_flow() {
  if [[ -n "${PROFILE_FILE:-}" ]]; then
    apply_profile_overrides "$PROFILE_FILE"
  elif [[ -n "${PROFILE:-}" && -f "$PROFILE_DIR/${PROFILE%.env}.env" ]]; then
    apply_profile_overrides "$PROFILE_DIR/${PROFILE%.env}.env"
  fi
  collect_config_env
  apply_mode
  set_sm75_runtime_env
  START_TIMEOUT=${START_TIMEOUT:-900}
  print_review
  if [[ "${PRINT_CONFIG:-0}" == "1" ]] || has_arg "--print-config" "$@"; then
    return 0
  fi
  launch_server
}

main() {
  cd "$MANAGER_ROOT"

  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
  fi

  if [[ ! -x "$RUNTIME_ROOT/.venv/bin/python" ]]; then
    banner
    die ".venv is missing under RUNTIME_ROOT=$RUNTIME_ROOT. Run ./build.sh first or set RUNTIME_ROOT."
  fi

  mkdir -p "$LOG_DIR"

  if [[ "${1:-}" == "--non-interactive" || "${1:-}" == "--print-config" || "${NON_INTERACTIVE:-0}" == "1" || ! -t 0 ]]; then
    run_start_flow "$@"
  else
    service_manager
  fi
}

main "$@"
