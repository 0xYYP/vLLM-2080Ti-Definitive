# Non-Interactive Launcher

`launcher.sh` supports a complete non-interactive startup path for scripted
deployment, CI checks, and reproducible route validation.

The final precedence is:

`CLI > ENV > PROFILE > default`

`CLI` includes direct flags such as `--max-model-len 262144`, `--set
KEY=VALUE`, and `--unset KEY`.

## Flag Naming

Known launcher keys can be passed directly as `--lower-kebab-case`:

- `MODEL_DIR` -> `--model-dir`
- `PROFILE` -> `--profile`
- `MAX_MODEL_LEN` -> `--max-model-len`
- `MM_LIMIT_JSON` -> `--mm-limit-json`

For advanced environment variables that are not part of the direct launcher
surface, use:

- `--set KEY=VALUE`
- `--unset KEY`

`--unset KEY` clears inherited profile or environment values and then lets the
launcher fall back to lower-priority defaults. If you need to keep an explicit
empty string instead of falling back, use `--set KEY=`.
For example, `--unset COMPILATION_CONFIG_JSON` selects the generated compilation
config instead of treating the cleared value as an invalid explicit JSON value.

## Common Examples

Print the resolved launch summary without starting the server:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env
```

Start from a shipped profile and override only a few fields:

```bash
./launcher.sh --non-interactive \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --mode normal \
  --service-scope lan \
  --max-model-len 196608 \
  --gpu-devices 0,1
```

Environment values still work, but CLI wins over them:

```bash
MAX_MODEL_LEN=131072 \
MODE=fast \
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --max-model-len 262144
```

Use `--set` for advanced runtime envs:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set VLLM_TURBOQUANT_FLASHINFER_BACKEND=fa2 \
  --set CC=/usr/bin/gcc-12
```

Clear a profile-provided field and let the launcher refill the lower-level
default:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --unset max-model-len
```

Force an explicit empty value:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set REASONING_PARSER=
```

## Derived Defaults

Some launcher fields are still normalized after profile loading:

- `MESSAGE_TYPE`, `MM_LIMIT_JSON`, `LANGUAGE_MODEL_ONLY`, and
  `SKIP_MM_PROFILING`
- mode-derived runtime knobs such as `ENFORCE_EAGER`,
  `VLLM_SM75_SPEC_SYNC_MODE`, and
  `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH`
- family defaults such as `REASONING_PARSER=qwen3` on validated Qwen3 routes
- Qwen prefix-cache defaults such as `MAMBA_CACHE_MODE=align`

Those defaults only fill missing values. Explicit CLI values keep priority.

## Validation Workflow

Use `--print-config` first when changing scripts or deployment automation. It
prints the final launch summary and exits before starting vLLM, which makes it
the safest way to verify that every override landed on the final config.

## Machine-Readable Resolution Contract

`--print-config` emits `resolved_<key>=<value>` and
`resolved_<key>_source=<source>` fields for values that affect the final launch.
The source is one of `cli`, `env`, `profile`, `default`, `generated`, or
`derived`. The values continue to follow `CLI > ENV > PROFILE > default`; a
generated or derived value fills a value that was not supplied by those layers.

`COMPILATION_CONFIG_JSON` is a boundary value, not free-form launcher text. It
must be a JSON object. The launcher rejects malformed JSON, arrays, scalars, and
an explicit empty value before a server starts. A valid object is emitted in
canonical JSON form, so semantically identical input has stable machine-readable
output. An explicitly selected missing profile is an error and never falls back
silently to defaults.

`GPU_DEVICES` accepts a comma-separated list of numeric, non-negative GPU IDs.
Whitespace around numeric items is normalized. Empty input or empty items,
duplicate IDs, UUID values, and MIG values are rejected. UUID and MIG device
selection is intentionally outside this numeric CPU contract.

When `TP_SIZE` is omitted, it is derived from the number of resolved GPU IDs.
When supplied, `TP_SIZE` must be a positive integer equal to that count. The
resolved fields identify whether TP was supplied or derived.

`final_vllm_argv` is the shell-escaped final argument vector that would be sent
to vLLM after all normalization. It is emitted by `--print-config`; no service,
network request, model download, or GPU probing is performed by that command.
