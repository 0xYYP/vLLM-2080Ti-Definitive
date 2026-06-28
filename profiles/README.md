# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

This directory contains launch profiles for vLLM 2080Ti Definitive. A profile
is an `.env` preset for runtime parameters only; it does not include the
checkpoint path. Choose the model directory separately in `launcher.sh` or with
`MODEL_DIR=...`.

The shipped context and throughput numbers were validated on 2x RTX 2080 Ti
22GB cards with tensor parallel size 2.

Profile layout:

```text
profiles/
  templates/
  qwen27b/
    normal/
      fp8/
      int4/
    fast/
      fp8/
      int4/
    user/
```

Launch modes:

- `safe`: conservative fallback mode for maximum compatibility.
- `normal`: recommended daily mode for stable deployments.
- `fast`: high-performance mode for higher throughput.
- `aggressive`: more aggressive mode with the highest performance and quality risk.

`profiles/templates/` contains optional chat-template presets. They are selected
from the launcher as a global service setting; route profiles do not store chat
templates, GPU devices, ports, reasoning defaults, or tool-calling defaults.
For the shipped Qwen3/Qwen3.6 routes, the launcher fills in `qwen3` as the
reasoning parser when one is not set so startup smoke and chat parsing stay
aligned with the model's default reasoning behavior. Set `REASONING_PARSER=off`
if you need to run without a reasoning parser for diagnostics.

File names describe the intended route:

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV positioning:

- `fp16kv`: quality route.
- `int8kv`: capacity / balance route; currently shipped only as `normal`
  profiles.
- `tqk8v4`: TurboQuant K8V4 compression route; currently shipped only for
  quality-passed `fast` profiles.

The shipped TQK8V4 profiles use `MAX_BATCHED_TOKENS=2560`, which is the
validated setting for the prefix-cache path with aligned Qwen hybrid cache
blocks.

## Validated Profiles

### FP8

Tested checkpoint: Jackrong/Qwopus3.6-27B-v2-FP8, about 29G.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

### AWQ/GPTQ-INT4

Tested checkpoint: GPTQ-INT4, about 19G.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 250K per workspace | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 250K per workspace | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |

Throughput uses the `4096/128` test shape and is shown as
`prefill tok/s / decode tok/s`. Chinese quality smoke is run before throughput;
routes that fail quality are not kept as profiles.
