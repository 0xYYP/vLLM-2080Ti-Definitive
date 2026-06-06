# Model Profile Routes

This document records the user-facing profile matrix for the dual RTX 2080 Ti /
SM75 runtime. `Profile` is the `.env` file selected by `start.sh`; the concrete
checkpoint is still selected separately with `MODEL_DIR`.

The active profile matrix is evidence-gated. A context value is promoted only after a
real large-prompt request returns HTTP 200, finishes the stream, and generates
at least one completion token. Load-only, READY-only, health-only, and empty
stream results are not capacity evidence.

Profile names encode the launcher mode:

- `stable-*`: stable mode, FP16/default KV only.
- `speed-*`: speed mode, used for quantized KV, capacity, and high-performance
  routes. This is the default evidence mode for future profile validation.

The table schema is:

`Profile | Mode | Weight precision | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note`

Labels such as `256K` are human-readable tiers; the exact token values live in
the profile files.

Throughput details are kept in
[Qwen3.6 KV Throughput Sweep](qwen36-kv-throughput-sweep.md) and
[MTP Task Sensitivity](mtp-task-sensitivity.md).

## Qwen3.6 27B - FP8

| Profile | Mode | Weight precision | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| stable-qwen27-fp8-fp16kv-mtp3-100k.env | stable | FP8 | FP16/default | 100K | 0.92 | 2048 | 1 | MTP3 | text | Strict large-prompt pass at 100K; default FP8 quality/stability route. |
| speed-qwen27-fp8-int8kv-mtp3-240k.env | speed | FP8 | INT8 | 240K | 0.95 | 2048 | 1 | MTP3 | text | Strict large-prompt pass at 240K. The older 256K row returned an empty stream and is not active. |
| speed-qwen27-fp8-int8kv-yarn-mtp3-216k.env | speed | FP8 | INT8 + YaRN | 216K | 0.94 | 2048 | 1 | MTP3 | text | Strict large-prompt pass at 216K. 224K and 512K failed admission, so this is not a 512K route. |

## Qwen3.6 27B - INT4

FP8 and INT4 checkpoint routes use the Marlin weight path in this runtime. AWQ,
GPTQ, and future compatible INT4-style formats are checkpoint packaging choices,
not separate profile families.

| Profile | Mode | Weight precision | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| stable-qwen27-int4-fp16kv-mtp3-256k.env | stable | INT4 | FP16/default | 256K | 0.90 | 2048 | 1 | MTP3 | text | Strict large-prompt pass at full native context; primary INT4 full-context route. |
| speed-qwen27-int4-int8kv-workspace2-mtp3.env | speed | INT4 | INT8 | 128K x 2 | 0.90 | 2048 | 2 | MTP3 | text | Strict large-prompt pass at 128K with two sequences. This is workspace isolation, not parallel long-prefill throughput. |

## Demoted Routes

These routes are intentionally not active deployment profiles:

| Route | Status | Reason |
|---|---|---|
| FP8 INT8 256K | Demoted | 256K started but returned an empty stream; 240K is the strict pass. |
| FP8 INT8 + YaRN 512K | Demoted | 512K failed KV admission; the current strict pass is 216K. |
| FP8 TQK8V4 256K | Experimental | Speed-mode probe failed startup at 256K and 248K; 240K opened the request but became GPU-idle/CPU-bound, so it is not a strict pass. |
| Qwen INT4 TQ4NC workspace4 | Demoted | 128K down to 64K returned empty streams under strict request validation. |
| Qwen INT4 / FP8 multimodal TQ4NC | Demoted | No strict image route currently exists; previous long-prompt probes failed or were not strict evidence. |
| Qwen INT4 TQ4NC + YaRN 1M | Demoted | No strict large-prompt pass; not an active capacity profile. |
| Gemma4 FP16/default KV | Demoted | 100K and lower large-context startup probes failed; keep only short-context speed evidence. |
| Gemma4 INT8 KV | Demoted | 256K down to 64K failed startup with the page-size unification error. |
| Gemma4 TQ4NC KV | Demoted | 43K startup could be reached after fixes, but the real request produced no completion tokens. |

## Current Conclusion

- Active deployment profiles are limited to the five files listed above.
- Qwen remains the mature route. MTP3 stays in the active profile set because it
  has shown useful decode speedup and has strict large-prompt evidence in these
  profile shapes.
- Gemma remains a supported checkpoint for speed experiments, but there is no
  active Gemma capacity profile until a strict large-prompt request passes.
- Compressed KV routes that only load, report capacity, or return empty streams
  are treated as experimental artifacts, not deployment profiles.
