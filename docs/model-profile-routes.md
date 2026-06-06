# Model Profile Routes

This document lists only the deployment profiles that have been tested in the
stable runtime path. Speed-mode and quantized-KV profiles are intentionally not
listed here until they are retested.

`Profile` is the `.env` file selected by `start.sh`; the concrete checkpoint is
selected separately with `MODEL_DIR`.

Evidence rule: a profile is listed only after a real large-prompt request
returns HTTP 200, finishes the stream, and generates at least one completion
token. Load-only, READY-only, health-only, and empty-stream results are not
capacity evidence.

## Qwen3.6 27B - Stable FP16 KV

| Profile | Weight precision | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note |
|---|---|---|---:|---:|---:|---:|---|---|---|
| stable-qwen27-fp8-fp16kv-mtp3-100k.env | FP8 | FP16/default | 100K | 0.92 | 2048 | 1 | MTP3 | text | Stable FP8 quality route; strict large-prompt pass at 100K. |
| stable-qwen27-int4-fp16kv-mtp3-256k.env | INT4 | FP16/default | 256K | 0.90 | 2048 | 1 | MTP3 | text | Stable INT4 full-context route; strict large-prompt pass at native 256K. |

## Notes

- Active stable profiles live directly under `profiles/`.
- Experimental snippets live under `profiles/experimental/` and are not part of
  this stable matrix.
- Speed-mode, INT8 KV, TurboQuant KV, YaRN, multimodal, and workspace profiles
  need a fresh validation pass before being promoted into this document.
- Throughput background is kept in
  [Qwen3.6 KV Throughput Sweep](qwen36-kv-throughput-sweep.md) and
  [MTP Task Sensitivity](mtp-task-sensitivity.md).
