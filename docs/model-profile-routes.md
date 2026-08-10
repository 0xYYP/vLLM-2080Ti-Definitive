# Model Profile Routes

This document records the evidence rules behind the deployment profiles. The
current profile catalog, meanings, and measured throughput live in
[Profile Guide](../profiles/README.md).

`Profile` is the relative `.env` path selected by `launcher.sh`; the concrete
checkpoint is selected separately with `MODEL_DIR`.

## Evidence Rule

Full pass means a real request returns HTTP 200, finishes the stream, and the
Chinese quality smoke has no repetition collapse, broken output, garbling, or
clear wrong-response behavior.

A memory plateau only proves lower capacity risk. If quality smoke fails, the
route is not promoted as a profile even when capacity or synthetic throughput
looks usable.

Load-only, READY-only, health-only, small-window smoke, and empty-stream results
are not capacity evidence.

## KV Positioning

- FP16/default KV is the quality route.
- INT8 KV is the capacity / balance route; currently shipped only as
  `normal` / piecewise profiles. Decode defaults to the dequant bridge
  (continuation/cascade chunked path) instead of the native O(KV) full scan,
  which degraded linearly with context. Measured on 0.6.16rc4 (warm,
  completions, MTP3, dual 2080 Ti, 272 layout): 4K 29.56 / 60K 21.46 / 100K
  16.08 tok/s decode (native extrapolation ~1.5-2 tok/s at 250K). A/B
  (2026-08-10, same vLLM 6426afb, only flashinfer switched): 0.6.8.post1 vs
  0.6.16rc4 show no bridge-path difference (4K 29.80 vs 29.56); the 2026-08-07
  records (4K 70.16) are not reproducible under the current fixed code —
  confirmed unrelated to the flashinfer version. A replay on the historical
  code (c805572 + 0.6.8.post1 + 260 layout) measured 4K 30.05 / 60K 20.07 /
  100K 20.06 / 125K 16.19, so all three 2026-08-07 records (70.16 / 44.23 /
  32.09) stay **permanently marked as non-reproducible historical
  measurements** (unrelated to the flashinfer version; the true cause was
  that session's measurement method). Note: absolute throughput differs
  between the 260-layout replay and the current 272 layout (100K 20.06 vs
  16.08, ~+25%); the layout's effect on absolute throughput was not isolated.
  The
  experimental FlashInfer decode variant (`VLLM_INT8KV_FA_DECODE=1`) is
  slower than the bridge (4K 18.15 tok/s, occupancy-bound) and stays off by
  default. The 245K preset (`int8kv-245K-mtp3-text-only.env`) is a
  configuration ceiling verified for prefix-cache-hit decode only; cold-start
  prefill above ~60K OOMs (262144 OOMs on a single 100K write). See
  `docs/int8kv-fa-decode.md`.
- TQK8V4 is the TurboQuant compression route; currently shipped only for
  quality-passed `fast` profiles.
- TQ4NC had capacity experiments, but is not used in the current shipped
  profiles.

## Notes

- Profiles are organized as `profiles/<model>/<mode>/<weight>/<route>.env`.
- `normal` is the current recommended production route. `fast` keeps only
  high-performance routes that passed quality smoke. `safe` is the launcher
  eager fallback mode, not the current shipped profile directory.
- The same dual-2080-Ti runtime also validates Qwen3.6 35B FP8 MoE lanes. The
  shipped preset set now covers 256K `normal` and `aggressive` noMTP
  text-only lanes, 136K `normal` and `aggressive` noMTP text+image lanes, and
  a 178K `fast` MTP3 speed preset.
- FP8 + FP16KV `normal` is formally 256K and passes a long-prompt smoke at
  `262016/128`.
- FP8 + FP16KV `aggressive` is also validated at 256K. Its recorded throughput
  stays on the `4096/128` synthetic lane, while the near-full `262016/128`
  smoke is treated as a capacity proof because stream chunk coalescing can
  overstate long-run decode speed.
- FP8 + FP16KV text+image is validated at 136K in both `normal` and
  `aggressive`. Both passed `138240/128`; `139008/64` also passed at the edge,
  while `139136/32` exceeds the configured `139264` limit.
- FP8 + FP16KV `fast` is currently validated at 178K and passes a long-prompt
  smoke at `182144/128`.
- FP8 + TQK8V4 is validated at 256K for text-only and 240K for text-image.
  The image route uses GPU util 0.96.
- fast + INT8KV is not kept: capacity or synthetic throughput may pass, but
  Chinese quality smoke showed repeated or broken output.
- Legacy `fast` + INT8KV compatibility issues should be reported and fixed, but
  those fixes do not promote the route back into the shipped fast catalog.
- Throughput background is kept in
  [Qwen3.6 KV Throughput Sweep](qwen36-kv-throughput-sweep.md) and
  [MTP Task Sensitivity](mtp-task-sensitivity.md).
