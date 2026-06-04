# vLLM SM75 CUDA 12.8 Runtime Probe

Created 2026-05-29 as an isolated venv copy from the previous stable/experiment
vLLM lane. Purpose: replace the PyTorch runtime from CUDA 13.x to CUDA 12.8
while preserving local vLLM, FlashQLA, FlashInfer, MTP, and KV-cache patches.

## 2026-05-29 validation

- Runtime changed from `torch 2.11.0+cu130` to `torch 2.11.0+cu128` in this
  isolated venv only.
- Light validation passed: torch CUDA reports 12.8, CUDA matmul works, vLLM
  0.21.0 imports, and FlashInfer 0.6.8 imports.
- Qwen-family 27B AWQ MTP3 non-eager service started successfully on dual RTX
  2080 Ti with AWQ Marlin, FlashInfer attention, FlashQLA legacy GDN prefill,
  and BF16 draft compatibility.
- First cu128 startup recompiled the vLLM torch compile cache because the
  previous cache was built under a different PyTorch runtime.
- PP4096/TG128 warm result: `1736.68 / 81.21 tok/s`.
- No CUDA launch failure, EngineDead, or OOM was seen in this validation. The
  FA2 sm80 warning still appears on 2080 Ti, but the route continues through
  FlashInfer/FlashQLA and is not fatal.

Throughput is written as `prefill / decode tok/s`.

## 2026-05-31 Qwen-family 100 tok/s single-request peak

Rechecked the Qwen-family 27B AWQ + native MTP3 route under the CUDA 12.8 stable
runtime after the 100 tok/s target was relaxed from a stable median requirement
to a single-row peak requirement.

Evidence:

- Runtime: `torch 2.11.0+cu128`, CUDA runtime 12.8, vLLM 0.21 local SM75 stack.
- Route: AWQ Marlin, FlashInfer sampler and attention, FlashQLA legacy GDN
  prefill, native MTP3, CUDAGraph/no-eager.
- Benchmark shape: single request, PP4096/TG128, `max_model_len=8192`,
  `max_num_seqs=1`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.86`.
- Peak row: `1747.52 / 100.98 tok/s`.
- Packaged route: `PROFILE=qwen27-awq-mtp3-peak`, which uses non-TurboQuant KV
  and the SM75 nosync speculative path.

This is the clean cu128 evidence for the project claim that dual RTX 2080 Ti
vLLM can reach 100 tok/s single-request decode. It is not a median and should
not be mixed with aggregate multi-request throughput.

## SM75 speculative-sync auto policy

`VLLM_SM75_SPEC_SYNC_MODE=auto` now adapts to the KV route:

- non-TurboQuant KV: skip the two async-spec stream syncs for higher decode
  throughput;
- `turboquant_*` KV: keep the syncs because the TQ/CUDAGraph route needs them
  for graph safety.

Manual overrides are available for debugging: `safe` always keeps the syncs,
while `nosync` always skips them.

## Gemma4 K12 no-eager retest on cu128

Retested a previously risky Gemma4-family parameter combination under the cu128
venv:

- target: `<MODEL_ROOT>/<GEMMA4_TARGET_CHECKPOINT>`
- draft: `<MODEL_ROOT>/<GEMMA4_ASSISTANT_CHECKPOINT>`
- KV: `turboquant_4bit_nc`
- MTP: assistant speculative decoding, `num_speculative_tokens=12`
- non-eager/CUDAGraph: `enforce_eager=False`, capture size 13
- max len 8192, max batched tokens 4096, max seqs 1, TP=2

Result: startup passed model/draft loading, TurboQuant attention selection,
FlashInfer/TQ prefill setup, and torch.compile. It failed during CUDAGraph
memory profiling / minimal KV cache setup with:

```text
NotImplementedError: The page size of the layer is not divisible by the maximum page size. Cannot unify by adjusting block_size.
```

This is not a CUDA launch failure. cu128 does not fix this exact K=12
non-eager+tq4nc Gemma4 route by itself; the failure points to Gemma4 mixed
KV/page-size compatibility with TQ4NC + assistant MTP12 + CUDAGraph.

## Gemma4 K3 no-eager retest on cu128

Retested the same Gemma4-family TQ4NC + assistant-MTP + non-eager route with
`num_speculative_tokens=3` and CUDAGraph capture size 4.

Result: failed with the same KV page-size unification error as K=12. Conclusion:
the failure is not specific to K=12.

## Qwen-family MTP3 eager A/B on cu128

Retested Qwen-family 27B AWQ MTP3 with the same PP4096/TG128 lane as the cu128
non-eager baseline, changing only `--enforce-eager`.

- non-eager baseline: `1736.68 / 81.21 tok/s`
- eager measure1: `1642.36 / 32.69 tok/s`
- eager measure2: `1651.07 / 30.15 tok/s`

Speculative acceptance stayed 100%, so the decode loss is not acceptance-related.
It is consistent with losing compile/CUDAGraph benefits.

Conclusion: on Qwen-family 27B MTP3, eager costs roughly 5% prefill and about
60% decode versus non-eager.

## Qwen-family MTP3 cu128 production-candidate recheck

Rechecked 2026-05-29 after a later MTP performance regression where structured
generation still passed but speculative acceptance was `0.0%` and decode fell
to roughly `20 tok/s`.

Root cause: the launcher did not pass the Qwopus BF16 draft compatibility flag
into the vLLM worker environment. The known-good cu128 logs had this variable
active, which makes the BF16 MTP draft load without inheriting the target
AWQ/Marlin quantization config. Without it, draft/target logits mismatch and
speculative decoding accepts nothing.

Launcher fix:

- `bench_tools/remote_start_vllm_cu128.sh` now preserves the Qwopus BF16 draft
  compatibility flag.

Validation command shape:

```bash
VLLM_QWOPUS_MTP_BF16_DRAFT=1 \
MODEL_DIR=<MODEL_ROOT>/<QWEN_FAMILY_AWQ_CHECKPOINT> \
SERVED_NAME=qwen27-awq-mtp3-cu128-bf16draft-verify \
PORT=19266 MAX_MODEL_LEN=8192 GPU_UTIL=0.90 MAX_BATCHED_TOKENS=4096 \
MAX_NUM_SEQS=1 MTP_K=3 MODEL_FAMILY=qwen QUANTIZATION=awq_marlin \
STABLE_ROOT=<STABLE_ROOT> \
bench_tools/remote_start_vllm_cu128.sh
```

Results:

- PP4096/TG128 run1: `1531.95 / 76.56 tok/s`
- PP4096/TG128 run2: `1713.74 / 80.50 tok/s`
- Synthetic acceptance returned to `100.0%`.
- Structured generation smoke remained stable: valid JSON `10/10`, strict pass
  `9/10`, mean latency about `2282 ms`, p95 about `3716 ms`, completion
  throughput about `49.65 tok/s`.
- Real workload speculative acceptance settled around `87-90%`.

Conclusion: CUDA 12.8 is the stable runtime target as long as the BF16 MTP draft
compatibility flag is part of the launch profile. The earlier regression was a
launch-profile mismatch, not a kernel performance loss.

## Gemma4 TQ4NC MTP3 non-eager shared-KV repair

Follow-up patching made the Gemma4-family `turboquant_4bit_nc` + assistant MTP3
+ non-eager route start and complete PP4096/TG128.

Fixes applied in the cu128 experiment venv:

- Shared draft KV spec lookup now resolves missing draft layer specs through
  `kv_sharing_target_layer_name`.
- A stale AOT/compile-cache failure is avoided by passing
  `VLLM_DISABLE_COMPILE_CACHE=1`.
- Gemma4 TQ hybrid KV grouping is enabled with
  `VLLM_GEMMA4_TQ4NC_GROUP_UNIFORM_TYPES=1`.
- Shared draft decode fallback no longer feeds a standard FP16 5D KV cache into
  the TurboQuant byte-dequant kernel. FP16 shared cache layout is handled as
  `[blocks, 2, block_size, Hk, D]`.
- Shared target/draft KV-head mismatch is handled by repeating or slicing cached
  KV heads to match the draft layer head count.
- The attempted FlashInfer paged-decode fast path for this FP16 shared cache is
  gated off by default (`VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER=0`). On this
  SM75 stack it hit FlashInfer `Unsupported max_mma_kv: 0` / Python GPF during
  experiments, so the stable route uses the gather fallback.

Validation:

- Synchronous debug run after the FP16 shared-cache/head-map fix completed
  PP4096/TG128 at `1030.40 / 5.43 tok/s`. This was only for correctness under
  `CUDA_LAUNCH_BLOCKING=1`.
- Normal runtime stable run completed PP4096/TG128 at `1201.46 / 11.14 tok/s`.

Conclusion: the route is correctness-stable enough to complete synthetic
PP4096/TG128 without the previous page-size, illegal-memory, or shared-KV shape
failures. It is not yet a performance win because decode is limited by the FP16
shared-cache gather fallback.

## Shared-FP16 fast-path follow-up

The attempted shared-FP16 fast paths must not be treated as stable:

- Triton shared-FP16 paged decode completed PP4096/TG128 but regressed to
  `1068.29 / 2.03 tok/s`, worse than stable gather.
- FlashInfer standalone decode can run the relevant SM75 shapes with
  non-tensor-core mode, but the ad-hoc vLLM integration calls
  `BatchDecodeWithPagedKVCacheWrapper.plan()` from attention forward. FlashInfer
  documents that `plan()` cannot be used inside CUDA Graph or torch.compile, so
  this is incompatible with the target non-eager/CUDAGraph route.
- Retrying shared-FP16 FlashInfer inside vLLM reached READY but the first
  PP4096/TG128 request hit CUDA launch failure and then a device loss. Do not
  enable this path outside isolated crash repros.

Current safety gate:

- `VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER=1` alone is not enough to enable
  FlashInfer shared-FP16.
- `VLLM_GEMMA4_TQ4NC_SHARED_FP16_TRITON=1` alone is not enough to enable Triton
  shared-FP16.
- Either fast path also requires
  `VLLM_GEMMA4_TQ4NC_SHARED_FP16_UNSAFE_EXPERIMENT=1`.
- The old FlashInfer helper that calls
  `BatchDecodeWithPagedKVCacheWrapper.plan()` from attention forward has an
  additional guard:
  `VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER_FORWARD_PLAN=1`.

The next real FlashInfer fix should move wrapper planning into the
metadata/runner setup path, not keep planning inside `turboquant_attn.py`
forward.

## FP8 TQK8V4 continuation SDPA auto-cap

Added an opt-in resource-control guard for TurboQuant continuation SDPA:

- `VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK`
- `VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS`

When both are set, the effective continuation chunk is capped by:

```text
min(q_chunk, max_qk_cells // seq_len)
```

The same path now writes chunk outputs into a preallocated output tensor instead
of accumulating chunks in a Python list and concatenating at the end. Defaults
remain unchanged when the env vars are unset.

Validation on dual RTX 2080 Ti, FP8 Qwen-family 27B, native MTP3,
`turboquant_k8v4`, `MAX_MODEL_LEN=69632`, `MAX_BATCHED_TOKENS=1024`,
`MAX_NUM_SEQS=1`:

- Manual `qchunk256`, `PP4096/TG128`: `1294.69 / 68.85 tok/s`.
- Auto-cap `qchunk512`, `max_qk_cells=16777216`, `PP4096/TG128`:
  `1347.02 / 74.15 tok/s`.
- Manual `qchunk256`, `PP65536/TG128`: `393.26 / 37.20 tok/s`.
- Auto-cap `qchunk512`, `max_qk_cells=16777216`, `PP65536/TG128`:
  `381.08 / 36.97 tok/s`.

Invalid boundary checks:

- `max_qk_cells=17301504`, `17825792`, and `18874368` all failed
  `PP65536/TG128` with `EngineCore encountered an issue` and
  `completion_tokens=0`.
- Bare `qchunk384` and `qchunk512` without the cap also failed real
  long-context generation.

Conclusion: `16777216` is the current tested safe auto-cap for this profile. It
turns an otherwise invalid `qchunk512` long-context run into a valid one and
improves short-context throughput, but it does not yet improve long-context
speed over manual `qchunk256`. Treat it as a resource-control improvement, not
the final W8A8 performance answer.
