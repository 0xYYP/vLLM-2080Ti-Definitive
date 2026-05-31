# vLLM Qwen27/Qwopus SM75 CUDA 12.8 Runtime Probe

Created 2026-05-29 as an isolated venv copy from the current stable/experiment vLLM lane.
Purpose: replace PyTorch runtime from cu13 to cu128 while preserving local vLLM/FlashQLA/FlashInfer/MTP patches.

## 2026-05-29 validation

- Runtime changed from torch 2.11.0+cu130 to torch 2.11.0+cu128 in this isolated venv only.
- Light validation passed: torch CUDA reports 12.8, CUDA matmul works, vLLM 0.21.0 imports, FlashInfer 0.6.8 imports, flash_attn_turing imports.
- Qwopus AWQ MTP3 non-eager service started successfully on dual RTX 2080 Ti with AWQ Marlin, FlashInfer attention, FlashQLA legacy GDN prefill, and MTP BF16 draft compatibility.
- First cu128 startup recompiled vLLM torch compile cache because the previous cache was built under a different PyTorch runtime.
- PP4096/TG128 warm result: prefill 1736.68 tok/s, decode 81.21 tok/s.
- No CUDA launch failure, EngineDead, or OOM was seen in this validation. The FA2 sm80 warning still appears on 2080 Ti, but the route continues via FlashInfer/FlashQLA and is not fatal.

## Gemma4 K12 no-eager retest on cu128

Retested the previously risky Gemma4 parameter combination under the cu128 venv:

- target: `/data/models/vllm/cyankiwi-gemma-4-31B-it-AWQ-4bit` (vLLM detects `compressed-tensors`, WNA16 Marlin)
- draft: `/data/models/vllm/google-gemma-4-31B-it-assistant`
- KV: `turboquant_4bit_nc`
- MTP: assistant speculative decoding, `num_speculative_tokens=12`
- non-eager/cudagraph: `enforce_eager=False`, capture size 13
- max len 8192, max batched tokens 4096, max seqs 1, TP=2

Result: startup passed model/draft loading, TurboQuant attention selection, FlashInfer/TQ prefill setup, and torch.compile. It failed during cudagraph memory profiling / minimal KV cache setup with:

`NotImplementedError: The page size of the layer is not divisible by the maximum page size. Cannot unify by adjusting block_size.`

This is not a CUDA launch failure and cu128 does not fix this exact K=12 non-eager+tq4nc Gemma4 route. The failure points to Gemma4 mixed KV/page-size compatibility with tq4nc + MTP12/cudagraph. Production services were restored after the test.

## Gemma4 K3 no-eager retest on cu128

Retested the same Gemma4+tq4nc+assistant-MTP+non-eager route with `num_speculative_tokens=3` and cudagraph capture size 4.

Result: failed with the same KV page-size unification error as K=12:

`NotImplementedError: The page size of the layer is not divisible by the maximum page size. Cannot unify by adjusting block_size.`

Conclusion: the failure is not specific to K=12. It affects this Gemma4 mixed-attention/SWA + `turboquant_4bit_nc` + assistant MTP + non-eager/cudagraph route even at MTP3. Production services restored after the test.

## Qwopus MTP3 eager A/B on cu128

Retested Qwopus AWQ MTP3 with the same PP4096/TG128 lane as the cu128 non-eager baseline, changing only `--enforce-eager`.

- non-eager baseline measure: prefill 1736.68 tok/s, decode 81.21 tok/s.
- eager measure1: prefill 1642.36 tok/s, decode 32.69 tok/s.
- eager measure2: prefill 1651.07 tok/s, decode 30.15 tok/s.
- SpecDecoding acceptance stayed 100%, so the decode loss is not acceptance-related; it is consistent with losing compile/cudagraph benefits.

Conclusion: on Qwopus/Qwen27 MTP3, eager costs roughly 5% prefill and about 60% decode versus non-eager. This supports the hypothesis that Gemma4 eager MTP underperforms because eager disables compile/cudagraph, even though Gemma4 non-eager currently fails on KV page-size unification.

## Qwopus MTP3 cu128 production-candidate recheck

Rechecked 2026-05-29 after a later Qwopus MTP performance regression where
Router6 quality still passed but speculative acceptance was `0.0%` and decode
fell to roughly `20 tok/s`.

Root cause: `remote_start_vllm_cu128.sh` did not pass
`VLLM_QWOPUS_MTP_BF16_DRAFT=1` into the vLLM worker environment. The old good
cu128 logs had this variable active, which makes the Qwopus BF16 MTP draft load
without inheriting the target AWQ/Marlin quantization config. Without it, the
draft/target logits mismatch and speculative decoding accepts nothing.

Launcher fix:

- `/data/stable/vllm-sm75-tp2-cu128/bench_tools/remote_start_vllm_cu128.sh`
  now preserves `VLLM_QWOPUS_MTP_BF16_DRAFT`.
- Backup:
  `/data/stable/vllm-sm75-tp2-cu128/bench_tools/remote_start_vllm_cu128.sh.bak-20260529-qwopus-env`.

Validation command shape:

```bash
VLLM_QWOPUS_MTP_BF16_DRAFT=1 \
MODEL_DIR=/data/models/vllm/mconcat-Qwopus3.6-27B-v2-AWQ-4bit \
SERVED_NAME=qwopus27-awq-mtp3-cu128-bf16draft-verify-20260529 \
PORT=19266 MAX_MODEL_LEN=8192 GPU_UTIL=0.90 MAX_BATCHED_TOKENS=4096 \
MAX_NUM_SEQS=1 MTP_K=3 MODEL_FAMILY=qwen QUANTIZATION=awq_marlin \
/data/stable/vllm-sm75-tp2-cu128/bench_tools/remote_start_vllm_cu128.sh
```

Results:

- PP4096/TG128 run1: prefill `1531.95 tok/s`, decode `76.56 tok/s`.
- PP4096/TG128 run2: prefill `1713.74 tok/s`, decode `80.50 tok/s`.
- Synthetic acceptance returned to `100.0%`.
- Router6 limit10:
  - weighted score `33.6`
  - strict pass `9/10`
  - valid JSON `10/10`
  - mean latency `2281.82 ms`
  - p95 latency `3715.82 ms`
  - completion throughput over summed request latency `49.65 tok/s`
- Router6 acceptance settled around `87-90%`.

Conclusion: cu128 is the correct Qwopus/Qwen27 production-candidate runtime
for the current Miniclaw CUDA 12.8 system, as long as the Qwopus BF16 MTP draft
compatibility flag is part of the launch profile. The earlier regression was
not a kernel performance loss; it was a launch-profile mismatch.

## Gemma4 tq4nc MTP3 non-eager shared-KV repair

Follow-up patching made the Gemma4 AWQ `turboquant_4bit_nc` + assistant MTP3 + non-eager route start and complete PP4096/TG128.

Fixes applied in the cu128 experiment venv:

- Shared draft KV spec lookup now resolves missing draft layer specs through `kv_sharing_target_layer_name`.
- A stale AOT/compile-cache failure is avoided by passing `VLLM_DISABLE_COMPILE_CACHE=1`.
- Gemma4 TQ hybrid KV grouping is enabled with `VLLM_GEMMA4_TQ4NC_GROUP_UNIFORM_TYPES=1`.
- Shared draft decode fallback no longer feeds a standard FP16 5D KV cache into the TurboQuant byte-dequant kernel. FP16 shared cache layout is handled as `[blocks, 2, block_size, Hk, D]`.
- Shared target/draft KV-head mismatch is handled by repeating or slicing cached KV heads to match the draft layer head count.
- The attempted FlashInfer paged-decode fast path for this FP16 shared cache is gated off by default (`VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER=0`). On this sm75 stack it hit FlashInfer `Unsupported max_mma_kv: 0` / Python GPF during experiments, so the stable route uses the gather fallback.

Validation artifacts:

- Synchronous debug run after the FP16 shared-cache/head-map fix completed PP4096/TG128:
  `results/gemma4_awqct_tq4nc_mtp3_noneager_headmapfix_cudablock_pp4096_tg128_20260529.jsonl`
  Result: prefill 1030.40 tok/s, decode 5.43 tok/s. This was only for correctness under `CUDA_LAUNCH_BLOCKING=1`.
- Normal runtime stable run completed PP4096/TG128:
  `results/gemma4_awqct_tq4nc_mtp3_noneager_stablegather_pp4096_tg128_20260529.jsonl`
  Result: prefill 1201.46 tok/s, decode 11.14 tok/s.

Conclusion: the route is now correctness-stable enough to complete synthetic PP4096/TG128 without the previous page-size, illegal-memory, or shared-KV shape failures. It is not yet a performance win: decode is still limited by the FP16 shared-cache gather fallback. A real fast path would need a working sm75 FlashInfer/FA2-compatible paged decode for the shared FP16 draft cache, or a dedicated lightweight paged decode kernel for this layout.

## Shared-FP16 fast-path follow-up

The attempted shared-FP16 fast paths must not be treated as stable:

- Triton shared-FP16 paged decode completed PP4096/TG128 but regressed to `prefill 1068.29 tok/s`, `decode 2.03 tok/s`, worse than stable gather (`1201.46/11.14`).
- The same synthetic repeated-token PP4096/TG128 prompt showed `Avg Draft acceptance rate: 0.0%` for both stable gather and sharedtriton, so this prompt does not prove Gemma MTP acceleration.
- FlashInfer standalone decode can run the relevant sm75 shapes with non-tensor-core mode (`D=256,Hq=16,Hkv=8` and `D=512,Hq=16,Hkv=2`), but the ad-hoc vLLM integration calls `BatchDecodeWithPagedKVCacheWrapper.plan()` from attention forward. FlashInfer documents that `plan()` cannot be used inside CUDA Graph or `torch.compile`, so this is incompatible with the target non-eager/cudagraph route.
- Retrying shared-FP16 FlashInfer inside vLLM reached READY but the first PP4096/TG128 request hit CUDA launch failure and then Xid 79/154 (`GPU has fallen off the bus`, `Node Reboot Required`). Do not enable this path on production.

Current local safety gate:

- `VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER=1` alone is not enough to enable FlashInfer shared-FP16.
- `VLLM_GEMMA4_TQ4NC_SHARED_FP16_TRITON=1` alone is not enough to enable Triton shared-FP16.
- Either fast path also requires `VLLM_GEMMA4_TQ4NC_SHARED_FP16_UNSAFE_EXPERIMENT=1`.
- The old FlashInfer helper that calls `BatchDecodeWithPagedKVCacheWrapper.plan()` from attention forward now has a second, narrower guard: `VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER_FORWARD_PLAN=1`. Keep it off outside isolated crash repros.

The next real FlashInfer fix should move wrapper planning into the metadata/runner setup path, not keep planning inside `turboquant_attn.py` forward.
See local design note `FLASHINFER_SHARED_FP16_PLAN.md`.

## Next safe validation when Miniclaw returns

1. Verify the node recovered:
   `nvidia-smi -L` and `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`.
2. Ensure no experimental process remains:
   `pgrep -af "VLLM::|vllm|ptxas|triton" || true`.
3. Sync the local safety-gated `turboquant_attn.py` into the cu128 experiment venv.
4. Start only the stable gather route: keep both shared-FP16 fast paths off and do not set `VLLM_GEMMA4_TQ4NC_SHARED_FP16_UNSAFE_EXPERIMENT`.
5. Re-run PP4096/TG128 and compare against `1201.46/11.14`.
6. Restore production services and health-check `18100/18110` before any further experiments.

Local helper for the above sequence:

`/home/max/tmp/gemma_noneager_fix/run_safe_stable_gather_validation.sh`
