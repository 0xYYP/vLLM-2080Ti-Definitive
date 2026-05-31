# vLLM SM75 Stable Build - CUDA 12.8

Canonical build identity: `vllm-sm75-tp2-cu128`.

The directory name `/data/stable/vllm-sm75-tp2-cu128` is historical and model-biased. Treat this tree as the shared stable build for Miniclaw dual RTX 2080 Ti / SM75 / TP=2 / CUDA 12.8, not as a Qwen27-AWQ-only environment. Models may change; this build identity is tied to the hardware/runtime stack.

Promoted on 2026-05-29 as the main Miniclaw dual RTX 2080 Ti vLLM lane.

This build is the shared stable runtime for current Qwen27/Qwopus/QuantTrio and Gemma4 GPTQ vLLM routes.

Runtime contract:
- Host CUDA/toolkit: CUDA 12.8
- PyTorch runtime: torch 2.11.0+cu128
- vLLM: 0.21.0 with local SM75/FlashQLA/FlashInfer/TurboQuant/MTP patches
- FlashQLA: /data/stable/FlashQLA-SM70-SM75
- flash-attention-turing: /data/stable/flash-attention-turing-sm75-compare-20260520/src

Previous cu130 stable archive:
/data/stable/vllm-sm75-tp2-cu128-archive-20260529-cu130-before-cu128-stable

Stable profiles:
- PROFILE=qwopus27-awq-mtp3 bench_tools/stable_profile.sh
- PROFILE=qwopus27-awq-mtp3-tq4nc-fi bench_tools/stable_profile.sh
- PROFILE=quanttrio27-awq-mtp3 bench_tools/stable_profile.sh
- PROFILE=gemma4-gptq-tq4nc-mtp3 bench_tools/stable_profile.sh
- PROFILE=gemma4-gptq-tq4nc-nomtp bench_tools/stable_profile.sh

Current reference results:
- Qwopus27 AWQ MTP3 PP4096/TG128: about 1713.74 prefill tok/s, 80.50 decode tok/s. Router6 limit10: weighted 33.6, strict 9/10, valid JSON 10/10, completion throughput 49.65 tok/s.
- Qwopus27 AWQ MTP3 tq4nc + FlashInfer prefill PP4096/TG128: measure runs 1732.16/84.94 and 1746.72/85.33 prefill/decode tok/s. This requires `VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM=0`; without it, sm75 head_dim=256 falls back to flash-attn-turing and decode dropped to about 45 tok/s. Artifact: `results/qwopus27_awq_mtp3_tq4nc_flashinferprefill_cu128_candidate_repeat_20260529.jsonl`.
- Qwopus27 AWQ MTP3 tq4nc + FlashInfer prefill + CUDAGraph Router6 limit10 after the 2026-05-30 graph-safety fix: weighted 33.6, strict 9/10, valid JSON 10/10, mean latency 2149.51 ms, p95 3940.13 ms, prompt throughput 157.38 tok/s, decode throughput 53.31 tok/s, aggregate 210.70 tok/s. Artifact: `/home/max/Develop/Router6/results/qwopus27-awq-tq4nc-mtp3-cg-emptytailfix-router6-limit10-20260530.json`. Startup log: `/data/stable/vllm-sm75-tp2-cu128/vllm-qwopus27-awq-tq4nc-mtp3-cg-emptytailfix-20260530-20260529-235820.log`.
- Qwopus27 AWQ MTP3 tq4nc + FlashInfer prefill + CUDAGraph Router6 full manifest: weighted 89.94, strict 26/30, valid JSON 30/30, mean latency 1868.08 ms, p95 2126.35 ms, prompt throughput 179.76 tok/s, decode throughput 59.76 tok/s. Artifact: `/home/max/Develop/Router6/results/qwopus27-awq-tq4nc-mtp3-cg-stable-full60-router6-20260530.json`.
- Qwopus27 AWQ MTP3 tq4nc + FlashInfer prefill + CUDAGraph Ragent6 0.2.2 zh-CN full60 at 16k context: score 52/60, weighted 87.4, invalid 0, aborted 0, adapter errors 0, prefill/request 748.30 tok/s, decode/completion 33.01 tok/s, 181 turns. Result dir: `/home/max/Develop/Ragent6/results/qwopus27-awq-tq4nc-mtp3-cg-stable-ctx16k-ragent6-full60-20260530/eval`. Metrics: `/home/max/Develop/Ragent6/results/qwopus27-awq-tq4nc-mtp3-cg-stable-ctx16k-ragent6-full60-20260530/qwopus_ctx16k_full60_metrics.json`.
- QuantTrio Qwen3.6 27B AWQ MTP3 PP4096/TG128: about 1757-1770 prefill tok/s, 85-87 decode tok/s on current cu128 patched build.
- Gemma4 31B GPTQ tq4nc MTP3 PP4096/TG128: about 1558-1580 prefill tok/s, 43-44 decode tok/s. Router6 limit10 after the 2026-05-30 profile fix: weighted 33.4, strict 10/10, valid JSON 10/10, mean latency 3540.39 ms, p95 5076.41 ms. Artifact: `/home/max/Develop/Router6/results/gemma4-gptq-tq4nc-mtp3-profilefix-router6-limit10-20260530.json`.
- Gemma4 31B GPTQ tq4nc no-MTP PP4096/TG128: about 1562-1596 prefill tok/s, 31.5-31.7 decode tok/s. Router6 limit10 after the 2026-05-30 profile fix: weighted 32.76, strict 10/10, valid JSON 10/10, mean latency 2756.61 ms, p95 3307.54 ms. Artifact: `/home/max/Develop/Router6/results/gemma4-gptq-tq4nc-nomtp-profilefix-router6-limit10-20260530.json`.

Known boundaries:
- Qwopus MTP requires VLLM_QWOPUS_MTP_BF16_DRAFT=1, otherwise speculative acceptance collapses.
- Qwopus/Qwen `tq4nc` on sm75 must keep FlashInfer prefill enabled for head_dim=256. The profile `qwopus27-awq-mtp3-tq4nc-fi` carries this override; do not apply it globally to Gemma4 without a separate Gemma validation.
- Qwopus/Qwen `tq4nc+MTP` full CUDAGraph requires the 2026-05-30 TurboQuant graph-safety patch: speculative continuation batches with no prefill tail must route through the TQ spec-decode path, and FlashInfer sampler is disabled by default (`VLLM_USE_FLASHINFER_SAMPLER=0`) while TQ FlashInfer prefill remains enabled.
- `PROFILE=qwopus27-awq-mtp3-tq4nc-fi` defaults to `MAX_MODEL_LEN=16384`. 8192 works for Router6 but is too small for full Ragent6 0.2.2 zh-CN because some cases combine large prompts with a 2048-token output budget.
- Gemma4 stable route is GPTQ target plus original assistant safetensors. AWQ+tq4nc had quality failures.
- Gemma4 `tq4nc` must keep FlashInfer TQ prefill enabled on sm75 and must not enable the D256/D512 TQ decode SDPA fallback by default. Those decode fallbacks caused immediate repetitive garbage output even in no-MTP Router6.
- Gemma4 shared-FP16 FlashInfer/Triton fast paths are off by default and should remain experimental.
- Historical Qwen27 ~101 decode tok/s is a recorded peak, not a current reproducible stable requirement.
