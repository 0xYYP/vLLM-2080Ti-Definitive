# vLLM SM75 Stable Build - CUDA 12.8

Canonical build identity: `vllm-sm75-tp2-cu128`.

This is the shared stable build for a dual RTX 2080 Ti / SM75 / TP=2 / CUDA
12.8 vLLM runtime. Models may change; the build identity is tied to the
hardware/runtime stack.

Promoted on 2026-05-29 as the main dual RTX 2080 Ti vLLM lane.

Runtime contract:

- Host CUDA/toolkit: CUDA 12.8
- PyTorch runtime: `torch 2.11.0+cu128`
- vLLM: `0.21.0` with local SM75, FlashQLA, FlashInfer, TurboQuant, INT8 KV,
  and MTP/CUDAGraph safety patches
- FlashQLA root: `<FLASHQLA_ROOT>`
- Stable root: `<STABLE_ROOT>`
- Model root: `<MODEL_ROOT>`

Stable profile families:

- `PROFILE=qwen27-awq-mtp3`
- `PROFILE=qwen27-awq-mtp3-peak`
- `PROFILE=qwen27-awq-mtp3-tq4nc-fi`
- `PROFILE=qwen27-awq-mtp3-int8kv`
- `PROFILE=gemma4-gptq-tq4nc-mtp3`
- `PROFILE=gemma4-gptq-tq4nc-nomtp`
- Gemma4 FP16/default-KV + assistant MTP5 is the current peak benchmark route;
  keep it as an explicit benchmark/profile override rather than replacing the
  short-context TQ4NC service route.

Current reference results:

- Qwen-family 27B AWQ, native MTP3, PP4096/TG128, CUDA 12.8 single-request
  peak: `1747.52 / 100.98 tok/s`. This is the headline 100 tok/s evidence for
  the dual RTX 2080 Ti vLLM route; it is a single-row peak, not a median or
  aggregate throughput.
- Qwen-family 27B AWQ, MTP3, PP4096/TG128: about `1710-1750 / 80-85 tok/s`.
- Qwen-family 27B AWQ, MTP3, TQ4NC KV, FlashInfer prefill, PP4096/TG128:
  about `1730-1750 / 85 tok/s` when head_dim=256 stays on FlashInfer/FA2
  prefill.
- Qwen-family 27B AWQ, MTP3, TQ4NC KV, CUDAGraph graph-safety path: structured
  generation validation remained stable after the 2026-05-30 fix; representative
  decode throughput was about `53-60 tok/s` in agent-style workloads.
- Qwen-family 27B GPTQ, INT8 KV, YaRN factor 2, no-eager/CUDAGraph MTP3:
  long-context capacity smoke completed near 520K prompt tokens after the INT8
  continuation/cascade dequant bridge. This is a capacity route, not an
  interactive default.
- Gemma4-family 31B GPTQ, FP16/default KV, assistant MTP5, locked GPU1/GPU2,
  PP4096/TG128: warm peak `1655.65 / 99.64 tok/s`; warm median about
  `1657.26 / 98.89 tok/s`. A cold row reached `100.11 tok/s` decode, but its
  TTFT includes warmup and should not be used as prefill evidence.
- Gemma4-family 31B GPTQ, TQ4NC KV, no-MTP, PP4096/TG128: about
  `1562-1596 / 31.5-31.7 tok/s` after the profile fix. This is the current
  TQ4NC service-speed reference.
- Gemma4-family 31B GPTQ, TQ4NC KV, assistant MTP5, locked GPU1/GPU2,
  PP4096/TG128: historical negative evidence only. Warm rows stayed around
  `1660 / 11.6 tok/s` and streamed 128 one-token chunks, indicating that useful
  speculative acceptance did not materialize under this TQ4NC route.
- Gemma4-family 31B GPTQ, TQ4NC KV, MTP3, PP4096/TG128 historical locked-clock
  peak: `1622.20 / 50.54 tok/s`; median `1619.46 / 47.01 tok/s`. Keep this as
  historical compatibility evidence, not the current Gemma peak route.
Throughput is written as `prefill / decode tok/s`.

Known boundaries:

- Some Qwen-family BF16 draft checkpoints require a launch compatibility flag;
  otherwise speculative acceptance can collapse because the draft model inherits
  the target quantization config. Current profiles pass
  `VLLM_QWOPUS_MTP_BF16_DRAFT=1` for Qwopus-style BF16 MTP draft tensors.
- Qwen-family `tq4nc` on SM75 must keep FlashInfer/FA2 prefill enabled for
  head_dim=256. Disabling this path caused a large decode regression in this
  build.
- Qwen-family `tq4nc+MTP` with CUDAGraph requires the 2026-05-30 TurboQuant
  graph-safety patch: speculative continuation batches with no prefill tail must
  route through the TQ spec-decode path.
- FlashInfer sampler remains disabled by default for the stable Qwen-family TQ
  route. It is kept as a compatibility/debug knob, not as a performance default.
- Native `262144` context is the baseline target for Qwen-family checkpoints.
  Reported GPU KV token capacity above that number is concurrency capacity, not
  proof of valid single-request context beyond the model-declared limit.
- YaRN 524K is a Qwen-family RoPE-extension/capacity route. Gemma4-family is
  marked unsupported for this feature in the current stable runtime: its config
  uses Gemma4 per-layer `proportional/default` RoPE parameters, and a 524K probe
  requires `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` rather than applying a real YaRN
  scaling profile.
- Gemma4-family stable route is GPTQ target plus an external assistant draft
  when MTP is enabled. FP16/default KV + assistant MTP5 is the peak speed route.
  TQ4NC remains useful for compressed short-context/capacity compatibility, but
  TQ4NC + assistant MTP5 regresses decode heavily in the current evidence.
- Gemma4-family shared-FP16 FlashInfer/Triton fast paths are safety-gated and
  remain experimental.
- Qwen-family peak decode around `100 tok/s` is a recorded cu128 single-request
  peak, not the median service requirement.

SM75 sync policy:

- `VLLM_SM75_SPEC_SYNC_MODE=auto` is the default. It skips the two async-spec
  stream syncs for non-TurboQuant KV, which is the faster Qwen-family peak path.
- `turboquant_*` KV profiles set `VLLM_SM75_SPEC_SYNC_MODE=safe` and keep those
  syncs because they were added to prevent TurboQuant/CUDAGraph races.
- `PROFILE=qwen27-awq-mtp3-peak` is the convenience peak profile: non-TQ KV,
  native MTP3, `max_model_len=8192`, `max_num_batched_tokens=8192`,
  `gpu_memory_utilization=0.86`, and stats logging disabled.
- Qwen-family FP16/default-KV + native MTP5 is a useful manual interactive
  speed-feel override for single-user chat demos. Keep it as an explicit
  override/profile experiment; MTP3 remains the conservative stable reference
  because it is better balanced across acceptance rate and real workloads.
