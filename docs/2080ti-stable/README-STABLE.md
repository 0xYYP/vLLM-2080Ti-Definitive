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
- flash-attention-turing root: `<FLASH_ATTN_TURING_ROOT>`
- Stable root: `<STABLE_ROOT>`
- Model root: `<MODEL_ROOT>`

Stable profile families:

- `PROFILE=qwen27-awq-mtp3`
- `PROFILE=qwen27-awq-mtp3-tq4nc-fi`
- `PROFILE=qwen27-awq-mtp3-int8kv`
- `PROFILE=gemma4-gptq-tq4nc-mtp3`
- `PROFILE=gemma4-gptq-tq4nc-nomtp`

Current reference results:

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
- Gemma4-family 31B GPTQ, TQ4NC KV, MTP3, PP4096/TG128: about
  `1558-1580 / 43-44 tok/s` after the profile fix.
- Gemma4-family 31B GPTQ, TQ4NC KV, no-MTP, PP4096/TG128: about
  `1562-1596 / 31.5-31.7 tok/s` after the profile fix.

Throughput is written as `prefill / decode tok/s`.

Known boundaries:

- Some Qwen-family BF16 draft checkpoints require a launch compatibility flag;
  otherwise speculative acceptance can collapse because the draft model inherits
  the target quantization config.
- Qwen-family `tq4nc` on SM75 must keep FlashInfer/FA2 prefill enabled for
  head_dim=256. Forcing the Turing prefill path caused a large decode regression
  in this build.
- Qwen-family `tq4nc+MTP` with CUDAGraph requires the 2026-05-30 TurboQuant
  graph-safety patch: speculative continuation batches with no prefill tail must
  route through the TQ spec-decode path.
- FlashInfer sampler remains disabled by default for the stable Qwen-family TQ
  route. It is kept as a compatibility/debug knob, not as a performance default.
- Native `262144` context is the baseline target for Qwen-family checkpoints.
  Reported GPU KV token capacity above that number is concurrency capacity, not
  proof of valid single-request context beyond the model-declared limit.
- YaRN 524K is a RoPE-extension/capacity route and should be validated per model
  before serving real workloads.
- Gemma4-family stable route is GPTQ target plus an external assistant draft
  when MTP is enabled. The TQ4NC profile must keep FlashInfer TQ prefill enabled
  and should not enable D256/D512 TQ decode SDPA fallbacks by default.
- Gemma4-family shared-FP16 FlashInfer/Triton fast paths are safety-gated and
  remain experimental.
- Historical Qwen-family peak decode around `100 tok/s` is a recorded peak, not
  the current stable requirement.
