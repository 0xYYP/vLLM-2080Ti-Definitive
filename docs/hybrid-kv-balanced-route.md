# Hybrid KV Balanced Route

This is an experimental route for the long-context balance target:

- KV memory below `0.80x` of fp16 KV.
- PP65536/TG512 long-context decode at least `0.70x` of the fp16 KV control.
- Chinese quality smoke and Needle-in-a-Haystack are effectively unchanged
  against the fp16 control.

The route is not a new KV format. It uses vLLM's existing
`--kv-cache-dtype-skip-layers` support: most attention layers use a compact KV
dtype, while selected attention layers skip KV quantization and stay on
fp16/default KV.

## Candidate Profile

The current Qwen3.6 27B FP8 candidate is:

```text
profiles/qwen27b/experimental/fp8/hybrid-fp8kv-65K-mtp3-text-only.env
```

This profile exists to test allocator compatibility and the speed/quality
tradeoff. FP8 KV is not yet a recommended 2080 Ti route; promote it only if the
validation gates below beat the existing fp16/int8/TQ choices.

The INT8 candidate is still kept as the quality-oriented compact-KV route:

```text
profiles/qwen27b/experimental/fp8/hybrid-int8kv-65K-mtp3-text-only.env
```

There is also an all-INT8 65K diagnostic control:

```text
profiles/qwen27b/experimental/fp8/int8kv-65K-mtp3-text-only.env
```

Use it to compare against hybrid candidates at the same request-sized context
limit. It mirrors the normal all-INT8 capacity route with `MAX_MODEL_LEN`
tightened to `66048`. The shipped 252K all-INT8 route is a capacity route and
can exercise different long-decode behavior, so it should not be used as the
only speed control for the 65K balance gate.

Earlier hybrid skip-layer profiles could fail startup on Qwen hybrid models
because compact KV pages, fp16 skip pages, and Mamba align padding were computed
from different page sizes. The Mamba align path now includes the fp16 skip page
in its compatible page-size calculation, but both FP8 and INT8 hybrid profiles
remain experimental until server startup, throughput, and quality evidence are
recorded. The INT8 candidate explicitly disables the INT8 FlashInfer prefill
path because its generated SM75 head-dim-256 kernel can fail NVCC compilation on
the current CUDA 12.8 / FlashInfer 0.6.8 runtime; this is a conservative startup
route, not a promoted performance route.

It targets the 65K validation lane instead of maximum context capacity. The
route is experimental until the throughput and quality gates below have real
dual RTX 2080 Ti evidence.

The skip list was generated from a Qwen3.6 27B config with 16 attention layers:

```bash
python3 tools/hybrid_kv_plan.py \
  --model-dir "$MODEL_DIR" \
  --kv-dtype fp8
```

For `head_size=256`, fp8 KV estimates to `0.5000x` fp16 KV per quantized
attention layer. Keeping 9 of 16 attention layers in fp16 gives an estimated
hybrid KV ratio of `0.7812x`, below the `0.80x` capacity gate.

## Validation Gates

Use the same `MODEL_DIR`, GPU pair, `MODE`, port, request shape, tokenizer, and
sampling settings for all controls.

1. Print-config gate:

```bash
MODEL_DIR="$MODEL_DIR" \
PROFILE=qwen27b/experimental/fp8/hybrid-fp8kv-65K-mtp3-text-only.env \
MODE=fast \
./launcher.sh --print-config
```

The output must show:

- `KV precision: fp8`
- `KV fp16 skip layers: 3,11,19,27,35,39,47,55,63`
- `MAX_MODEL_LEN=66048`
- `MAX_BATCHED_TOKENS=2560`
- `MTP_K=3`

2. Throughput gate:

Run the fp16 control first, then the hybrid candidate. Record at least one
warmup and multiple measured runs if possible.

```bash
tools/profile_request.py \
  --model-dir "$MODEL_DIR" \
  --served-name qwen27b-fp8-fp16kv-112K-mtp3-text-only-cu128 \
  --base-url http://127.0.0.1:8000/v1 \
  --endpoint completions \
  --prompt-tokens 65536 \
  --gen-tokens 512 \
  --label fp16_65k \
  --out /tmp/hybrid_kv_65k.jsonl \
  --ignore-eos \
  --pure-filler
```

```bash
tools/profile_request.py \
  --model-dir "$MODEL_DIR" \
  --served-name qwen27b-fp8-hybrid-fp8kv-65K-mtp3-text-only-cu128 \
  --base-url http://127.0.0.1:8000/v1 \
  --endpoint completions \
  --prompt-tokens 65536 \
  --gen-tokens 512 \
  --label hybrid_fp8_65k \
  --out /tmp/hybrid_kv_65k.jsonl \
  --ignore-eos \
  --pure-filler
```

Pass condition:

```text
hybrid_decode_tok_s >= fp16_decode_tok_s * 0.70
```

The historical FP8 MTP3 fp16 KV reference at PP65536/TG512 is `70.8 tok/s`, so
the reference target is about `49.6 tok/s`. This historical number is only a
sanity reference; the pass/fail comparison must use the fp16 control from the
same run.

3. Quality gate:

- Run the Chinese quality smoke on fp16, the all-compact-KV control, and hybrid.
- Run the same NIAH points on fp16, the all-compact-KV control, and hybrid.
  Start with the known long-context middle-depth points before expanding to a
  full heatmap.
- The needle text must be present in generated samples before blaming KV
  quantization.

Promote the route only if fp16 passes, the all-compact-KV control does not
reveal an eval or prompt issue, and hybrid has no material smoke/NIAH loss.
