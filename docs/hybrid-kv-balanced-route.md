# Hybrid KV Balanced Route

This is an experimental route for the long-context balance target:

- KV memory below `0.80x` of fp16 KV.
- PP65536/TG512 long-context decode at least `0.70x` of the fp16 KV control.
- Chinese quality smoke and Needle-in-a-Haystack are effectively unchanged
  against the fp16 control.

The route is not a new KV format. It uses vLLM's existing
`--kv-cache-dtype-skip-layers` support: most attention layers use
`int8_per_token_head`, while selected attention layers skip KV quantization and
stay on fp16/default KV.

## Candidate Profile

The current Qwen3.6 27B FP8 candidate is:

```text
profiles/qwen27b/experimental/fp8/hybrid-int8kv-65K-mtp3-text-only.env
```

It targets the 65K validation lane instead of maximum context capacity. The
route is experimental until the throughput and quality gates below have real
dual RTX 2080 Ti evidence.

The skip list was generated from a Qwen3.6 27B config with 16 attention layers:

```bash
python3 tools/hybrid_kv_plan.py \
  --model-dir "$MODEL_DIR" \
  --aligned-int8
```

For `head_size=256`, aligned `int8_per_token_head` estimates to `0.5312x` fp16
KV per quantized attention layer. Keeping 9 of 16 attention layers in fp16 gives
an estimated hybrid KV ratio of `0.7949x`, below the `0.80x` capacity gate.

## Validation Gates

Use the same `MODEL_DIR`, GPU pair, `MODE`, port, request shape, tokenizer, and
sampling settings for all controls.

1. Print-config gate:

```bash
MODEL_DIR="$MODEL_DIR" \
PROFILE=qwen27b/experimental/fp8/hybrid-int8kv-65K-mtp3-text-only.env \
MODE=fast \
./launcher.sh --print-config
```

The output must show:

- `KV precision: int8_per_token_head`
- `KV fp16 skip layers: 3,11,19,27,35,39,47,55,63`
- `MAX_MODEL_LEN=66048`
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
  --served-name qwen27b-fp8-hybrid-int8kv-65K-mtp3-text-only-cu128 \
  --base-url http://127.0.0.1:8000/v1 \
  --endpoint completions \
  --prompt-tokens 65536 \
  --gen-tokens 512 \
  --label hybrid_int8_65k \
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

- Run the Chinese quality smoke on fp16, all-int8, and hybrid.
- Run the same NIAH points on fp16, all-int8, and hybrid. Start with the known
  long-context middle-depth points before expanding to a full heatmap.
- The needle text must be present in generated samples before blaming KV
  quantization.

Promote the route only if fp16 passes, all-int8 does not reveal an eval or
prompt issue, and hybrid has no material smoke/NIAH loss.
