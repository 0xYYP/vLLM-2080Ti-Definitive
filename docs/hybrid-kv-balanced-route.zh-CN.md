# Hybrid KV 平衡路线

这是一个实验路线，目标是验证长上下文下的平衡点：

- KV 显存低于 fp16 KV 的 `0.80x`。
- PP65536/TG512 长上下文 decode 不低于 fp16 KV control 的 `0.70x`。
- 中文质量 smoke 和 Needle-in-a-Haystack 相对 fp16 control 基本无损。

这不是新的 KV 格式。它使用 vLLM 已有的
`--kv-cache-dtype-skip-layers`：大部分 attention 层使用更紧凑的 KV dtype，
指定的 attention 层跳过 KV 量化，保留 fp16/default KV。

## 候选 Profile

当前 Qwen3.6 27B FP8 候选为：

```text
profiles/qwen27b/experimental/fp8/hybrid-fp8kv-65K-mtp3-text-only.env
```

这个 profile 用来验证 allocator 兼容性以及速度/质量取舍。FP8 KV 还不是已经推荐的
2080 Ti 路线；只有验证门槛证明它优于现有 fp16/int8/TQ 取舍后，才应晋升。

INT8 候选仍然保留，作为偏质量的 compact-KV 路线：

```text
profiles/qwen27b/experimental/fp8/hybrid-int8kv-65K-mtp3-text-only.env
```

早期 hybrid skip-layer profiles 在 Qwen hybrid 模型上可能启动失败，因为 compact KV
page、fp16 skip page 和 Mamba align padding 使用了不同的 page-size 口径。现在 Mamba
align 会把 fp16 skip page 纳入兼容 page-size 计算；但 FP8 和 INT8 hybrid profiles
都仍需服务器启动、吞吐和质量证据，才能脱离 experimental。INT8 候选会显式关闭
INT8 FlashInfer prefill 路径，因为当前 CUDA 12.8 / FlashInfer 0.6.8 runtime 下，
生成的 SM75 head-dim-256 kernel 可能 NVCC 编译失败；这是一条保守启动路线，不是
已晋升的性能路线。

它面向 65K 验证 lane，而不是最大上下文容量。只有拿到真实双 RTX 2080 Ti
吞吐和质量证据后，这条路线才能从 experimental 晋升。

skip list 来自 Qwen3.6 27B config，共 16 个 attention 层：

```bash
python3 tools/hybrid_kv_plan.py \
  --model-dir "$MODEL_DIR" \
  --kv-dtype fp8
```

在 `head_size=256` 下，fp8 KV 的单个量化 attention 层 KV 估算为 fp16 的
`0.5000x`。16 个 attention 层里保留 9 个 fp16 层，整体 hybrid KV 估算为
`0.7812x`，低于 `0.80x` 容量门槛。

## 验证门槛

所有 control 必须使用同一个 `MODEL_DIR`、GPU 组合、`MODE`、端口、请求形状、
tokenizer 和采样设置。

1. Print-config 门槛：

```bash
MODEL_DIR="$MODEL_DIR" \
PROFILE=qwen27b/experimental/fp8/hybrid-fp8kv-65K-mtp3-text-only.env \
MODE=fast \
./launcher.sh --print-config
```

输出必须包含：

- `KV precision: fp8`
- `KV fp16 skip layers: 3,11,19,27,35,39,47,55,63`
- `MAX_MODEL_LEN=66048`
- `MAX_BATCHED_TOKENS=2560`
- `MTP_K=3`

2. 吞吐门槛：

先跑 fp16 control，再跑 hybrid 候选。条件允许时，至少保留一次 warmup 和多次
measured run。

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

通过条件：

```text
hybrid_decode_tok_s >= fp16_decode_tok_s * 0.70
```

历史 FP8 MTP3 fp16 KV 在 PP65536/TG512 的参考值是 `70.8 tok/s`，所以参考目标约为
`49.6 tok/s`。这个历史数字只能做 sanity reference；最终通过/失败必须使用同一次
测试里的 fp16 control。

3. 质量门槛：

- 中文质量 smoke 要同时跑 fp16、全 compact-KV control、hybrid。
- NIAH 要在 fp16、全 compact-KV control、hybrid 上跑相同点位。先用已知敏感的
  长上下文中间深度点位，确认后再扩成完整 heatmap。
- 判定 KV 量化问题前，必须确认生成样本里确实包含 needle 文本。

只有 fp16 通过、全 compact-KV control 没暴露 eval/prompt 问题，并且 hybrid 没有
实质性 smoke/NIAH 损失时，才能晋升这条路线。
