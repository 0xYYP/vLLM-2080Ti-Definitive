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

另有一个 65K all-INT8 诊断 control：

```text
profiles/qwen27b/experimental/fp8/int8kv-65K-mtp3-text-only.env
```

它用于在同样的请求级上下文上限下对照 hybrid 候选。它等价于 normal all-INT8 容量
路线只把 `MAX_MODEL_LEN` 收紧到 `66048`。内置 252K all-INT8 路线是容量路线，长
decode 可能触发不同路径，因此不应该作为 65K 平衡门槛的唯一速度 control。

如果 normal all-INT8 control 出现长 decode 崩塌，先用下面的诊断矩阵拆变量，再考虑
更底层的 kernel 重写：

```text
profiles/qwen27b/experimental/fp8/int8kv-65K-mtp3-text-only.env
profiles/qwen27b/experimental/fp8/int8kv-65K-fastgraph-mtp3-text-only.env
profiles/qwen27b/experimental/fp8/int8kv-65K-fast2560-mtp3-text-only.env
profiles/qwen27b/experimental/fp8/int8kv-65K-fastaligned-mtp3-text-only.env
profiles/qwen27b/experimental/fp8/int8kv-65K-fastaligned3d-mtp3-text-only.env
```

这些 profile 分别隔离 normal 路径、fast graph policy、更大的
`MAX_BATCHED_TOKENS`、aligned int8 head stride，以及显式打开的 per-token-head
3D decode 路径。它们只是诊断 profile，不是已晋升的部署预设。3D 路径只通过
`VLLM_INT8KV_ENABLE_3D_DECODE=1` 打开；只有服务器 sweep 同时证明吞吐提升且质量
无回归后，才考虑晋升。

新增 3D decode 诊断档的原因是：历史 FP8 MTP3 INT8 KV 在 PP65536/TG512 下的参考值
是 `1227.9 / 42.8 tok/s`，而后续加上的 per-token-head guard 会把它强制压到 2D
decode，可能导致长上下文吞吐崩塌。

服务器上可以用下面命令一次跑完整矩阵：

```bash
MODEL_DIR=/data/models/Qwen3.6-27B-FP8 \
ALLOW_STOP_EXISTING=1 \
tools/int8kv_65k_diag_sweep.sh
```

脚本会把 JSONL 记录和中位数摘要写到 `/tmp/int8kv-65k-diag-*`。

也可以用 `PROFILE_LIST` 跑聚焦子集，避免一个中间诊断档失败时挡住关键对照：

```bash
MODEL_DIR=/data/models/Qwen3.6-27B-FP8 \
RUNS=1 \
ALLOW_STOP_EXISTING=1 \
PROFILE_LIST="qwen27b/experimental/fp8/int8kv-65K-mtp3-text-only.env qwen27b/experimental/fp8/int8kv-65K-fastaligned3d-mtp3-text-only.env" \
tools/int8kv_65k_diag_sweep.sh
```

### 2026-07-06 聚焦 sweep 阶段证据

服务器：dual RTX 2080 Ti，`/data/models/Qwen3.6-27B-FP8`，分支提交
`4785621`，`RUNS=1`，`--endpoint completions --ignore-eos --pure-filler`。
结果文件：

- `/tmp/int8kv-65k-diag-20260706-232611/profile.jsonl`
- `/tmp/int8kv-65k-diag-20260706-232611/summary.tsv`
- `/tmp/int8kv-65k-diag-20260706-234240/profile.jsonl`
- `/tmp/int8kv-65k-diag-20260706-234240/summary.tsv`

| Profile | 关键变量 | PP4096/TG128 prefill / decode | PP65536/TG512 prefill / decode | `decode_long` 路径 |
| --- | --- | ---: | ---: | --- |
| `int8kv-65K` | normal, MBT2048, aligned=0, 3D=0 | `6133.74 / 43.57` | `24548.46 / 5.25` | `use_3d=False`, `seq_threshold_3d=64` |
| `int8kv-65K-fastgraph` | fast, MBT2048, aligned=0, 3D=0 | `1565.75 / 74.36` | `24281.64 / 24.31` | `use_3d=False`, `seq_threshold_3d=4` |
| `int8kv-65K-fast2560` | fast, MBT2560, aligned=0, 3D=0 | `1582.42 / 73.61` | `25773.52 / 24.43` | `use_3d=False`, `seq_threshold_3d=4` |
| `int8kv-65K-fastaligned3d` | fast, MBT2560, aligned=1, 3D=1 | `1536.90 / 81.19` | `24355.03 / 45.05` | `use_3d=True`, `seq_threshold_3d=4` |

解读：

- normal all-INT8 的 PP65536/TG512 decode 复现为 `5.25 tok/s`，`decode_long`
  明确走 2D。
- fast graph / FlashInfer prefill 相关策略把 65K decode 拉到约 `24.3 tok/s`；
  `MAX_BATCHED_TOKENS=2048 -> 2560` 对 decode 几乎没有增益。
- `fastaligned3d` 把 65K decode 恢复到 `45.05 tok/s`，已经回到历史 INT8
  参考 `42.8 tok/s` 区间；但它仍是诊断 profile，晋升前必须补质量验证。

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
