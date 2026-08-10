# Profile 导引

语言：[English](README.md) | 简体中文

这里是 vLLM 2080Ti Definitive 自带的启动 profile。一个 profile 只是运行参数
的 `.env` 预设，不包含模型权重路径；权重目录通过 `launcher.sh` 或
`MODEL_DIR=...` 单独选择。

这里列出的上下文容量和吞吐数据，验证硬件是双 RTX 2080 Ti 22GB，tensor
parallel size 2。

目录结构：

```text
profiles/
  templates/
  qwen27b/
    normal/
      fp8/
      int4/
    fast/
      fp8/
      int4/
    user/
  qwen35b/
    normal/
      fp8/
    aggressive/
      fp8/
    fast/
      fp8/
    user/
```

启动模式：

- `safe`：保守回退模式，优先保证可用性。
- `normal`：推荐日常模式，适合稳定部署。
- `fast`：高性能模式，适合追求更高吞吐的场景。
- `aggressive`：更加激进的模式，性能与质量风险最高。

`profiles/templates/` 存放可选 chat template 预设。它们通过 launcher 作为全局
服务设置选择；具体 route profile 不保存 chat template、GPU、端口、reasoning
默认值或工具调用默认值。
对内置 Qwen3/Qwen3.6 路线，launcher 会在未显式设置时补上 `qwen3`
reasoning parser，让启动 smoke 和聊天解析都跟模型默认 reasoning 行为保持一致。
如需诊断无 reasoning parser 路径，可设置 `REASONING_PARSER=off`。

文件名描述路线：

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV 精度定位：

- `fp16kv`：质量路线。
- `int8kv`：容量 / 平衡路线；当前只作为 `normal` profile 保留。
- `tqk8v4`：TurboQuant K8V4 压缩路线；当前只保留质量通过的 `fast` profile。
- 官方 Qwen3.6 35B 当前提供 FP8 权重 + FP16 KV 的纯文本和图文预设。

内置 TQK8V4 profile 使用 `MAX_BATCHED_TOKENS=2560`，这是 Qwen hybrid cache
block 对齐后的 prefix-cache 路径已验证设置。

## 已验证 Profile

### Qwen3.6 27B FP8

测试权重：Jackrong/Qwopus3.6-27B-v2-FP8，约 29G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/normal/fp8/int8kv-245K-mtp3-text-only.env` | normal | 245K | INT8 | 3 | text-only | 1 | 710.0 / 6.3 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

`int8kv-245K` 行的口径不同：prefill 710.0 tok/s 是 pp100K 首次写入实测
（138456 tokens，约 195s），decode 6.3 tok/s 是 250K 上下文 FA-decode 实测
（64 tokens，prefix-cache 命中、无 debug 日志；0.6.8 时代、实验性 decode
variant）。0.6.16rc4 下默认桥路径在 128K profile 实测 4K 29.56 tok/s
（warm / completions / MTP3）。同环境 A/B（vLLM 固定 6426afb，仅切换
flashinfer）确认 0.6.8.post1 与 0.6.16rc4 差异 <1.5%（4K 29.80 vs 29.56），
2026-08-07 的更高桥记录（4K 70.16）不可复现，归因当时测量条件而非
flashinfer 版本。245K profile 预留 5.9GiB KV 池，每卡仅剩
~0.9GiB 余量，长上下文首次写入（prefill）即 OOM，只能 prefix 命中后做
decode；262144 在单次 100K 写入即 OOM。245K 为配置上限，仅
prefix-cache 命中后 decode 验证过（冷启动 60K+ prefill 未验证且已观测
OOM）。decode 加速细节与限制见
[`docs/int8kv-fa-decode.md`](../docs/int8kv-fa-decode.md)。

### Qwen3.6 35B FP8

测试目标权重：Qwen/Qwen3.6-35B-A3B-FP8，约 36G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/normal/fp8/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | 6705.13 / 97.33 |
| `qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 1 | 5485.13 / 95.20 |
| `qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` | aggressive | 256K | FP16 | 0 | text-only | 1 | 6843.01 / 124.01 |
| `qwen35b/aggressive/fp8/fp16kv-136K-nomtp-text-image.env` | aggressive | 136K | FP16 | 0 | text+image | 1 | 5422.83 / 124.11 |
| `qwen35b/fast/fp8/fp16kv-178K-mtp3-text-only.env` | fast | 178K | FP16 | 3 | text-only | 1 | 5889.20 / 195.95 |

### Qwen3.6 27B AWQ/GPTQ-INT4

测试权重：QuantTrio/Qwen3.6-27B-AWQ、mconcat/Qwopus3.6-27B-v2-AWQ-4bit，以及
llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4，
约 19G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 每工作区 250K | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 每工作区 250K | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |
