# Profile 验证蓝图与结果

本文只保留当前 clean 验证后的 profile 结果。正式 profile 清单与用户说明以
[Profile 导引](../profiles/README.zh-CN.md) 为准。

## 判定标准

一个 profile 只有同时满足以下三项，才算成立：

- 启动 / admission 通过目标上下文。
- 中文质量 smoke 通过，没有重复、残缺、乱码或明显答非所问。
- `4096/128 ignore_eos` 合成吞吐有有效 `prefill / decode` 数字。

## 已成立 Profile

| Profile | 状态 | 关键参数 | PP4096/TG128 | 结论 |
|---|---|---|---:|---|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | 通过 | FP8；normal；FP16 KV；128K；MTP3 | 1619.48 / 84.71 | FP8 质量文本路线；128K 已重新验证通过。 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | 通过 | FP8；normal；INT8 KV；252K；MTP3 | 1605.10 / 44.09 | 长上下文 fallback，decode 慢。 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | 通过 | FP8；fast；FP16 KV；112K；MTP3 | 1615.58 / 83.69 | FP8 fast 质量文本路线；120K admission 失败，128K 吞吐稳定性未通过。 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | 通过 | FP8；fast；TQK8V4；256K；MTP3 | 1615.81 / 81.06 | GPU util 0.94 下已验证 256K。 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | 通过 | FP8；fast；TQK8V4；240K；MTP3；图文 | 1605.61 / 80.67 | GPU util 0.96 下已验证 240K。 |
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | 通过 | INT4；normal；FP16 KV；256K；MTP3 | 1738.06 / 97.79 | INT4 主力质量文本路线。 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | 通过 | INT4；normal；FP16 KV；240K；MTP3；图文 | 1760.14 / 94.48 | INT4 质量图文路线。 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | 通过 | INT4；fast；FP16 KV；256K；MTP3 | 1734.98 / 87.00 | GPU util 0.94 下已验证 256K。 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | 通过 | INT4；fast；TQK8V4；256K；MTP3 | 1744.67 / 100.81 | INT4 fast 主力压缩路线。 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | 通过 | INT4；fast；TQK8V4；双工作区 250K；MTP3 | 1739.23 / 99.91 | 当前最好的双工作区 fast 路线。 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | 通过 | INT4；normal；INT8 KV；双工作区 250K；MTP3 | 1740.51 / 49.06 | 可用 fallback，decode 慢。 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | 通过 | INT4；normal；INT8 KV；512K YaRN；MTP3 | 1734.14 / 48.16 | GPU util 0.94 下已验证 512K。 |
| `qwen35b/normal/fp8/fp16kv-256K-nomtp-text-only.env` | 通过 | 官方 FP8；normal；FP16 KV；256K；noMTP | 6705.13 / 97.33 | 当前正式 35B 容量文本路线。 |
| `qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env` | 通过 | 官方 FP8；normal；FP16 KV；136K；noMTP；图文 | 5485.13 / 95.20 | 当前正式 35B 图文路线。 |
| `qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` | 通过 | 官方 FP8；aggressive；FP16 KV；256K；noMTP | 6843.01 / 124.01 | 通过 launcher 预热回冲后可稳定起服；当前 35B aggressive 文本容量路线。 |
| `qwen35b/aggressive/fp8/fp16kv-136K-nomtp-text-image.env` | 通过 | 官方 FP8；aggressive；FP16 KV；136K；noMTP；图文 | 5422.83 / 124.11 | 当前 35B aggressive 图文路线；短测 decode 明显高于 normal 图文。 |
| `qwen35b/fast/fp8/fp16kv-178K-mtp3-text-only.env` | 通过 | 官方 FP8；fast；FP16 KV；178K；MTP3 | 5889.20 / 195.95 | 当前正式 35B fast 文本路线。 |

## 不保留路线

| Candidate | 结果 | 原因 |
|---|---|---|
| FP8 + FP16KV fast 120K/128K | 不保留 | 120K admission 失败；128K 可启动并通过短质量 smoke，但 PP4096/TG128 吞吐触发 EngineCore 500；正式 fast 回退为 112K。 |
| FP8 + TQK8V4 256K text-image | 暂不保留 | 当前图文已验证上限为 240K；256K 图文未晋升。 |
| fast + INT8KV | 不保留 | 容量或合成速度可过，但中文质量 smoke 失败。 |
| TQ4NC 正式 profile | 不保留 | 当前正式路线优先 TQK8V4；TQ4NC 只保留为历史实验结论。 |
