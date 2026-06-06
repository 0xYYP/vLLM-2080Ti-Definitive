# 模型 Profile 路线

本文记录双 RTX 2080 Ti / SM75 runtime 的用户侧 Profile 矩阵。`Profile`
就是 `start.sh` 选择的 `.env` 文件；具体 checkpoint 仍然通过 `MODEL_DIR`
单独选择。

Active profile 矩阵采用证据门槛：上下文大小只有在真实大 prompt 请求返回 HTTP 200、
stream 正常结束、并且至少生成 1 个 completion token 后，才会进入 active
profile。只 load 成功、READY、health 通过、小窗口 smoke、空 stream，都不算容量
证据。

Profile 名称直接编码 launcher 模式：

- `stable-*`：稳定模式，只允许 FP16/default KV。
- `speed-*`：高性能模式，用于量化 KV、容量路线和性能路线。后续 profile
  验证默认按这个模式推进。

Active deployment profile 直接放在 `profiles/` 顶层。实验 profile 放在
`profiles/experimental/`，这样它仍然在同一个 profile 树里，但不会进入 launcher
默认菜单。

表格字段统一为：

`Profile | Mode | 权重精度 | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note`

`256K` 这类标签是便于阅读的档位；精确 token 数写在 profile 文件里。

吞吐细节见
[Qwen3.6 KV 吞吐 Sweep](qwen36-kv-throughput-sweep.zh-CN.md) 和
[MTP 任务敏感性](mtp-task-sensitivity.md)。

## Qwen3.6 27B - FP8

| Profile | Mode | 权重精度 | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| stable-qwen27-fp8-fp16kv-mtp3-100k.env | stable | FP8 | FP16/default | 100K | 0.92 | 2048 | 1 | MTP3 | text | 100K 真实大 prompt 严格通过；默认 FP8 质量/稳定路线。 |
| speed-qwen27-fp8-int8kv-mtp3-240k.env | speed | FP8 | INT8 | 240K | 0.95 | 2048 | 1 | MTP3 | text | 240K 真实大 prompt 严格通过。旧 256K 行返回空 stream，不再 active。 |
| speed-qwen27-fp8-int8kv-yarn-mtp3-216k.env | speed | FP8 | INT8 + YaRN | 216K | 0.94 | 2048 | 1 | MTP3 | text | 216K 真实大 prompt 严格通过。224K 和 512K admission 失败，因此不是 512K 路线。 |

## Qwen3.6 27B - INT4

FP8 和 INT4 checkpoint 在这个 runtime 中通过 Marlin weight path 承接。AWQ、
GPTQ，以及后续兼容的 INT4 类格式，都是 checkpoint 封装选择，不拆成独立
Profile 家族。

| Profile | Mode | 权重精度 | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| stable-qwen27-int4-fp16kv-mtp3-256k.env | stable | INT4 | FP16/default | 256K | 0.90 | 2048 | 1 | MTP3 | text | 原生满血上下文真实大 prompt 严格通过；INT4 主力满血上下文路线。 |
| speed-qwen27-int4-int8kv-workspace2-mtp3.env | speed | INT4 | INT8 | 128K x 2 | 0.90 | 2048 | 2 | MTP3 | text | 双 seq 下 128K 真实大 prompt 严格通过；这是工作区隔离，不是并行长 prefill 吞吐。 |

## 已降级路线

以下路线不进入 active deployment profiles：

| 路线 | 状态 | 原因 |
|---|---|---|
| FP8 INT8 256K | 已降级 | 256K 可以启动但返回空 stream；严格通过值是 240K。 |
| FP8 INT8 + YaRN 512K | 已降级 | 512K KV admission 失败；当前严格通过值是 216K。 |
| FP8 TQK8V4 256K | 实验路线 | speed 模式下 256K 和 248K 启动失败；240K 进入请求后变成 GPU idle / CPU-bound，不算严格通过。 |
| Qwen INT4 TurboQuant workspace4 | 已降级 | 从 128K 降到 64K 都是空 stream，未通过严格请求验证。 |
| Qwen INT4 / FP8 多模态 TurboQuant | 已降级 | 当前没有严格 image 路线；之前长 prompt 探针失败或证据不够严格。 |
| Qwen INT4 TurboQuant + YaRN 1M | 已降级 | 没有真实大 prompt 严格通过，不是 active 容量 profile。 |
| Gemma4 FP16/default KV | 已降级 | 100K 及更低大上下文启动探针失败；只保留短上下文速度证据。 |
| Gemma4 INT8 KV | 已降级 | 256K 降到 64K 都启动失败，触发 page-size unification 错误。 |
| Gemma4 TurboQuant KV | 已降级 | 修复后 43K 可以进到请求阶段，但真实请求没有生成 completion token。 |

## 当前结论

- Active deployment profiles 只保留上面 5 个文件。
- 实验 profile 片段保留在 `profiles/experimental/`；只有明确要测试实验路线时，
  才在 launcher 里把 Profile directory 指向这个子目录。
- Qwen 仍然是成熟路线。MTP3 保留在 active profile 中，因为它有明确 decode
  加速收益，并且这些 profile 形态已经有严格大 prompt 证据。
- Gemma 仍然是支持的测速/实验 checkpoint，但在真实大 prompt 请求严格通过前，
  不再提供 active 容量 profile。
- 压缩 KV 路线如果只是能 load、报告容量、或返回空 stream，只能算实验记录，
  不能算部署 profile。
