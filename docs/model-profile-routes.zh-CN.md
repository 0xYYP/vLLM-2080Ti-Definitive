# 模型 Profile 路线

本文只列出已经在 stable runtime 路径下测试过的部署 profile。Speed 模式和量化
KV profile 暂不写入这里，等重新测试后再补。

`Profile` 是 `start.sh` 选择的 `.env` 文件；具体 checkpoint 仍然通过
`MODEL_DIR` 单独选择。

证据规则：只有真实大 prompt 请求返回 HTTP 200、stream 正常结束、并且至少生成
1 个 completion token 后，profile 才会进入本文档。只 load 成功、READY、
health 通过、小窗口 smoke、空 stream，都不算容量证据。

## Qwen3.6 27B - Stable FP16 KV

| Profile | 权重精度 | KV | Context | GPU util | Batch tokens | Seqs | MTP | Message | Note |
|---|---|---|---:|---:|---:|---:|---|---|---|
| stable-qwen27-fp8-fp16kv-mtp3-100k.env | FP8 | FP16/default | 100K | 0.92 | 2048 | 1 | MTP3 | text | Stable FP8 质量路线；100K 真实大 prompt 严格通过。 |
| stable-qwen27-int4-fp16kv-mtp3-256k.env | INT4 | FP16/default | 256K | 0.90 | 2048 | 1 | MTP3 | text | Stable INT4 满血上下文路线；原生 256K 真实大 prompt 严格通过。 |

## 说明

- Active stable profile 直接放在 `profiles/` 顶层。
- 实验片段放在 `profiles/experimental/`，不属于本文的 stable 矩阵。
- Speed 模式、INT8 KV、TurboQuant KV、YaRN、多模态、工作区 profile 都需要重新
  验证后，才能进入本文档。
- 吞吐背景记录见
  [Qwen3.6 KV 吞吐 Sweep](qwen36-kv-throughput-sweep.zh-CN.md) 和
  [MTP 任务敏感性](mtp-task-sensitivity.md)。
