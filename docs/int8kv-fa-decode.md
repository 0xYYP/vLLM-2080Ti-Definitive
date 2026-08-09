# INT8 KV FA decode（SM75 FlashInfer decode variant）

本文档记录 `VLLM_INT8KV_FA_DECODE=1` 依赖的 FlashInfer 补丁与启用条件，
供复现 2080 Ti / SM75 上 int8-per-token-head KV 的长上下文 decode 加速。

## 背景

`int8_per_token_head` 的 KV cache 带 per-token/per-head fp32 scale，FlashInfer
的标准 `batch_decode` kernel 不认识这种布局：kernel 按固定 stride 加载 KV
且没有 scale 变换钩子，因此默认 decode 会回退到 vLLM 原生 unified_attention
的 O(KV) 全量扫描（长上下文 decode 线性退化，pp4K→65K 为 44→5 tok/s）。

本实现为 FlashInfer 的 decode kernel 增加一个 JIT 编译的
`Int8TokenHeadScaleAttention` variant，在 kernel 内部应用 per-token-head
scale（ValueTransform / ProbabilityTransform），使 decode 走与 fp16 相同的
flash-decoding 分块并行路径。实测 250K 上下文 decode 6.3 tok/s（原生外推
~1.5-2 tok/s，约 3-4 倍），KV 长度 21.8K→250K（11.5×）时 decode 仅减速
2.8×（原生为线性 8.8×）。

## FlashInfer 补丁（必须）

路径：`.venv/lib/python3.11/site-packages/flashinfer/data/include/flashinfer/`

1. `attention/variant_helper.cuh`：新增 `REGISTER_VALUE_TRANSFORM` 与
   `REGISTER_PROBABILITY_TRANSFORM` 宏，定义 `ValueTransform` /
   `ProbabilityTransform` 钩子（模板函数，`__VA_ARGS__` 内联变换体）。
2. `attention/decode.cuh`：在 `BatchDecodeWithPagedKVCacheWrapper` 模板
   中实例化 variant 时调用 `variant.ValueTransform` / `variant.ProbabilityTransform`
   对 int8 KV 与 attn 概率应用 per-token-head scale。
3. `compute/vec_dtypes.cuh`：为 `vec_t<int8_t, N>` 与 `vec_cast<int8_t, ...>`
   增加特化（文件无 include guard，需在文件头补 `#pragma once` 避免重复
   定义）。kv_cache head_dim 必须 16 对齐（见下），否则 128-bit vector load
   在 head 偏移处 misaligned address。

以上补丁为实验性、未随本仓库分发；升级/重装 FlashInfer 后需重新应用。

## 启用条件

- 环境变量：`VLLM_INT8KV_FA_DECODE=1`、
  `VLLM_INT8KV_ALIGNED_HEAD_STRIDE=1`（kv_cache head padding 16 对齐，
  head_dim 从 `head_size+4` 提升到 `round_up(head_size+4, 16)`）、
  `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864`（FlashInfer workspace，
  prefill 临时 buffer 需 ≥33.8MiB）。
- 模型 guard：无 sliding-window、无 logits-soft-cap、`head_size % 16 == 0`
  （`_try_int8kv_fa_decode` 在入口拒绝，避免静默错误输出）。
- 回退：任何异常（缺补丁、对齐失败、workspace 不足）都回退原生路径并
  记录一次 `fa_decode_failed` 日志，不改变正确性边界。

## 显存注意

双 2080 Ti（21.49GiB）下，272 对齐使 245K 上下文的 KV pool 需求约
4.71GiB，运行余量约 4.4-4.5GiB：`MAX_MODEL_LEN=262144` 时 pp100K 一次
写入即 OOM（50MiB），极限可用上下文为 250880（245K，见
`profiles/qwen27b/normal/fp8/int8kv-245K-mtp3-text-only.env`）。
