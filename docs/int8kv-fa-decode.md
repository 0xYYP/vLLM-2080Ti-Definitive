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
4. `attention/prefill.cuh`（direct_paged 的 V scale，2026-08-07 新增）：
   `compute_sfm_v` 增加 `Params`/`variant`/`batch_idx`/`kv_idx_base`/
   `kv_head_idx` 参数，在 int8→f16 的 B fragment（mma B，col-major，
   `k = 2*(tid/4)+{0,1}`）cast 后应用 per-token-head `v_scale`（`half2`
   乘法，4 组覆盖 m16n16k16 的两个 m16n8k16 半边）。用自包含
   `vllm_has_v_scale_member<Params>`（SFINAE 检测 `maybe_v_scale_cache`
   成员，不依赖 flashinfer 的 `has_*_v` 宏，标准 fa2 JIT 未定义该宏）。
   三个调用点（single/ragged/paged）同步更新；ragged 桥路径 dequant 后
   KV 为 fp16、Params 无 scale 成员，走 `if constexpr` 空分支不受影响。
5. `attention/decode.cuh`（batch>1 的 scale 索引，2026-08-07 新增）：
   `update_local_state` 增加 `batch_idx = 0u` 默认参数，`ValueTransform`
   调用从硬编码 `0u` 改为 `batch_idx`；batch kernel 调用点传 `batch_idx`。
   否则 batch>1 时 `physical_scale_index` 用 `indptr[0]` 计算 page，会读到
   request 0 的 scale 区域（verify batch=4 场景 indices 共享时数据恰好
   相同，但 page 偏移仍会越界读 indices，触发 misaligned/错数据）。

以上补丁已随本仓库分发于 `patches/flashinfer-0.6.8.post1/`（修改后的完整文件 + apply.sh）；升级/重装 FlashInfer 后运行 `bash patches/flashinfer-0.6.8.post1/apply.sh <site-packages>` 重新应用（会校验版本并备份原文件）。

## 启用条件

- 环境变量：`VLLM_INT8KV_FA_DECODE=1`、
  `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864`（FlashInfer workspace，
  prefill 临时 buffer 需 ≥33.8MiB）。`VLLM_INT8KV_ALIGNED_HEAD_STRIDE=1`
  自 2026-08-07 起不再是必需：`kv_cache_interface.py` 的
  `get_kv_cache_shape` 与 `KVCacheSpec.page_size_bytes` 已对非 ALIGNED
  模式也做 head stride 16B 对齐（`round_up(head_size+4, 16)` = 272，
  与 ALIGNED 模式一致），否则 FlashInfer 128-bit KV load 在
  head 偏移 260×2B=520B 处 misaligned address（batch=1 的 decode 也可能
  触发，batch>1 的 verify 必然触发）。272 对齐使每页 KV 增大约
  178KiB（block_size=1584），128K 上下文约多 60MiB 需求。
- 模型 guard：无 sliding-window、无 logits-soft-cap、`head_size % 16 == 0`
  （`_try_int8kv_fa_decode` 在入口拒绝，避免静默错误输出）。
- 回退：任何异常（缺补丁、对齐失败、workspace 不足）都回退原生路径并
  记录一次 `fa_decode_failed` 日志，不改变正确性边界。

## MTP verify 多 query 支持（2026-08-07 实验记录）

- **batch=4 拆解（已验证正确，性能不达标）**：`_try_int8kv_fa_decode` 对
  `num_actual_tokens<=4` 的 MTP verify 建模为共享同一 KV 前缀的 batch=4
  请求（`paged_kv_indptr=[0,nb,2nb,3nb,4nb]`、indices 按 request 重复以覆盖
  `indptr[-1]`、last_page_len 数组）。flashinfer decode wrapper 原生支持
  batch>1；decode.cuh 的 `update_local_state` 需传真实 `batch_idx`
  （否则 `physical_scale_index` 用 `indptr[0]` 计算 page 越界，见第 5 条）。
  实测（warm，272 布局）：4K 52.76 tok/s、125K 25.07 tok/s——125K 比桥
  （32.09）慢 28%，因为每 request 独立读全部 KV（4× 带宽），未达理论
  带宽（~10ms）。
- **kernel 内多 query（QO_LEN=4，正确性未达标，已回退）**：decode.cuh
  batch kernel 改 q_vec[4]/st[4]/s[4] 数组 + qo 循环（共享 KV tile 加载，
  1× 带宽），配合 `q_len_per_req=4`。printf 实证 q/o 数值合理（非垃圾），
  但 MTP verify 仍早停（125K completion=1）——疑似 o 部分 head 或
  softmax 边界问题，未定位。已回退到 batch=4 拆解（decode.cuh 恢复
  原版单 query kernel + batch_idx patch）。
- **真正提速路径**：kernel 内多 query 的共享 KV flash-decoding（1× 带宽），
  正确性修复后预期 10-20ms/step 量级。调试线索：`FA decode out row`
  逐行统计正常（std 0.24-0.46）、kernel printf q/o 合理——需对比桥输出
  逐元素 diff 定位错位。

## 显存注意

双 2080 Ti（21.49GiB）下，272 对齐使 245K 上下文的 KV pool 需求约
4.71GiB，运行余量约 4.4-4.5GiB：`MAX_MODEL_LEN=262144` 时 pp100K 一次
写入即 OOM（50MiB），极限可用上下文为 250880（245K，见
`profiles/qwen27b/normal/fp8/int8kv-245K-mtp3-text-only.env`）。
