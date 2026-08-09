# flashinfer 0.6.16rc4 补丁（vLLM 2080 Ti Definitive Edition）

本目录包含 `vLLM 2080 Ti Definitive Edition` 在 flashinfer **0.6.16rc4**（源码编译，
nvcc 12.8 / SM75）之上的定制补丁。相比 `patches/flashinfer-0.6.8.post1/`（预编译
wheel + 不可改的 JIT 内核），0.6.16 以源码形式安装，所有 kernel 头文件与 Python
模块均可直接修改，因此修复了 0.6.8 版本无法落地的 partition-kv 多 query 问题。

## 背景

vLLM fork 的 SM75 int8 per-token-head KV 解码依赖 flashinfer 的 CUDA-core
`batch_decode` 内核，并通过自定义 `Int8TokenHeadScaleAttention` variant
（vLLM 侧 `_int8kv_decode_jit_args` 注入）应用逐 (token, head) 的 k/v scale。
flashinfer 0.6.16rc4 相比 0.6.8 做了大量重构：

- 删除了 int8 KV 的 `dtype_map_kv` 条目与 `REGISTER_VALUE_TRANSFORM` /
  `REGISTER_PROBABILITY_TRANSFORM` 宏（0.6.8 有）；
- `DISPATCH_GQA_GROUP_SIZE` 不再支持 group_size=6（12 q heads / 2 kv heads）；
- vec_dtypes 无 int8 特化；
- CUDA-core decode 内核仍为单 query（`q_vec` 单值），无多 query 支持；
- `jit/attention/modules.py` 的 `gen_customize_batch_decode_module` 签名与
  0.6.8 的 `gen_batch_decode_module` 不同（13 个位置参数直接对应 vLLM 的 jit_args）；
- 离线安装（无 PyPI 网络）时 cutlass-dsl 缺失，顶层 import 会因旧版 cutlass
  缺少 `nvgpu.OperandMajorMode` 抛 AttributeError；
- `flashinfer/comm/fd_exchange.py` 使用 Python 3.13 语法 `array.array[int]`，
  Python 3.11 下 import 抛 TypeError。

## 补丁内容

| 文件 | 改动 |
| --- | --- |
| `flashinfer/__init__.py` | `fused_moe`/`gemm` 顶层 import 用 `suppress(ImportError, AttributeError)` 包裹（离线无 cutlass-dsl 时跳过，attention/decode 路径不需要） |
| `flashinfer/cute_dsl/utils.py` | `is_cute_dsl_available()` 增加 `nvgpu.OperandMajorMode` API 探测，旧版 cutlass 不再误判可用 |
| `flashinfer/jit/utils.py` | `dtype_map_kv` 加回 `torch.int8 -> "int8_t"` 条目 |
| `flashinfer/comm/fd_exchange.py` | 顶部加 `from __future__ import annotations`（py3.11 无 `array.array[int]` 下标语法） |
| `include/flashinfer/utils.cuh` | `DISPATCH_GQA_GROUP_SIZE` 加 `group_size == 6` 分支 |
| `include/flashinfer/vec_dtypes.cuh` | 移植 0.6.8 的 int8 特化：`vec_t<int8_t, vec_size>` 偏特化 + `vec_cast<float,int8_t>` / `vec_cast<int8_t,float>` / `vec_cast<half,int8_t>` / `vec_cast<int8_t,half>` |
| `include/flashinfer/attention/variant_helper.cuh` | 加 `REGISTER_VALUE_TRANSFORM` / `REGISTER_PROBABILITY_TRANSFORM` 宏（vLLM variant 需要） |
| `include/flashinfer/attention/decode.cuh` | ① `update_local_state` 参数化（+params/variant/kv_idx_base/kv_head_idx/qo_head_idx/batch_idx）并在 v 累加处调用 `variant.ValueTransform`（per-token v_scale）；② QO_LEN=4 多 query 内核（q_vec[4]/st[4]/s[4] + qo 循环 + 偏移公式，q_vec 循环内加载减 register）；③ partition-kv 修复：kernel 输出 `out_pos = partition ? qo_idx*num_chunks_k+bx : bx*QO_LEN+qo_idx`、tmp_lse 偏移 ×QO_LEN、`MergeStates(..., seq_len=QO_LEN)`；④ `last_indptr = paged_kv.indptr[batch_idx+1]`（batch_size 承载 q_len_per_req 而非真实 batch）；⑤ 不 partition 时 `kv_chunk_size` 回退 `kv_len`（workspace 指针未被写入） |
| `include/flashinfer/attention/scheduler.cuh` | ① `DecodeSplitKVIndptr` 按 q_len_per_req 展开行（request_indices/o_indptr 每 qo 行）；② `request_indices`/`kv_tile_indices`/`o_indptr` 的 int workspace 按 vector 实际大小分配（`std::max(padded_batch_size, vec.size())`）；③ `real_batch_size` 声明移出 if 块（作用域修复）+ fallback 推断加"indptr 真共享"条件（`indptr_h[batch_size] == indptr_h[1]`） |

## 应用

```bash
bash patches/flashinfer-0.6.16rc4/apply.sh .venv/lib/python3.11/site-packages
```

补丁基于源码编译安装（`pip install . --no-deps --no-build-isolation`，
`BUILD_NIXL_EP=0`，nvcc 12.8）。JIT 内核在首次调用时由 flashinfer 用
`~/.cache/flashinfer/0.6.16rc4/75/` 下的 nvcc 编译，改内核头文件后需清缓存：
`rm -rf ~/.cache/flashinfer/0.6.16rc4/75/generated/vllm_int8kv*`。

## 验证（warm，Qwen3.6-27B-FP8，TP2 双 2080 Ti，int8_per_token_head，MTP3，
vLLM completions 口径）

| 场景 | 125K completion | decode tok/s |
| --- | --- | --- |
| 桥（vLLM 原生 int8 decode） | 128 | 32.09 |
| 0.6.8 batch=4 拆解（前基线） | 128 | 25.07 |
| **0.6.16 kernel 内多 query（本补丁）** | **128** | **31.93** |

全谱（warm completions）：4K kernel 71.23 vs 桥 70.16（+1.5%）；60K 44.31 vs 44.23
（+0.2%）；100K 36.43 vs 36.65（−0.6%）；125K 31.93 vs 32.09（−0.5%）——kernel
与桥全谱持平，短上下文略胜。

调优尝试（2026-08-08）：① num_stages 强制 2-stage（DISPATCH_COMPUTE_CAP_DECODE_
NUM_STAGES_SMEM 阈值 8→7，SM75 用 2 级 cp_async 流水）→ 125K 31.73（无效果）；
② kv_chunk_size 固定 1024（并行度探针）→ 125K 31.71（无效果）。结论：125K 的
瓶颈是 QO_LEN kernel 的 register 压力（st[4]/s[4]/q_vec[4] 约 200+ 寄存器）导致
occupancy 仅约 1 block/SM（22 并发 block），partition 的 488 个 chunk 需 20+ 波次
串行——非 pipeline 预取或 chunk 并行度问题。进一步优化需降低 register 占用
（s 数组挪 smem / 精简 state_t），成本高、收益有限（上限 ~40 tok/s），当前维持
与桥持平的 31.93。

正确性：对比桥逐元素 diff（DEBUG 模式）定位到 0.6.8 的两个根因——plan tensor
竞态（vLLM 侧 `_int8kv_fa_decode_plan_tensors` 存 self 修复）与 partition-kv 的
tmp workspace 单 query 布局（0.6.8 预编译 .so 不可改；0.6.16 源码可改，本补丁
修复）。125K 内容正确（"the quick brown fox..." 循环文本），无 early-stop。

## 注意

- 应用本补丁后，`flashinfer/comm` 的 flashinfer all-reduce 需 vLLM 侧
  `flashinfer_all_reduce.py` 的 `except (ImportError, TypeError)` 兜底
  （0.6.16 的 comm 在 py3.11 下 import 崩），vLLM 集成改动见
  `vllm/distributed/device_communicators/flashinfer_all_reduce.py`。
- 性能未达理论 45 tok/s（22 ms/step）：125K partition_kv 场景下 QO_LEN 内核的
  register 压力（st[4]/s[4]）降低 occupancy，实测 ~125 ms/step；occupancy /
  kv_chunk_size 调优留作后续。
