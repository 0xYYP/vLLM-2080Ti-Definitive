# flashinfer 补丁（vLLM 2080 Ti Definitive Edition）

本目录保存 SM75 int8-per-token-head KV 所需的 FlashInfer 源码补丁（修改后的
完整文件 + 应用脚本），消除"升级/重装 FlashInfer 后静默回退原生路径"的
复现性风险。

## 内容

- `decode.cuh` / `prefill.cuh` / `variant_helper.cuh` / `vec_dtypes.cuh`：
  flashinfer 0.6.8.post1 修改后的完整文件（前 3 个对应
  `data/include/flashinfer/attention/`，`vec_dtypes.cuh` 在
  `data/include/flashinfer/` 根目录；apply.sh 按目标路径覆盖）。
- `apply.sh`：校验 flashinfer 版本为 0.6.8.post1 后覆盖到
  `flashinfer/data/include/flashinfer/`，原文件备份为 `.bak-<日期>`。

## 用法

```bash
bash patches/flashinfer-0.6.8.post1/apply.sh <site-packages 目录>
# 例：
bash patches/flashinfer-0.6.8.post1/apply.sh .venv/lib/python3.11/site-packages
```

## 补丁内容（相对 flashinfer 0.6.8.post1 官方源码）

1. `vec_dtypes.cuh`：文件头补 `#pragma once`（原文件无 include guard）；
   为 `vec_t<int8_t, N>` 增加 int8 特化（`vec_cast` 等），供 128-bit KV load
   与 mma 使用。
2. `variant_helper.cuh`：`REGISTER_LOGITS_TRANSFORM` / `REGISTER_VALUE_TRANSFORM`
   等宏（`__VA_ARGS__` 内联变换体），供 JIT 生成的 variant 声明复用。
3. `decode.cuh`：`BatchDecodeWithPagedKVCacheWrapper` 模板实例化 variant 时
   调用 `variant.ValueTransform` / `variant.ProbabilityTransform` 应用
   per-token-head scale；`update_local_state` 增加 `batch_idx` 参数
   （batch>1 时 `physical_scale_index` 必须用真实 request 索引，否则
   scale 越界——见 docs/int8kv-fa-decode.md 第 5 条）。
4. `prefill.cuh`：`compute_sfm_v` 增加 `Params`/`variant`/`batch_idx`/
   `kv_idx_base`/`kv_head_idx` 参数，在 int8→f16 的 mma B fragment
   cast 后应用 per-token-head `v_scale`（自包含 `vllm_has_v_scale_member`
   SFINAE；single/ragged/paged 三个调用点同步）——见
   docs/int8kv-fa-decode.md 第 4 条。

## 升级注意

- 升级 flashinfer 前先还原 `.bak-<日期>`（或记录差异）；新版本补丁需
  重新适配（上游 kernel 结构可能变化）。
- 补丁未应用时，`VLLM_INT8KV_FA_DECODE=1` / `VLLM_INT8KV_FA_DIRECT_PAGED=1`
  的调用会以 `fa_decode_failed` / `direct_paged_failed` 记录并静默回退
  原生路径——性能退化为原生 O(KV) 扫描，正确性不受影响。
