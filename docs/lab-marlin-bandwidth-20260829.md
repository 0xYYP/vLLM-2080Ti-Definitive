# 验证记录：Marlin int4 GEMM 带宽取证（2026-08-29）

## 1. 目标与背景

- 触发问题：用户问 "Marlin int4 GEMM 有解决办法吗？w8a8 / w4a8 会不会显著提速 decode？"
- 待验证假设：① 单流 decode 是权重带宽受限；② Marlin 在 SM75 上是否吃满显存带宽（此前粗略估算出现 "37 倍水分" 的存疑数字，需实测钉死）；③ w8a8/w4a8 是否有提速空间。
- 验证方式：从真实 AWQ checkpoint 加载一层权重，按 vLLM 生产的完整 transform 链（permute → gptq_marlin_repack → marlin_permute_scales → apply_gptq_marlin_linear）复现 decode 形状的 GEMM，用普通计时与 torch.profiler 双测量，扫描 M=1/2/4/8 与两种 N。

## 2. 环境

- cybros，2× RTX 2080 Ti（SM75），CUDA 12.8，torch 2.11.0+cu128
- 模型：`/data/models/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ`（AWQ int4，group=128，uint4 + zero-point，compressed-tensors）
- 复用 worktree `/tmp/dv`（feat/draft-vocab，含完整 vllm 源码）+ `/opt/vllm-2080ti-definitive/.venv`
- 取样的层：`model.language_model.layers.11.self_attn.q_proj`（N=12288）与 `model.language_model.layers.11.mlp.gate_proj`（N=17408）

## 3. 方法（复现脚本）

`/tmp/marlin_repro.py`（普通计时，M 可配置）与 `/tmp/marlin_repro_prof.py`（torch.profiler + kernel key + 依赖链），均：

1. 从 safetensors 读 `weight_packed`（[OUT, K/8] int32）、`weight_scale`（bf16）、`weight_zero_point`；
2. 执行与生产 `MarlinLinearKernel.process_weights_after_loading` 相同的转换：
   - `wp.permute(1,0)` → `ops.gptq_marlin_repack(..., num_bits=4, is_a_8bit=False)`（perm 空 int32）
   - `marlin_permute_scales(ws_fp16, group_size=128)` ；`marlin_zero_points(unpack_cols(zp.t(), 4, ...))`
   - `marlin_make_workspace_new` + 空 g_idx
3. 调 `apply_gptq_marlin_linear(x=randn(M, K), wtype=scalar_types.uint4, ...)`（生产同款入口，is_k_full=True）。

## 4. 数据（实测）

### 4.1 单层 GEMM 时间与带宽（普通计时，60 次取均）

| 形状 | M | per-call | 权重大小 | implied BW |
|---|---|---|---|---|
| q_proj N=12288 | 1 | 81.8 µs | 31.5 MB | 384.6 GB/s |
| q_proj N=12288 | 2 | 82.3 µs | 31.5 MB | 382.1 GB/s |
| q_proj N=12288 | 4 | 82.9 µs | 31.5 MB | 379.3 GB/s |
| q_proj N=12288 | 8 | 84.1 µs | 31.5 MB | 374.3 GB/s |
| gate_proj N=17408 | 4 | 108.5 µs | 44.6 MB | 410.6 GB/s |

- 结论 A：**M=1/2/4/8 时间基本不变**（权重读取主导，batch 行数非敏感）。
- 结论 B：单次调用实测带宽 **374–415 GB/s ≈ 616 GB/s 理论的 61–67%**（kernel 本身健康，无显著低效水分）。

### 4.2 kernel 身份（torch.profiler key）

复现与生产（服务 profiler）kernel 名一致：

```
void marlin::Marlin<1125899906910725l, 1125899906843648l,
                    1125899906910725l, 1125899906910725l, 256, 1, 8, 8, ...>
```

（profiler 下复现时间 81.7–108.5 µs/call，与普通计时一致 ⇒ **torch.profiler 无放大**）

### 4.3 理论下限与实测 step 的对照

- 每 decode step 单卡需读 int4 权重 ≈ 19.6 GiB ÷ TP2 ≈ 9.8 GB；616 GB/s → **理论下界 ≈ 15.9 ms/step**。
- 生产实测（fp16kv + MTP2 + 采样，4K）：95.1 char/s ≈ 45 tok/s ≈ **~22 ms/token**（含 attention/RNN/sampler 全部路径）。
- 结论 C：整 step 时间与"光读一遍权重"的理论下界处于同一数量级（≈70% 饱和量级）⇒ **decode 步成本本质是权重带宽硬墙**。

### 4.4 对 w8a8 / w4a8 的判定

- 带宽已 ~60–85% 饱和：单流 decode 无量化格式能显著提速。
- w8a8：权重字节 2× ⇒ 每步读取翻倍 ⇒ 只会更慢（-30~40% 量级）；int8 算力优势仅适用 compute-bound（prefill/大 batch），decode 用不上。
- w4a8：权重仍 4bit（带宽不变）+ int8 MMA 算力——**对单流 decode 无帮助**（带宽瓶颈未变）；其价值面仅在"投机解码多行 verify 的 compute 侧"，且 SM75 Marlin int8 变体可用性仍需另验（未测）。**不作为当前提速手段建议。**

## 5. 结论（含修正声明）

1. **Marlin kernel 本身健康**：61–67% 带宽利用率，无 "37 倍水分"（此前粗略估算有误——初版以 profiler 平均时长误推，本轮以**同 kernel 复现 + 真实 tok/s** 修正，作废旧估算）。
2. **decode 步成本 = 带宽硬墙**（每 token 读一遍全部权重是体系结构限制；w8a8/w4a8 均无法突破该墙）。
3. int8 KV 劣化（attention 27×）结论**不受影响**（与 GEMM 无关）。
4. 新疑点（未闭环）：MTP k=4 比 k=2 慢 40%——Marlin 对 M 不敏感，故该劣化**不在 GEMM 上**（疑 draft 前向次数或 verify 其他路径），留作后续验证项。

## 6. 复验清单

```bash
cd /tmp/dv
# ① M 扫描（普通计时）：
PYTHONPATH=/tmp/dv /opt/vllm-2080ti-definitive/.venv/bin/python /tmp/marlin_repro.py 60
#   预期：M=1 ~81.8us / 384GB/s；M=8 ~84.1us / 374GB/s（±3%）
# ② profiler + kernel key：
PYTHONPATH=/tmp/dv /opt/vllm-2080ti-definitive/.venv/bin/python /tmp/marlin_repro_prof.py
#   预期：q_proj ~82us，gate ~109us；key 前导 "marlin::Marlin<1125899906910725l, ..."
# ③ 生产对照（需要服务）：fp16kv+MTP2 120K decode step profiler 中该 kernel key 相同。

# 复验注意：
# - 必须用 /tmp/dv worktree 的 vllm 源码（PYTHONPATH），venv 为 /opt/.../.venv
# - ScalarType 用 scalar_types.uint4（has_zp=1 的 AWQ 风格）；错用 int4/uint4b8 会触发
#   "b_type must be u4 or u8 when has_zp=True" 或 "Invalid thread config"（这两报错本身也是
#   SM75 Marlin 配置矩阵的复验点）
# - 脚本在 TP=1（CUDA_VISIBLE_DEVICES=0）下运行；生产 TP=2 分片后权重减半、时间约减半
```

## 7. 数据/脚本存档

- 脚本：`/tmp/marlin_repro.py`、`/tmp/marlin_repro_prof.py`
- 采样层：layers.11 q_proj / gate_proj（AWQ）
- 本记录文档：`docs/lab-validation-sampler-draftvocab-20260828.md` §11（见后文追加）或本文件独立存档