# 执行计划：int8 KV 长上下文劣化修复（split-KV 立项）

状态：**待用户复审**。复审通过后按阶段在 cybros 后台执行，全程日志 + 自动止损。
目标不是让 int8 超过 fp16，而是**让 int8 单流长上下文恢复到接近 fp16 水平**，从而解锁 int8 的容量路线（KV 池约 2 倍、并发多流/更大上下文池）。

## 0. 背景与目标

- 已证实的现状（2026-08-29 实测，torch.profiler 120K decode step）：
  - int8_per_token_head 与 fp16 的差异**几乎全在 unified_attention kernel**：10.4ms（1.1%）→ 282.7ms（24.7%），**27×**；Marlin GEMM 等其余路径基本不变（fp16 73% vs int8 52%，绝对耗时≈一致）。
  - 单流退化：64K -91%、120K -94%（char/s），TTFT 14-31×；quality（needle 9/9、greedy hash 确定）正常。
  - fp16 下 attention 仅 1.1%（方向③对 fp16 无意义，维持关闭）；**int8 下 24.7% ≥ 20%，方向③ 对症**。
- 目标：通过 split-KV（3D 并行拆分长 KV）把 int8 的 attention 成本压回接近 fp16；验证正确性（per-seq causal、per-token-head scale、page 边界）不回归；量化容量收益。
- 边界：不重开 fp16 生产选择；int8 修复价值在容量场景（并发/大池），单流如无容量需求则维持 fp16。

## 1. 前置资产与已知线索

| 资产/线索 | 位置 |
|---|---|
| int8 劣化 profiler 数据 | `docs/lab-validation-sampler-draftvocab-20260828.md` §10（single-step，计划内需补 3 次中位） |
| unified_attention 入口 | `vllm/v1/attention/ops/triton_unified_attention.py`（2D/3D 选择） |
| TritonAttentionImpl | `vllm/v1/attention/backends/triton_attn.py` |
| SM75 planner | `vllm/v1/attention/sm75_attention_planner.py` |
| int8kv decode 既有机制 | `VLLM_INT8KV_FA_DECODE`、`VLLM_INT8KV_FA_DIRECT_PAGED_NOSPLIT`（int8 专用、不直接外推 fp16；本项目即 int8 场景，可直接评估复用） |
| per-token-head V scale 先例 | flashinfer prefill.cuh per-token-head v_scale hook（direct_paged 修复，未入库，需重新评估） |
| FlashQLA legacy 后端 | `vllm/v1/attention/backends/fa_utils.py` 提及 FlashQLA/FlashInfer/TurboQuant SM75 专用 |
| 诊断/基准资产 | `/tmp/kvab_bench.py`、`/tmp/kvab_needle.py`、`/tmp/kvab_result_{fp16,int8}.json`、stream_120k.py |
| 服务启动模板 | 见 `docs/lab-validation-sampler-draftvocab-20260828.md` §2 与 kvab 实验记录 |

## 2. 阶段划分与验收标准

### 阶段 0：技术侦察（1-2 小时，纯代码/小实验，不写生产内核）

目标：确认 int8 verify attention（multi-query，q=2）当前走 unified_attention 的哪个变体，以及 split-KV/3D 路径的可开启性。

- 0.1 读 `triton_unified_attention.py`：2D/3D 选择条件、3D 是否=split-KV、解码路径（q=1 vs q=2 verify）各自走哪个变体；`sm75_attention_planner.py` 对 int8 decode 的路由。
- 0.2 确认 int8 per-token-head 的 dequant 位置：kernel 内（v_scale hook 类型）还是 bridge 层（kernel 外）；`VLLM_INT8KV_FA_DECODE` 与默认 bridge 的差异在统一 attention 时代的落点。
- 0.3 查 FlashQLA legacy / flashinfer 对「int8_per_token_head + split-KV + q=2 + SM75」的支持面（.so/plan/per-seq causal）。
- 0.4 **决策点 A**（数据说话）：
  - A1（3D 可经配置/planner 开启）→ 阶段 1；
  - A2（需新 kernel）→ 用 SM75 资源约束评估（寄存器/occupancy/smem；参照 QO_LEN 否决先例），可行 → 阶段 2 实现，不可行 → 关闭交付调查结论。
- 验收：产出路径结论 + 证据（代码位置/plan 支持面），写入日志。

### 阶段 1：最小验证（1-2 小时；仅当 A1）

- 1.1 在 int8 服务开启 3D/split-KV 路径（配置/planner 最小改动），冒烟：单请求正常 + KV 路由日志（确认 split 生效的 kernel 名/参数变化）。
- 1.2 **attention 占比复测（3 次 decode step 取中位**，120K）：目标 attention 时间占比相对 24.7% 明显下降（≤15% 视为方向有效）。
- 1.3 单流三档 A/B（4K/64K/120K，prefix-hit，warm 后 3 次中位 char/s）：目标 int8+split 相对 int8 基线（48.2/7.2/4.1）显著提升，向 fp16（82.0/80.3/71.9）靠拢。
- 1.4 正确性初检：needle 3 场景×3 深度命中、greedy 输出与 int8 基线 hash 一致（同一 prompt）。
- 止损：占比较未降或 char/s 无提升 → 关闭并交付数据（不进入实现）。

### 阶段 2：实现与正确性强化（2-4 小时；仅当 A2 或 1.2/1.3 通过后需要补强）

- 2.1（A2 时）实现 split-KV verify kernel（Triton/FlashQLA 适配），重点边界：per-seq causal、per-token-head scale、page 边界/last_page_len、TP=2 语义——先正确性后性能。
- 2.2 正确性验收（必须品）：**split-KV 输出与未拆分 reference 使用同一 Q/KV/page table/scale/metadata 逐元素对比**（可及则含 LSE/softmax 中间态）；4K/64K/120K 各 3 次 greedy 与基线 prefix 一致；25k 复述逐字；compute-sanitizer 若可用跑最小复现。
- 2.3 性能复测同 1.2/1.3。
- 止损：任一正确性失败 → 回滚该提交并记录；不超阶段预算。

### 阶段 3：容量收益量化（0.5-1 小时）

- 3.1 int8 vs fp16 的 KV 池容量对比（启动日志 KV 池 tokens ×2 确认）。
- 3.2 并发多流示范：MAX_NUM_SEQS=2-4 同长上下文（64K）的吞吐 vs fp16——验证"容量路线"成立条件。
- 3.3 判定：容量收益可量化且单流不再劣化 → int8+split 作为「容量优先」可选 profile 记录（不并入 fp16 性能结论）。

### 阶段 4：收尾（0.5 小时）

- 4.1 全量数据入档（`docs/lab-validation-sampler-draftvocab-20260828.md` 追加 §11 或新文档）。
- 4.2 分支提交（新分支 `feat/int8kv-splitkv`，不并入 feat/draft-vocab）。
- 4.3 现场清理（服务/显存/shm/临时 patch 还原）。
- 4.4 最终汇报：修复收益表 + 容量路线建议 + 是否合入主流程提议。

## 3. 时间预算（合计 5-9 小时，后台执行 + 自动止损）

| 阶段 | 预计 | 说明 |
|---|---|---|
| 0 | 1-2h | 侦察 + 决策点 A |
| 1 | 1-2h | 最小验证（A1） |
| 2 | 2-4h | 实现（仅 A2 或补强） |
| 3 | 0.5-1h | 容量量化 |
| 4 | 0.5h | 收尾 |

## 4. 执行方式

- cybros 后台：`nohup setsid` + 日志 `/tmp/int8kv-<phase>.log`；本地每 ~10 分钟短查询（防挂参数 + ServerAliveInterval），绝不 wait 长任务。
- 每次服务重启前 `rm -f /dev/shm/psm_* /dev/shm/sem.mp-*`；残留进程以 `VLLM::*` comm 识别。
- 每阶段结束写阶段日志（数据 + 决策）；全部完成汇总报告。

## 5. 风险与止损

| 风险 | 止损 |
|---|---|
| 3D split-KV 在 SM75/int8 不可开启（plan/资源不支持） | 阶段 0 决出即关闭，交付调查结论 |
| split 后 attention 占比不降/char/s 不升 | 阶段 1 关闭，不进入实现 |
| per-seq causal / per-token-head scale 正确性 bug | 先正确性后性能；逐元素 reference 对比；失败回滚 |
| pull 请求/网络不稳 | 提交本地保留，网络恢复自动推送 |
| 与解码 3D 并行路径（decode_context_parallel）冲突 | 侦察阶段确认互斥条件，不用时关闭 |

## 6. 输出物

1. 阶段日志（每阶段数据+决策）
2. int8 split-KV A/B 与占比数据表（3 次中位）
3. 正确性对比记录（reference 逐元素 + needle + hash）
4. 容量收益量化表
5. 分支 `feat/int8kv-splitkv` + 文档更新
6. 最终汇报（收益 + 容量路线建议）