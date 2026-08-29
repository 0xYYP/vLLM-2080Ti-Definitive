# 执行计划：int8 KV 长上下文劣化修复（split-KV 立项）

状态：**复审结论：有条件批准阶段 0**（技术侦察）。阶段 1 需在阶段 0 产出 4 项证据后进入；暂不按 A1 直接开启 3D，也不允许未经路径闭环就进入 kernel 实现。
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

### 阶段 0：技术侦察（1-2 小时，纯代码/小实验，不写生产内核）——已批准执行

目标：产出 4 项证据（复审要求），确认 int8 verify attention 的真实路由与 split-KV/3D 可用性。**未证明前不按 A1 直接开启 3D**。

- 0.1 **证据 1：真实路由矩阵**。分别以 (a) q=1 no-MTP 与 (b) MTP 启用（verify 实际 query 行数）两种服务配置，各发 120K 请求；记录 `query_start_loc`、`max_query_len`、`num_actual_tokens`、实际 draft token 数（**不预设 q=2**），产出「q × attention 变体」路由矩阵。
- 0.2 **证据 2：三条路径 provenance**。forward() 顺序：实验性 FA decode → FA prefill/bridge → unified attention（`triton_attn.py:2132`）。为 default bridge、direct-paged（`VLLM_INT8KV_FA_DECODE=1`）、unified attention 三条路线分别保存：完整 `--print-config`、全部 `VLLM_INT8KV_*` 环境变量、`INT8 KV FlashInfer ... used`/`fa_decode_failed`/native fallback 日志、原始 profiler 文件。**282.7ms unified_attention 必须与同次实验的上述路由日志绑定**（不能仅凭 kernel 名推断）。
- 0.3 **证据 3：3D 可用性，源码 + 运行时双重证据**。核实 `triton_unified_attention.py` 中 3D 的全部关闭条件（含 max_seqlen_q>1 强制关 3D、INT8_PER_TOKEN_HEAD 无条件关 3D）；确认 `VLLM_INT8KV_FA_DIRECT_PAGED_NOSPLIT` 仅系 FlashInfer wrapper 的 `disable_split_kv` 参数（`triton_attn.py:1121`），不是 unified-attention 3D 开关。运行时：实测 q=1/q=MTP 下 3D 是否可达。**若 3D 对 per-token-head int8 不可用 → 直接关闭 A1**（不把 planner 改动误当 split-KV 实现）。
- 0.4 **证据 4：SM75 资源评估**（若 3D 不可行需新 kernel）：寄存器/occupancy/smem 约束（参照 QO_LEN 否决先例），重估时间预算。
- 验收：全部 4 项证据写入日志后才进入阶段 1。

### 阶段 1：最小验证（1-2 小时；仅当 A1）

- 1.1 在 int8 服务开启 3D/split-KV 路径（配置/planner 最小改动），冒烟：单请求正常 + KV 路由日志（确认 split 生效的 kernel 名/参数变化）。
- 1.2 **attention 占比复测（3 次 decode step 取中位**，120K，与路由日志/profiler 文件绑定）：目标 attention 占比相对 24.7% 明显下降。**注意：占比下降不是充分成功标准**——若 bridge/dequant/synchronization 成本同时上升，不判成功；成功需同时满足端到端提升。
- 1.3 单流三档 A/B（4K/64K/120K，**cold prefill + prefix-cache-hit decode 分别记录**，warm 后 3 次中位 char/s + 离散度）：成功标准 = 相对 int8 基线（48.2/7.2/4.1）有**预定义的最低提升**，且相对 fp16（82.0/80.3/71.9）有明确的边界预期（如 ≥80% 水平）；**OOM/workspace 扩容边界一并记录**（现有 245K 证据限定 prefix-hit，cold prefill >60K 有 OOM 未验边界）。
- 1.4 正确性初检：needle 3 场景×3 深度命中、greedy 输出与 int8 基线一致（同一 prompt）。
- 止损：占比未降或 char/s 无提升或冷 prefill OOM → 关闭并交付数据（不进入实现）。

### 阶段 2：实现与正确性强化（2-4 小时；仅当 A2 或 1.2/1.3 通过后需要补强）

- 2.1（A2 时）实现 split-KV verify kernel（Triton/FlashQLA 适配），重点边界：per-seq causal、per-token-head scale、page 边界/last_page_len、TP=2 语义——先正确性后性能。
- 2.2 正确性验收（必须品）：**split-KV 与未拆分 reference 使用同一 Q/KV/page table/scale/causal metadata 逐元素对比**（覆盖 partial/last page、per-seq causal、GQA、TP=2；可及则含 LSE/softmax 中间态），**明确定义误差标准（atol/rtol）**，hash 一致仅作补充不作充分证明；4K/64K/120K 各 3 次 greedy 与基线 prefix 一致；25k 复述逐字；compute-sanitizer 若可用跑最小复现。
- 2.3 性能复测同 1.2/1.3。
- 止损：任一正确性失败 → 回滚该提交并记录；不超阶段预算。

### 阶段 3：容量收益量化（0.5-1 小时）

- 3.1 int8 vs fp16 的 KV 池容量对比：记录启动日志**实际 KV pool tokens/bytes**（计入 per-token-head scale、对齐、page rounding、GDN 状态、workspace、剩余显存），区分 cold context 容量与 prefix-hit 容量（不只看理论字节 2×）。
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