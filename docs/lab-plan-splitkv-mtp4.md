# 执行计划：长上下文优化（方向③ split-KV verify + MTP k=4 叠加）

状态：**待用户审批**。审批通过后按阶段在 cybros 后台执行，全程日志 + 自动止损。

## 0. 背景与目标

- 已完成并验证：draft-vocab（自建 16384 词频表，接受率 45.5%，4K +7.2% / 16K +2.5% char/s）。
- 已知事实：SM75 decode kernel occupancy 锁死（ncu 实证 12.5%、2 blocks/SM）；125K decode 93ms/步 vs 理论带宽 10ms（~9 倍差距）；项目已有 int8kv split-KV（direct_paged / NOSPLIT 开关）机制。
- 目标 A：MTP k=2 → k=4 叠加（低成本，草稿变便宜后 k=4 几乎免费）。
- 目标 B：方向③ split-KV verify——长上下文（60K-125K）下把 verify attention 的并行度拉起来。**先测量占比，占比不足则关闭**。
- **基线配置（用户确认的本设备好配置，2026-08-28）**：int4 权重 + fp16kv KV + 262144 上下文 + MTP2 + safe 同步模式（对应 profile `qwen38-27b/normal/int4/fp16kv-256K-mtp2-uncensored-text-image.env`）。**TQK8V4 与 nosync 经实测存在质量风险，不用于本计划任何基线或验证**。

## 1. 前置资产（均已就位）

| 资产 | 位置 |
|---|---|
| draft-vocab 分支（引擎 + quant_config 修正） | `feat/draft-vocab` HEAD `a0a43f1` |
| 16384 词频表 | `prepare/draft_vocab_qwen38_cn_16384.json` |
| 模型资产（切片权重） | `/data/models/.../model_extra_tensors.safetensors` + `mtp_draft_vocab_ids.pt`（已保留） |
| 采样/统计/切片脚本 | `prepare/sample_model_outputs.py`、`build_draft_vocab.py`、`build_draft_head.py` |
| A/B 基准脚本 | `/tmp/ab_bench_sampler.py`（warm 1 + 3 取中位，char/s + TTFT） |
| 服务可执行参数 | 见验证文档 §2（定稿需换成 TQK8V4 + nosync + 256K） |
| 已知坑 | 验证文档 §6（shm、别名、worktree 软链、warm） |

## 2. 阶段划分与验收标准

### 阶段 0：定稿配置基线 + verify 占比测量（~40 分钟）

- 0.1 定稿配置已确认（int4 + fp16kv + 262144 + MTP2 + safe）；以 profile `fp16kv-256K-mtp2-uncensored-text-image.env` 为基准，直接按其参数启动（不引入 TQK8V4/nosync）。
- 0.2 起**定稿配置（fp16kv+262K）**基线服务（feat/draft-vocab 代码 + 表资产 + MTP k=2），跑 4K/64K/120K 三档上下文 A/B 数据（warm + 3 中位）——存档为后续对照。
- 0.3 verify attention 占比测量（三选一，按可用性）：
  a. `sudo -n ncu` profile 一个 120K 上下文 decode 步（cybros 有 sudo ncu 先例），读 kernel 时间表；
  b. torch profiler（vLLM `--collect-detailed-traces` 或临时 kernel 计时）；
  c. 服务日志/SpecDecoding metrics（若统计可用）。
- **决策点 A**：verify attention（含 target 主 attention）步骤占比 ≥20%（步耗时贡献）→ 继续 B；<20% → ③ 关闭，投入 k=4 叠加与最终报告。
- 验收：产出占比数据表（kernel 名 + 耗时 + 占比），写入日志。

### 阶段 1：MTP k=4 叠加实验（~1 小时）

- 1.1 同服务，`--speculative-config {"method":"mtp","num_speculative_tokens":4}`。
- 1.2 冒烟：一次请求正常 + SpecDecoding metrics 读取（注意启动参数需去掉 `--disable-log-stats` 才能看接受率）。
- 1.3 四档 A/B：4K/16K/64K/120K，对照阶段 0.2 的 k=2 数据。
- 1.4 验收：
  - 接受率（mean acceptance length）≥ k=2 基线或仅小幅下降；
  - char/s 相比 k=2 **+5% 以上**；
  - greedy 抽查与 k=2 输出前缀一致（正确性）；
  - 稳定运行（无崩溃、无 illegal access）。
- 止损：k=4 崩溃（记忆：某些后端 k=4 有 illegal access 前科）→ 回退 k=3 复测，接受率/速度如仍过线则交付 k=3。

### 阶段 2：方向③ 探索与可行性（1-2 小时，纯调查不写内核）

- 2.1 阅读现有 split-KV 资产：`vllm/v1/attention/backends/triton_attn.py` 的 int8kv decode kernel 与 planner（`sm75_attention_planner.py`）、`VLLM_INT8KV_FA_DIRECT_PAGED_NOSPLIT` 开关语义——确认已实现的 split-kv 能否服务 verify（query_len=2）场景。
- 2.2 定位 FlashQLA legacy 的 verify 路径：spec-decode 的 attention 调用点（`vllm/v1/spec_decode/` 与 models/qwen3_5.py 的 verify 前向）当前走哪个后端/哪个 kernel，多 query 时是否 split。
- 2.3 FlashInfer 0.6.16 的 decode/verify kernel（含 flash-decoding / split-kv 变体）在 SM75 可用性盘点（.so 内是否有、plan 支持、per-seq causal）。
- **决策点 B**：
  - B1（现有 split-kv 可复用）→ 阶段 3.1 接入验证；
  - B2（需要新 Triton kernel）→ 用 SM75 资源约束评估（寄存器/occupancy/smem，参照既往 QO_LEN 否决先例）；可行 → 阶段 3.2；否决 → ③ 关闭，交付调查结论。
- 验收：可行/不可行结论 + 依据（代码路径或 kernel 资源分析），写入日志。

### 阶段 3：实现与正确性（仅当决策点 B 通过；2-4 小时）

- 3.1（B1）verify 路径接现有 split-kv：改 planner 路由或 spec-decode backend 参数，最小 diff。
- 3.2（B2）新 Triton split-KV verify kernel：按项目 kernel 惯例实现，重点边界：
  - per-seq causal（不同序列不同 query/context 长度）——上游 3 个 patch 的教训（causal-only assert、per-seq-causal 缺失、padded-page）；
  - last_page_len / padding（int8kv 既有坑）；
  - TP=2 语义（每 rank 数据切片）。
- 3.3 正确性验证（**必须品**）：4K/64K/120K 上下文各 3 次 greedy 长输出，与基线逐 token/text 前缀对比（text_prefixes=1 惯例）；中文长文复述（25k 文档）逐字比对；多轮会话无 garble；compute-sanitizer 若可用则跑最小复现。
- 3.4 性能预检：120K 上下文 decode 步耗时 vs 基线（kernel 计时）。
- 验收：正确性全部通过 + 性能预检有正收益（>0）才进入阶段 4；任一正确性失败 → 回滚该提交并记录。

### 阶段 4：长上下文 A/B 与收尾（~1 小时）

- 4.1 定稿配置（fp16kv+262K）全档 A/B：4K/16K/64K/120K ×（基线 k=2 ↔ draft-vocab k=2 ↔ 最优组合），每点 warm+3 中位。
- 4.2 综合指标：char/s、TTFT、接受率（metrics 日志）、稳定性（连续 20 请求无错）。
- 4.3 正确性终检：greedy 逐字一致性 + 复述任务。
- 4.4 结果入档：更新 `docs/lab-validation-sampler-draftvocab-20260828.md`（或新文档）记录全部分档数据；分支提交（feat/draft-vocab 或新分支 feat/split-kv-verify）。
- 4.5 最终汇报：收益表 + 建议（是否 PR 合入用户主流程、配置建议）。
- 验收：交付完整数据表 + 分支 + 文档。

## 3. 时间预算（合计 5.5-8.5 小时，睡前后台运行）

| 阶段 | 预计 | 说明 |
|---|---|---|
| 0 | ~40 min | 测量主导 |
| 1 | ~1 h | k=4 实验 |
| 2 | 1-2 h | 纯调查 |
| 3 | 2-4 h | 仅当 B 通过 |
| 4 | ~1 h | A/B + 文档 |

## 4. 执行方式（后台长任务）

- 全部动作在 cybros 执行：`nohup setsid` 起服务/长脚本，日志落 `/tmp/<phase>-*.log` + `~/ab-assets/` 数据文件；本地每 ~10 分钟短查询日志尾（单连接防挂参数 + `ServerAliveInterval`），绝不 wait 长任务。
- 每阶段完成即写阶段日志（含验收数据），全部阶段结束后汇总报告。
- 中途用户可随时中断（kill + 还原语义与验证文档 §6 一致）。

## 5. 风险与止损

| 风险 | 止损 |
|---|---|
| verify 占比低（<20%） | ③ 关闭，仅交付测量结论 |
| k=4 崩溃/接受率塌 | 回退 k=3 |
| 新 kernel 被 SM75 寄存器/occupancy 否决 | ③ 关闭（沿用既往 QO_LEN 先例） |
| split-kv 正确性 bug（per-seq causal 等） | 先正确性后性能；失败即回滚提交 |
| 网络/ssh 中断 | 任务在远端 nohup，本地重连继续查询 |
| 误用 TQK8V4/nosync | 基线一律用 fp16kv+262K+safe（用户确认），TQK8V4/nosync 不启用 |

## 6. 输出物清单

1. 阶段日志（每阶段数据 + 决策记录）
2. 定稿配置 A/B 数据表（存 `~/ab-assets/` + 文档）
3. 分支（`feat/draft-vocab` 续提交或新 `feat/split-kv-verify`）
4. 更新后的验证文档
5. 最终汇报（收益 + 是否合入建议）