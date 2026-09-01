# 决定记录：移除小字典，进入真实负载词频统计期（2026-08-30）

状态：**已执行**。小字典（16384 词频表 + vocab-truncated draft head）从
运行时路径移除；`DRAFT_VOCAB_TRACE` 真实负载统计保留，统计约一周后
用统计结果重建字典。

## 1. 决定

用户决定（2026-08-30）：

> 目前字典的方式，影响速度了，先移除小字典。后续一周统计好后再用小字典。

即：**暂不使用小字典跑生产**（它拖慢速度），同时让
`DRAFT_VOCAB_TRACE`（commit 5706853 的采集器）继续跑约一周，
收集真实工作负载（编程 / agent / 中文混合）的最终输出 token id 频率；
一周后用统计结果重建与用途匹配的字典。

## 2. 依据

- `docs/lab-validation-draft-vocab-mismatch-20260830.md`（同日验证）：
  16384 表 80.1% 为中文 token；编程语料覆盖率 63.1% 比中文闲聊 73.5%
  低 10.4pp；表外用词为代码关键词 / SQL / JSON schema / 英文技术词。
- 小字典的运行时开销：draft 路径每次 decode 都要对受限词表行打分后
  构造 `(B, vocab_size)` 全量 logits（`new_full(-inf)` + `index_copy_`
  回填），表外 id 恒 -inf 必被拒，接受率低（45.5%）→ 收益打折且
  全词表分配拖慢速度。

## 3. 执行内容（本提交）

- `vllm/model_executor/models/qwen3_5_mtp.py`：回退 draft head 补丁
  （`draft_lm_head` / `draft_vocab_ids` / `draft_logits_processor` /
  `compute_logits` 分支全部移除，恢复标准 lm_head 打分），
  与 main 完全一致。
- 删除 `prepare/draft_vocab_qwen38_cn_16384.json`（小字典本体，
  模拟采样产物）与 `prepare/sample_model_outputs.py`
  （113 条中文 prompt 模拟采样，被真实负载 trace 取代）。
- 保留：`vllm/v1/engine/draft_vocab_trace.py` + `output_processor.py`
  采集点（统计机制，无 `DRAFT_VOCAB_TRACE` env 时零开销）；
  `prepare/build_draft_vocab.py` / `build_draft_head.py`
  （一周后重建字典的工具，docstring 已更新说明新流程）。

## 4. 统计期使用方式（约一周，2026-08-30 → 09-06 前后）

1. 起服务前 `export DRAFT_VOCAB_TRACE=<model_dir>/draft_vocab_freq.json`
   （可选 `DRAFT_VOCAB_TRACE_FLUSH_SECS` / `_FLUSH_REQS`，默认
   300s / 2000 请求定期落盘；重启自动续写累积）。
2. 正常使用即可，无需其他动作；采集的是 MTP verify 后的最终采样
   token（含思考链与正文），被拒草稿不计。
3. 一周后导出：从 trace JSON 取 top 16384 生成新表（扩展
   `build_draft_vocab.py` 支持 `--freq-json` 直通，或复用其统计逻辑），
   替换后重跑接受率 A/B（验证口径见 mismatch 记录 §5）。

## 5. 相关资产与链接

- 统计采集器：`vllm/v1/engine/draft_vocab_trace.py`
- 依据记录：`docs/lab-validation-draft-vocab-mismatch-20260830.md`
- 重建工具：`prepare/build_draft_vocab.py`、`prepare/build_draft_head.py`
- 模型目录备用资产：`model_extra_tensors.safetensors` +
  `mtp_draft_vocab_ids.pt`（远端保留，未随本提交删除）