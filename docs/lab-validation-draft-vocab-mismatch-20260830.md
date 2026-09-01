# draft-vocab 词频表与用途匹配性验证（2026-08-30）

状态：**验证完成，用户假设成立**。16384 词频表以中文为主，与编程/agent
使用方式存在显著覆盖差距；修复方向为按真实用途重采样词频表。

## 1. 背景

- 当前服务（Q0）启用 draft-vocab（`feat/draft-vocab`，自建 16384 表
  `prepare/draft_vocab_qwen38_cn_16384.json`）。
- 表构建口径（commit 7b088e9）：113 条**多样中文 prompt** 采样模型自身输出
  268k token，取 top 16384（覆盖 96.32% held-out）。
- 用户怀疑：表主要从**中文闲聊**收集，与自身的**编程 / agent 工具调用**
  使用方式不匹配 → 覆盖率低 → draft 接受率低 → 收益打折。

## 2. 验证方法

- 环境：cybros，`/tmp/dv/.venv`，模型 `Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ`
  tokenizer（AutoTokenizer，trust_remote_code）。
- 静态覆盖率实验（`/tmp/dv_coverage2.py`）：三类语料
  （中文闲聊 20 条 / 编程代码技术 21 条 / agent 工具调用 20 条）
  分别 tokenize，统计表内 id 占比（覆盖率 + 频率加权覆盖率）。
- 表构成分类：16384 个 id 逐个 `tok.decode([id])` 按字符集归类。

## 3. 结果

### 3.1 表构成（16384 id）

| 类别 | 数量 | 占比 |
|---|---:|---:|
| 含中文 | 13116 | **80.1%** |
| ASCII 其他（符号/空白前缀） | 1521 | 9.3% |
| 纯英文 | 1489 | 9.1% |
| 其他（混合） | 235 | 1.4% |
| 空白/控制 | 23 | 0.1% |

→ **表以中文 token 为主体（80%），英文仅 ~9%**；与"中文闲聊收集"口径一致。

### 3.2 三类语料覆盖率

| 语料 | token 总数 | 表内 | 覆盖率 | 频率加权 |
|---|---:|---:|---:|---:|
| 中文闲聊/日常 | 366 | 269 | **73.5%** | 73.5% |
| 编程/代码/技术 | 586 | 370 | **63.1%** | 63.1% |
| agent/工具调用 | 600 | 443 | **73.8%** | 73.8% |

→ **编程语料覆盖率比中文低 10.4 个百分点**（63.1% vs 73.5%）；表外 37% 的
token 每步必拒（`const`/`SELECT`/`FROM`/`pool`/`users`/`mid`/`');` 等代码 token
几乎全在表外，见 §4 样例）。

- agent/工具调用覆盖率看似不低（73.8%），但样本为**中英混合**（工具描述中文、
  JSON 字段英文）；纯英文 tool schema / 工具名（`function`/`search_documents`/
  `parameters`/`query`/`limit`）同样在表外。

### 3.3 表外 token 样例（编程语料，出现≥1 次）

```
const  SELECT  FROM  users  pool  pg  its  ');'  .id  ')//'  '[:'  mid  ']);'
await  '{"'  type  '":"'  function  '","'  search  '_documents'  parameters  query  '}}}'
```

## 4. 结论

1. **假设成立**：16384 表 80% 是中文 token；编程/技术场景覆盖率 63.1%，
   比中文闲聊（73.5%）低 10.4pp；
2. 表外收词为代码关键词、SQL、JSON schema 字段、英文技术词——正是用户
   "编程 + agent" 日常负载的高频 token；
3. 覆盖率 ≈ draft 可打分 token 比例；表外必拒 → 每次 decoding step 中
   表外 token 位置无法被 draft 覆盖 → MTP 草稿推进在这些位置浪费。
   （注：覆盖率不直接等于接受率；服务端真实接受率需要 A/B 实测，见 §5 建议。）

## 5. 建议（若继续此方向）

- **按真实用途重采样词频表（已实现采集工具，commit 5706853）**：用真实使用
  负载统计数据——用户正常使用一周（编程/agent/中文混合），服务端自动收集
  **模型实际输出 token id** 频率：
  1. **启用**：起服务前 `export DRAFT_VOCAB_TRACE=<model_dir>/draft_vocab_freq.json`
     （示例脚本 `/tmp/start_trace_service.sh`；无 env 时零开销）；
  2. **使用一周**：正常用，后台 daemon 线程定期落盘（默认每 300s/2000 请求，
     可 `DRAFT_VOCAB_TRACE_FLUSH_SECS`/`_FLUSH_REQS` 调整），重启自动续写；
  3. **导出建表**：一周后从 trace JSON 取 top 16384 生成新表
     （扩展 `build_draft_vocab.py` 支持 `--freq-json` 直通，或复用其统计逻辑），
     替换现有 `draft_vocab_qwen38_cn_16384.json` 并重跑接受率 A/B。
- 重采样后预期：编程语料覆盖率显著回升（目标 ≥90%），draft 接受率与
  4K/16K 增益相应恢复；
- 验证口径：同本记录 3.2 的静态覆盖率 + 服务端 A/B（当前表 vs 新表，
  SpecDecoding Avg acceptance rate + 4K/32K/128K decode）。
- 注意：trace 采集的是**最终采样输出**（MTP verify 后），含思考链与正文，
  与 draft-vocab 服务的需求分布一致；被拒草稿不计（正确）。

## 6. 相关资产

- 表文件：`prepare/draft_vocab_qwen38_cn_16384.json`（16384 ids，min 0 max 248076）
- 采集链：`prepare/sample_model_outputs.py` → `prepare/build_draft_vocab.py`
  → `prepare/build_draft_head.py`
- 验证脚本：`/tmp/dv_coverage2.py`（远端）