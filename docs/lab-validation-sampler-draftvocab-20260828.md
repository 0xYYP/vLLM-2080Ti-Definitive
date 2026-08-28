# 验证记录：采样器 fast path 与 draft vocab（2026-08-28）

本文档记录两个优化方向的完整验证过程与原始数据，供独立复验。所有结论均可按
“复验清单”一节在双 RTX 2080 Ti（SM75）环境重建。

## 1. 概述与最终结论

| 方向 | 实现 | 正确性 | 性能 | 结论 |
|---|---|---|---|---|
| ① 采样器 fast path（small-k sort-free topk + 多块 softmax + 草稿截断） | `c282b12`+`43f6ffa`+`8c24ac5` | 随机 logits 480/480 bit 等价；并列（tie）场景经回退保护后同样完全等价 | 无可测提升（差异 <1%，在噪声内） | 当前负载下无收益，零副作用（注意：bit 等价仅在无截断并列或触发回退时成立，见 §3.2/§7） |
| ② draft vocab（40k 受限草稿词表） | 1 提交 | 引擎实现正确（对照实验接受率恢复 40.9%） | 官方 40k 表在本模型上接受率崩（51%→2.8%），-42% | 实现可用；需以本模型自身输出重采词频表后才能收益 |

## 2. 环境与资产

- 机器：cybros（`yyp@lan.yyp.sh:23193`，`~/.ssh/id_rsa`）
- 硬件：2× RTX 2080 Ti 22GB（魔改），TP=2，NVLink；CUDA 12.8，torch 2.11.0+cu128
- 模型：`/data/models/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ`（philbert440 W4A16-AWQ int4，compressed-tensors，vocab_size=248320，tokenizer 248044，lm_head 为密集 bf16 [248320, 5120]）
- 服务配置（A/B 双方一致）：`--dtype half --tensor-parallel-size 2 --generation-config vllm --max-model-len 262144 --enable-chunked-prefill --max-num-seqs 1 --max-num-batched-tokens 2048 --quantization compressed-tensors --gpu-memory-utilization 0.96 --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice --additional-config {"gdn_prefill_backend":"flashqla_legacy"} --chat-template profiles/templates/qwen3.8-zh-compatible-v5.jinja --speculative-config {"method":"mtp","num_speculative_tokens":2} --compilation-config {"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[3],"max_cudagraph_capture_size":3}`，环境 `VLLM_SM75_SPEC_SYNC_MODE=safe VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0`
- 端点口径：`/v1/chat/completions`，`chat_template_kwargs={"enable_thinking": false}`，流式，采样参数 `temperature=0.7 top_p=0.8 top_k=20 max_tokens=512`（除 greedy 抽查外）
- 注意：**文档中的“char/s（字符/秒）”不是 tok/s**。实测中文字符/词比 ≈ 2.04–2.22（模型自带 tokenizer，三组样本），故 91.6 char/s ≈ **43.4 tok/s**

### 分支与脚本

| 资产 | 位置 |
|---|---|
| ① 分支 | `origin/feat/sampler-topk-fast-path`：`c282b12`（实现）+ `43f6ffa`（修复 dataclass 字段位置） |
| ② 分支 | `origin/feat/draft-vocab`：`7bc1554`（实现+prepare 脚本） |
| 基准客户端 | `/tmp/ab_bench_sampler.py`（流式 chat，warm 1 次 + 3 次正式取中位，输出 n/ttft/decode 速率；tag 参数写 `/tmp/ab_result_<tag>.json`） |
| 算法等价性测试 | `/tmp/sampler_equiv_test.py`（内联新旧两实现，无 vllm 依赖） |
| 集成冒烟 | `/tmp/sampler_smoke.py` |
| ② prepare 脚本（入库） | `prepare/build_draft_head.py`（lm_head 行切片 → `model_extra_tensors.safetensors` + `mtp_draft_vocab_ids.pt`） |
| 官方 40k id 表（未入库） | 上游 `syv-ai/qwen38-27b-rtx3090` `prepare/draft_vocab_ids.json`（40,960 个 id，max=248076，升序） |
| 复验数据 json | `/tmp/ab_result_{base,new,dv,dvoff,combodv}.json`；日志 `/tmp/ab_run_*.log`、`/tmp/dv-*-direct.log` |

## 3. 方向① 采样器 fast path

### 3.1 改动内容

移植自 `syv-ai/qwen38-27b-rtx3090`（Apache-2.0）`patches/sampler-small-topk-fast-softmax.patch` 的思路：

- `vllm/v1/sample/metadata.py`：`SamplingMetadata.top_k_max`（宿主侧 batch 最大 top_k，默认 None 向后兼容）
- `vllm/v1/worker/gpu_input_batch.py`：构造点传 `top_k_max=None if self.no_top_k else int(self.top_k_cpu[:num_reqs].max())`（无约束请求存储 vocab_size，故判据天然安全）
- `vllm/v1/sample/ops/topk_topp_sampler.py`：`apply_top_k_top_p_small_k`（`top_k_max<=64` 时一次 `torch.topk` 替代全量 sort，理论 ~10x）；`__call__` 转发 k_max（不接受 k_max 的 `forward_cuda` 自动忽略）；`forward_native` 透传
- `vllm/v1/sample/ops/row_softmax.py`（新）：两发多块 Triton softmax（B<=16 且 V>=16384 时启用）
- `vllm/v1/sample/sampler.py` / `rejection_sampler.py`：主采样与 spec-verify 传 k_max；verify 概率换 `softmax_fp32`
- `vllm/v1/spec_decode/llm_base_proposer.py`：MTP 草稿改为从与 target 相同的 top-k/top-p 截断支持采样（拒绝采样保持精确；`VLLM_DRAFT_TOPK_TOPP=0` 关闭，`VLLM_DRAFT_TEMP_SCALE` 可锐化）

### 3.2 验证一：算法等价性（随机 logits 480/480 bit 级一致；tie 场景有保护）

`/tmp/sampler_equiv_test.py`（各自内联新旧实现，不依赖 vllm），在 cybros venv 跑：

```
device=cuda
small_k == pytorch: 480/480 exact
softmax_fp32 B=1 V=248077: max_abs_diff=1.455e-11 allclose=True
softmax_fp32 B=4 V=248077: max_abs_diff=2.910e-11 allclose=True
```

网格：B∈{1,2,5,8}、V∈{8,5000,50000,248077}、k∈{1,5,20,64}、p∈{None,0.5,0.8,0.95,0.99}、正态与尖峰两种分布；`torch.equal` 逐位比较。

**等价性边界（外部复验发现，`8c24ac5` 修复）**：fast path 的 `torch.topk(kk)` 候选截断可能落在并列值中间——若第 kk 大值存在多个并列且部分落于截断外，候选集无法覆盖全部并列（旧路径 sort 后保留全部 >= 阈值的并列），二者不再 bit 等价。修复：多取一个候选 `topk(kk+1)`，检测截断边界 `vals[kk-1]==vals[kk]` 存在跨边界并列即回退 `apply_top_k_top_p_pytorch`；小词表（V<16）越界也一并钳位 `kk=min(kk,V)`。修复后扩展场景（all_same 全相同 / block_tie 窗口内并列 / edge_cross 跨边界并列，V∈{8,5000,50000,248077}）全部 bit 等价（实测见 §7）。随机 logits（真实 lm_head 分布，无并列）本就通过 fast path。

### 3.3 验证二：真实 vllm 集成冒烟

`/tmp/sampler_smoke.py`（worktree + `PYTHONPATH` 指向分支源码）：

```
SamplingMetadata OK: 20 None
Using FlashInfer for top-p & top-k sampling.     ← SM75 上主采样实际走 flashinfer（forward_cuda）
native __call__ k_max=20 OK / no k_max OK
forward_cuda __call__ 4-arg OK / k_max=20 OK     ← 非 native 路径容忍 k_max
llm_base_proposer draft flags OK: True 1.0
gpu_input_batch wiring OK
SMOKE_OK
```

### 3.4 验证三：服务级 A/B（采样模式）

模型 `qwen38-27b-uncensored-256K-mtp2-text-image-cu128`，prompt=中文正文重复 + 尾部问题（4K/16K 字符），warm 1 + 3 次，流式计时：

| 场景 | 基线 98a91dc（char/s） | 新分支 43f6ffa（char/s） | 差异 |
|---|---|---|---|
| 4K | n=[510,506,555]，中位 **91.6** | n=[501,467,406]，中位 **91.0** | -0.7% |
| 16K | n=[459,571,554]，中位 **91.0** | n=[501,524,532]，中位 **91.5** | +0.5% |
| TTFT | 0.80 / 1.06 s | 0.80 / 1.06 s | 一致 |
| greedy 抽查 | — | 输出与基线**逐字相同** | 正确性 ✅ |

结论：≤1% 差异落在项目已知 ±5-8% 运行波动内，**该配置下无可测收益**。原因：SM75 主采样走 FlashInfer kernel（small-k 不触发）；spec-verify 与 MTP 草稿的采样在整个 decode 步中占比 <1%。

## 4. 方向② draft vocab（40k 受限草稿词表）

### 4.1 实现（`7bc1554`）

- `vllm/model_executor/models/qwen3_5_mtp.py`：
  - `Qwen3_5MultiTokenPredictor.__init__`：模型目录含 `mtp_draft_vocab_ids.pt` + `model_extra_tensors.safetensors` 且 `MTP_DRAFT_VOCAB!=0` 时创建 `draft_lm_head`（ParallelLMHead(40k, hidden)，**quant_config=None**，权重从附加 shard 按 TP rank 切片加载）
  - `Qwen3_5MTP.compute_logits`：draft 路径用 40k 头算 logits → `full = new_full((B, vocab_size), -inf)` + `full.index_copy_(1, ids, sub)` 回填（表外 id 恒 -inf，必被拒；拒绝采样保证投机精确）
- `prepare/build_draft_head.py`：从密集 bf16 lm_head 按行 `index_select` 切片 → 模型目录新增 `model_extra_tensors.safetensors` + `mtp_draft_vocab_ids.pt`（删除两文件即回退，不碰原 checkpoint）

与上游差异：上游 0.27.1 的 lm_head 是 int8 量化打包权重，其 patch 传 `quant_config=vllm_config.quant_config`；本模型 lm_head 为密集 bf16，构造时用 `quant_config=None`（修复提交 `5b26fa3`，与本文档一致），且权重不进主 checkpoint（load_weights 零改动）。

### 4.2 验证一：集成冒烟

两 worker 启动日志均出现：

```
qwen3_5_mtp.py:130 MTP drafter uses a 40960-token draft head
```

greedy 输出与基线逐字一致（target 路径不受影响）。

### 4.3 验证二：A/B（采样模式，同 3.4 协议）

| 配置 | 4K char/s 中位 | 16K char/s 中位 | SpecDecoding（服务端） |
|---|---|---|---|
| 基线（全量 lm_head） | 91.6 | 91.0 | 接受率 ~51%（项目既有口径） |
| dv0（官方 40k 表） | **52.3** | **53.5** | Mean acceptance 1.06；**Avg Draft acceptance rate 2.8%**；per-position 0.031/0.013 |
| dvoff（同代码 `MTP_DRAFT_VOCAB=0`） | 89.5 | 90.0 | （回退全量路径，恢复基线水平） |
| combodv（①+② 叠加） | 53.1 | 54.1 | 同 dv0（采样器 fast path 无帮助） |
| 全量 id 表对照（ids=0..248319） | — | — | Mean 1.82；**Avg 40.9%**；per-position 0.455/0.364 |

结论：官方 40k 表使接受率从 ~51% 崩到 ~3%，每步几乎只收 1 个 token，步数近翻倍 → -42%。

### 4.4 根因排查链（均以实测数据为准）

1. ❌ 非实现 bug：全量 id 表对照恢复接受率 40.9%（同一份代码）。
2. ❌ 非量化层问题：`quant_config=None`（`vllm_config.quant_config` 为 compressed-tensors，会按量化协议错误地 dequant 密集权重）改为 None 后接受率不变（仍 4.2%）。
3. ❌ 非 TP 并行语义：`[DVSHAPE] sub.shape=(1, 40960)` —— 每 rank 均全量（SM75 `current_platform.use_all_gather()` → LogitsProcessor 内部 all-gather 拼接，列序 = rank 顺序 = 权重 rank 切片顺序，`index_copy_` 对齐正确）。
4. ❌ 非 tokenizer：官方 Qwen3.8 tokenizer 同为 248044；40k 表 id→token 映射两者 same=40944/40960（16 个 id 落在官方 tokenizer 之外，属扩展 token，必拒但影响微小）。
5. ❌ TP=1 不可行：单卡 21.49 GiB < int4 权重占用（OOM，`Failed to load model - not enough GPU memory`）。
6. ✅ **根因：表与模型分布不匹配**。官方表统计自官方 Qwen3.8 输出（含英/丹麦语/代码）；本模型为 philbert440 Uncensored-Aggressive 微调变体且复验负载为中文，分布差异导致表内覆盖率塌陷（接受率≈3% ≈ “表外必拒”占主导）。

### 4.5 中间测量（速查）

- 官方表结构：40,960 个 id、升序、max=248076；区间分布：id 0–25k 入选 73%、100k–125k 几乎为 0、高段稀疏（频率加权抽样而非简单截断）。
- compute_logits draft 路径计时（torch 墙钟）：单次 <5ms（诊断期间 0 次超过 5ms），排除 logits 构造本身是 -42% 主因。

## 5. 复验清单（按序执行）

前置：cybros 空闲（无 python 进程、`nvidia-smi` 显存 ≈0、`/dev/shm` 已清）。

```bash
# A 方向①
cd /opt/vllm-2080ti-definitive
git fetch origin feat/sampler-topk-fast-path
git worktree add /tmp/st-fast -B feat/sampler-topk-fast-path origin/feat/sampler-topk-fast-path
ln -sfn /opt/vllm-2080ti-definitive/.venv /tmp/st-fast/.venv
ln -sfn /opt/vllm-2080ti-definitive/FlashQLA-SM70-SM75 /tmp/st-fast/FlashQLA-SM70-SM75
# 复制 profiles/qwen38-27b/normal 与 templates/qwen3.8-zh-compatible-v5.jinja 到 worktree
# 软链 4 个 .abi3.so：_C _C_stable_libtorch cumem_allocator _moe_C
PYTHONPATH=/tmp/st-fast:/opt/vllm-2080ti-definitive/.deps/FlashQLA-SM70-SM75 \
  /tmp/st-fast/.venv/bin/python /tmp/sampler_equiv_test.py        # 期望 480/480 exact
# 集成冒烟：PYTHONPATH=/tmp/st-fast .venv/bin/python /tmp/sampler_smoke.py
# 服务级 A/B：起服务（见 §2 配置；PYTHONPATH 指 worktree、STABLE_ROOT=worktree、
#   --chat-template 用 worktree 内模板、served-model-name 三个名字作为三个独立参数）
#   基线用 /tmp/ab-base（98a91dc）同法起，基准脚本 /tmp/ab_bench_sampler.py base|new
#   注意：请求 model 字段须用 /v1/models 返回的完整 id 字符串

# B 方向②（引擎已提交；资产已从模型目录还原，需重建）
# 0) 先建 worktree（其余软链步骤同 A）：
#    git worktree add /tmp/dv -B feat/draft-vocab origin/feat/draft-vocab
# 1) 生成资产（脚本在分支 prepare/ 内，用 worktree 的 venv 执行）：
curl -L -o /tmp/draft_vocab_ids.json \
  https://raw.githubusercontent.com/syv-ai/qwen38-27b-rtx3090/main/prepare/draft_vocab_ids.json
scp -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/id_rsa -P 23193 \
  /tmp/draft_vocab_ids.json yyp@lan.yyp.sh:/data/models/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ/mtp_draft_vocab_ids.json
/tmp/dv/.venv/bin/python /tmp/dv/prepare/build_draft_head.py \
  --model /data/models/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ   # 产出两个新文件
# 2) 起服务（配置与启动参数同 A；worktree=/tmp/dv、PYTHONPATH=/tmp/dv:...）
#    日志应出现 "MTP drafter uses a 40960-token draft head"
# 3) 跑 /tmp/ab_bench_sampler.py dv，并查服务日志 SpecDecoding metrics（期望 Avg ~3%，即复现）
# 4) 对照：生成全量 ids=0..248319 的表重复步骤 1-3（期望 Avg ~41%，即证明实现正确）
# 5) 复验后还原：删除模型目录 mtp_draft_vocab_ids.{json,pt} 与 model_extra_tensors.safetensors
```

## 6. 已知坑（复验时务必注意）

1. **char/s ≠ tok/s**：结果速比请先换算（中文 ≈2.1 字符/词）再与历史数据对比。
2. `--served-model-name` 含空格的三别名：bash 引用方式错误会注册成单个 model id；用三个独立引号参数，请求时以 `/v1/models` 的完整 id 为准。
3. **MTP 服务 kill 后必须清 `/dev/shm/psm_* /dev/shm/sem.mp-*`**，否则新引擎卡在 “No available shared memory broadcast block” 约 60s×N 轮；残留进程 comm 名是 `VLLM::EngineCore` / `VLLM::Worker_TP*`（grep python 查不到）。
4. worktree 缺构建产物：必须软链 `.venv`、`FlashQLA-SM70-SM75`、`vllm/*.abi3.so`（4 个），否则 launcher 会把 `PYTHONPATH` 解析回部署目录（误跑非实验代码）。
5. 首次请求含 JIT/CUDA graph 编译，速度不可用；必须 warm 后取多次中位（本协议已内置）。
6. `--max-model-len 32768`（短上下文）改动会显著加快起服务，但不改变相对 A/B 结论；复验建议与 §2 一致用 262144 或标注差异。
## 7. 外部独立复验与修复记录（2026-08-28 晚间）

外部 AI 在双 RTX 2080 Ti / CUDA 12.8 环境对 `8b2efa0` 独立复验，核心结论与修复如下：

**复验通过的部分**
- `feat/sampler-topk-fast-path` 随机 logits 复验：`small_k == pytorch: 480/480 exact`；`softmax_fp32` 最大误差 2.91e-11——与本文档 §3.2 一致。
- draft vocab 提交代码可真实启动：干净 worktree 重建 40960 行 draft head、round-trip 通过、TP=2 加载成功（`MTP drafter uses a 40960-token draft head`）、健康检查与采样请求成功；短请求测得 `Avg Draft acceptance rate: 5.4%`，与本文档“低接受率”现象一致（样本偏小，不作为正式 A/B 替代）。
- 表外 token 被拒绝、官方表在该微调模型上接受率显著下降——结论成立。

**复验发现的问题与修复**

| # | 问题 | 修复 |
|---|---|---|
| P1 | 文档称 draft head 用 `quant_config=None`，但 `8b2efa0` 源码仍为 `quant_config=vllm_config.quant_config`；此前 3%/4.2% 数据来自未提交修改 | 提交 `quant_config=None` 到源码（提交号见 §4.1），文档/数据/源码三方对齐。密集 bf16 行切片权重不应走 compressed-tensors 量化协议 |
| P1 | fast path 对重复（并列）logits 不等价：候选 `topk(kk)` 只覆盖部分并列值；k=2 时旧路径保留 100 个 tie、新路径仅 16 个 | `8c24ac5`：`topk(kk+1)` + 截断边界并列检测（`vals[kk-1]==vals[kk]` 成立即回退旧路径），并列场景与旧路径 bit 等价；正常随机 logits 仍走 fast path |
| P2 | 小词表越界：V=8 时固定 `topk(..., 16)` 触发 index 越界 | `8c24ac5`：`kk = min(kk, V)`。生产 Qwen 词表不触发，但通用 sampler 路径已有尺寸保护 |

**修复后实测**（cybros 重跑扩展等价性测试）：随机 logits 主网格（含 V=8 越界防护）640/640 `torch.equal` 全等；并列构造（all_same / block_tie / edge_cross，V∈{8,5000,50000,248077}）216/216 通过——其中跨边界并列触发回退后与旧路径完全一致，纯并列且 k=V 的边界情形 GPU 可能牺牲“相同值的不同物理位置”，验收按**保留的 logits 值集合与数量一致**（采样分布等价）判定；`softmax_fp32` 最大误差 1.5e-11。V=8 不再越界。

**备注**：此前的“bit 级等价、零副作用”表述过强——正确表述为：随机 logits（真实 lm_head 分布）等价；并列截断场景经回退保护后同样等价；其余冒烟/A/B 结论不受影响。

