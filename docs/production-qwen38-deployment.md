# 生产服务：qwen38-27b-uncensored（cybros）

本文记录 cybros 上对外 vLLM 服务的权威配置与操作方式。启动脚本是
`tools/start_qwen38_production.sh`，部署副本在
`/opt/vllm-2080ti-definitive/bin/start_qwen38_production.sh`。

## 服务身份

| 项 | 值 |
|---|---|
| 模型 | `/data/models/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ`（int4 W4A16） |
| served name | `qwen38-27b-uncensored-256K-mtp2-text-image-cu128`、`qwen3.8-27b-uncensored` |
| 端口 | `8000`（经 soup-proxy `:8020` 护栏与 New API 网关对外） |
| 栈 | 0.2 栈，`/data/models/.ai-stack-0.2` → `/opt/vllm-2080ti-definitive`，torch 2.13.0+cu130 |
| 日志 | `/data/models/.ai-stack-0.2/run-logs/qwen38-02-fp16kv-mtp4-g98-<时间戳>.log` |

served name 里的 `mtp2` 是历史字样、实际 `MTP k=4`；这两个名字被网关用于模型映射，
**不得改动**。

## 路由参数

256K 上下文 / `float16` KV / MTP k=4 / `max-num-seqs 1` 单流 / `gpu-memory-utilization 0.98`
/ `mamba-cache-mode align` / `--compression-config` 的 CUDA graph 捕获 `[5]`（= k+1）
/ 模板 `profiles/templates/qwen3.8-zh-compatible-v7.jinja`。

`max-num-batched-tokens 2048`、`compressed-tensors` 量化、`reasoning-parser qwen3`、
`tool-call-parser qwen3_coder` + `--enable-auto-tool-choice`、`--additional-config
{"gdn_prefill_backend":"flashqla_legacy"}`。

## 操作

```bash
S=/opt/vllm-2080ti-definitive/bin/start_qwen38_production.sh
bash $S              # 后台启动，等 /v1/models 就绪后返回（冷启动约 200s）
bash $S --status     # 进程 / 端口 / served 名 / 最新日志
bash $S --stop       # 优雅停止；--force 才升级 SIGKILL
bash $S --restart
bash $S --help       # 完整用法与可覆盖的环境变量
```

审计"现役进程是否就是脚本描述的那份配置"（argv 逐行比对）：

```bash
pid=$(bash $S --pids | head -1)
diff <(tr '\0' '\n' < /proc/$pid/cmdline | tail -n +4) <(bash $S --print-args)
```

2026-09-20 实测：对 `PID 2160605`（2026-09-17 21:04 启动、当时已连续运行 2 天 4 小时）
该 diff 无输出，43 个 argv 元素逐行相同。

每次启动都会把 `git HEAD`、工作区改动文件数、模板 md5 和完整 argv 写进日志头部，
事后排查配置漂移时先看这段，不要靠回忆。

## 开机自启（可选，尚未安装）

仓库带 `tools/qwen38-production.service` 与安装器：

```bash
sudo bash /opt/vllm-2080ti-definitive/bin/install_qwen38_production_service.sh
```

安装器会先停掉现存实例（含 nohup 起的），交给 systemd 托管并等就绪验收；预计停机
3–4 分钟。unit 用 `--foreground` 跑同一个脚本，日志同时进 journald 和 run-logs。
回退到 nohup 方式：`systemctl stop qwen38-production && bash $S`。

当前状态：**未安装**。在装上之前，机器重启后 vLLM 不会自动恢复（cybros 已连续运行
49 天，尚未遇到）。soup-proxy 是另一回事，它本来就是 enabled 的 systemd 服务。

## 与 launcher/profile 路线的关系

`profiles/qwen38-27b/normal/int4/fp16kv-256K-mtp2-uncensored-text-image.env` 走的是
`launcher.sh` + profile 的实验室路线，参数与生产不同（`MTP_K=2`、`GPU_UTIL=0.96`），
**不能用来复现或推断生产配置**。之所以生产不切到 launcher：`build_args` 会硬编码追加
模型目录名作为第三个 `--served-model-name`、按 `MESSAGE_TYPE` 自动补
`--limit-mm-per-prompt`，并把 `TORCHINDUCTOR_CACHE_DIR` 指到仓库目录而不是现役的
`cache/` 目录，三处都无法用环境变量关掉。

## 已知的复现缺口

启动脚本入库只解决了"怎么起"，从仓库完整重建生产还差两项，均待单独处理：

1. cybros 的工作区有 5 个文件相对其 HEAD 处于修改状态（`launcher.sh`、
   `compressed_tensors.py`、`qwen3_5.py`、`qwen3_5_mtp.py`、`output_processor.py`，
   合计 +98/−4 行），不在任何分支里。
2. `profiles/templates/` 下的 `qwen3.8-zh-compatible-v1..v6.jinja` 未入库（v7 已随本次
   入库）；另外 cybros 上的 v7 文件在 2026-09-20 01:15 被改过（md5 `9f61fbef`），
   该改动尚未重启生效、也未入库——仓库里的是当时在跑的 `cc3bccb4`。
