# 模型 Profile 路线

本文记录部署 profile 的证据口径。当前 profile 清单、含义和实测吞吐统一维护在
[Profile 导引](../profiles/README.zh-CN.md)。

`Profile` 是 `launcher.sh` 选择的相对 `.env` 路径；具体 checkpoint 仍然通过
`MODEL_DIR` 单独选择。

## 证据口径

完整通过表示真实请求返回 HTTP 200、stream 正常结束，并且中文质量 smoke 没有
重复、残缺、乱码或明显答非所问。

平台期通过只说明容量风险较低；如果质量 smoke 失败，即使容量或合成吞吐可用，
也不晋升 profile。

只 load 成功、READY、health 通过、小窗口 smoke、空 stream，都不算容量证据。

## KV 精度定位

- FP16/default KV 是质量路线。
- INT8 KV 是容量 / 平衡路线；当前只保留 `normal` / piecewise profile。
  decode 默认走 dequant bridge（continuation/cascade 分块路径），不再回退
  原生 O(KV) 全量扫描（随上下文线性退化）。0.6.16rc4 实测（warm /
  completions / MTP3 / 双 2080 Ti / 272 布局）：4K 29.56 / 60K 21.46 /
  100K 16.08 tok/s（原生 250K 外推 ~1.5-2 tok/s）。同环境 A/B（2026-08-10，
  vLLM 固定 6426afb，仅切换 flashinfer）：0.6.8.post1 与 0.6.16rc4 桥路径
  无性能差异（4K 29.80 vs 29.56）；2026-08-07 的更高记录（4K 70.16）在
  固定代码下不可复现，已确认与 flashinfer 版本无关；历史差异与
  c805572 时代的代码/测量口径差异相符，但具体因素未进一步拆解。实验性 FlashInfer
  decode variant（`VLLM_INT8KV_FA_DECODE=1`）慢于桥（4K 18.15 tok/s，
  occupancy 瓶颈），默认关闭。245K（`int8kv-245K-mtp3-text-only.env`）
  是配置上限，仅 prefix-cache 命中后 decode 验证过；冷启动 60K+ prefill
  已观测 OOM（262144 在单次 100K 写入即 OOM）。详见
  `docs/int8kv-fa-decode.md`。
- TQK8V4 是 TurboQuant 压缩路线；当前只保留质量通过的 `fast` profile。
- TQ4NC 有过容量实验，但当前正式 profile 不采用。

## 说明

- Profile 按 `profiles/<model>/<mode>/<weight>/<route>.env` 组织。
- `normal` 是当前推荐生产路线；`fast` 只保留质量 smoke 通过的高性能路线；
  `safe` 是 launcher 的 eager 回退档，不作为当前正式 profile 目录。
- 同一套双 2080 Ti runtime 也已经验证了 Qwen3.6 35B FP8 MoE 路线。正式
  预设现在覆盖 256K `normal` / `aggressive` noMTP 纯文本路线、136K
  `normal` / `aggressive` noMTP 图文路线，以及一条 178K `fast` MTP3
  速度预设。
- FP8 + FP16KV `normal` 正式上下文为 256K，并且 `262016/128` 长提示 smoke
  已通过。
- FP8 + FP16KV `aggressive` 也已验证到 256K。正式记录的吞吐仍以
  `4096/128` 合成短测为准；接近满长的 `262016/128` 只作为容量 smoke，
  因为长跑里流式 chunk 合并会把 decode 速度抬高。
- FP8 + FP16KV 图文路线在 `normal` 和 `aggressive` 下都已验证到 136K。
  两条都通过了 `138240/128`，边界上的 `139008/64` 也可通过，而
  `139136/32` 会超过配置的 `139264` 上限。
- FP8 + FP16KV `fast` 当前验证到 178K，并且 `182144/128` 长提示 smoke 已通过。
- FP8 + TQK8V4 已验证 256K 纯文本和 240K 图文；图文路线使用 GPU util 0.96。
- fast + INT8KV 不保留：容量或合成速度可以成立，但中文质量 smoke 出现重复或
  残缺输出。
- 旧 `fast` + INT8KV 的兼容问题仍应修复，但这不代表把该路线重新晋升为
  fast 正式 profile。
- 吞吐背景记录见
  [Qwen3.6 KV 吞吐 Sweep](qwen36-kv-throughput-sweep.zh-CN.md) 和
  [MTP 任务敏感性](mtp-task-sensitivity.md)。
