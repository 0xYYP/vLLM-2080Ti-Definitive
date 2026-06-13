# 模型 Profile 路线

本文记录部署 profile 的证据口径。当前 profile 清单、含义和实测吞吐统一维护在
[Profile 导引](../profiles/README.zh-CN.md)。

`Profile` 是 `launcher.sh` 选择的相对 `.env` 路径；具体 checkpoint 仍然通过
`MODEL_DIR` 单独选择。

目标 profile 和未完成路线追踪见
[Profile 蓝图草案](profile-blueprint.zh-CN.md)。

## 证据口径

完整通过表示真实请求返回 HTTP 200、stream 正常结束，并且中文质量 smoke 没有
重复、残缺、乱码或明显答非所问。

平台期通过只说明容量风险较低；如果质量 smoke 失败，即使容量或合成吞吐可用，
也不晋升 profile。

只 load 成功、READY、health 通过、小窗口 smoke、空 stream，都不算容量证据。

## KV 精度定位

- FP16/default KV 是质量路线。
- INT8 KV 是容量 / 平衡路线；当前只保留 `normal` / piecewise profile。
- TQK8V4 是 TurboQuant 压缩路线；当前只保留质量通过的 `fast` profile。
- TQ4NC 有过容量实验，但当前正式 profile 不采用。

## 说明

- Profile 按 `profiles/<model>/<mode>/<weight>/<route>.env` 组织。
- `normal` 是当前推荐生产路线；`fast` 只保留质量 smoke 通过的高性能路线；
  `safe` 是 launcher 的 eager 回退档，不作为当前正式 profile 目录。
- FP8 + FP16KV 在 `normal` 模式下正式上下文为 128K；当前 `fast` 路线为
  112K：120K admission 失败，128K 短质量 smoke 可过但 PP4096/TG128 吞吐
  稳定性未通过。
- FP8 + TQK8V4 已验证 256K 纯文本和 240K 图文；图文路线使用 GPU util 0.96。
- fast + INT8KV 不保留：容量或合成速度可以成立，但中文质量 smoke 出现重复或
  残缺输出。
- 吞吐背景记录见
  [Qwen3.6 KV 吞吐 Sweep](qwen36-kv-throughput-sweep.zh-CN.md) 和
  [MTP 任务敏感性](mtp-task-sensitivity.md)。
