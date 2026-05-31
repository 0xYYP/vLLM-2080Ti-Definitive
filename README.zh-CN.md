<!-- markdownlint-disable MD001 MD041 -->
# vLLM 2080 Ti Definitive Edition

面向双 RTX 2080 Ti / SM75 推理的终极版 vLLM 运行时。

这是一个硬件定向的 vLLM fork，用来保存已经跑通的 2080 Ti vLLM
栈：补丁源码、启动 profile、运行时说明和稳定环境记录。

语言：[English](README.md) | 简体中文

## 为什么用 RTX 2080 Ti 做 LLM 推理？

2018 年 8 月，NVIDIA 推出了划时代的 RTX 2080 Ti 系列显卡，并将玩家
显卡产品线从 GTX 带入 RTX 时代，从此开启了实时光追时代。这一代显卡
给无数电脑爱好者留下了难以磨灭的印记。八年之后，2080 Ti 依然可以在
2K 分辨率下流畅运行当下主流 3A 大作，可以说是老骥伏枥，志在千里。

而当年的 2080 Ti 还留下了两个非常关键的硬件空间：一是可以把 11 颗
1GB GDDR6 显存颗粒升级为 2GB 容量，从而获得 22GB 可用显存；二是它
保留了在 40 系之后被消费级显卡淘汰的 NVLink 高速互联接口。当高规格
核心、改造后的大显存、高速卡间互联，以及以今天眼光看依然很快的显存
带宽叠加在一起，我们在审视本地 AI 推理时发现，这个组合仍然有巨大的
用武之地。具体而言：

| 指标 | 2x 2080 Ti 22GB + NVLink | 3090 Ti 24GB 基线 | 倍率 |
|---|---:|---:|---:|
| 物理 CUDA core 数量 | 8,704 | 5,376 | 1.62x |
| SM 数量 | 136 | 84 | 1.62x |
| 物理 Tensor Core 数量 | 1,088 | 336 | 3.24x |
| Dense Tensor FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| 总物理显存带宽 | 1,232 GB/s | 1,008 GB/s | 1.22x |
| 总显存容量 | 44GB | 24GB | 1.83x |
| 二手价格锚点 | CNY 3,600，含 NVLink | 约 CNY 7,000-8,000 | 约 0.5x |

这个项目的核心判断很简单：用约一半 RTX 3090 Ti 二手价格，组出双
22GB RTX 2080 Ti + NVLink，并在 LLM 推理真正关心的物理资源上持平甚至
超过 3090 Ti，再通过 vLLM 运行时优化把这些资源转化成真实 token 产出。

这就是本 fork 的首要价值：把老但仍然很强的 Turing 硅片，通过 Marlin、
FlashQLA/FlashInfer/FA2、TurboQuant/INT8 KV、MTP 和 CUDAGraph 集成，
变成一个严肃可用的 27B/31B 级别推理平台。

## 核心功能矩阵

| 功能 | Qwen-family 27B | Gemma4-family 31B |
|---|---|---|
| AWQ Marlin | 🟢 推荐权重路线 | 🔴 非当前路线 |
| GPTQ Marlin | 🟢 推荐权重路线 | 🟢 推荐权重路线 |
| MTP speculative decoding | 🟢 原生加速路线 | 🟡 外部 draft 可用，但无速度收益 |
| `turboquant_4bit_nc` KV | 🟢 推荐容量路线 | 🟢 推荐容量路线 |
| `turboquant_k8v4` KV | 🟡 可用，但优先级较低 | 🔴 非当前路线 |
| INT8 KV continuation fast path | 🟢 推荐 MTP/容量路线 | 🔴 非当前路线 |
| YaRN 524K extension | 🟡 容量/离线实验 | 🔴 未验证 |
| No-eager/CUDAGraph noMTP | 🟢 推荐 graph 路线 | 🟢 推荐 graph 路线 |
| No-eager/CUDAGraph with MTP | 🟢 推荐 graph 路线 | 🟡 可用，但速度收益有限 |
| Fast prefill path | 🟢 FlashQLA + FlashInfer/FA2 | 🟢 FlashInfer/FA2 + SDPA512 控制 |
| Peak MTP=3 single-request PP4096/TG128 | 1841.7 / 101.3 tok/s | 1665.9 / 44.3 tok/s |

## 目标运行环境

- GPU：双 RTX 2080 Ti，SM75，tensor parallel size 2
- CUDA/PyTorch：CUDA 12.8，`torch 2.11.0+cu128`
- 基础 vLLM：`0.21.0`
- 仓库身份：`vllm-2080ti-definitive`
- 稳定运行时身份：`vllm-sm75-tp2-cu128`

这个 fork 不是通用的上游 vLLM 替代品。它保留的是 SM75/Turing serving、
FlashQLA、FlashInfer、TurboQuant KV、INT8 KV continuation、MTP/CUDAGraph
safety，以及 Qwen/Gemma 生产 profile 所需的本地补丁。

重要本地文档：

- `docs/2080ti-stable/README-STABLE.md`：稳定运行时契约和 benchmark 证据。
- `docs/2080ti-stable/README-cu128.md`：CUDA 12.8 迁移和验证记录。
- `bench_tools/stable_profile.sh`：模型/profile 环境预设。
- `bench_tools/remote_start_vllm_cu128.sh`：环境变量驱动的启动模板。

相关项目：

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双
  2080 Ti 模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。
  本仓库则聚焦于 vLLM 运行时源码、补丁和启动配置本身。

## 上游 vLLM README 中文说明

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
简单、快速、低成本的 LLM serving 引擎
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>文档</b></a> | <a href="https://blog.vllm.ai/"><b>博客</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>论文</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> |
</p>

vLLM 提供了网站帮助用户快速上手。请访问 [vllm.ai](https://vllm.ai)
了解更多信息；活动信息见 [vllm.ai/events](https://vllm.ai/events)。

---

## 关于 vLLM

vLLM 是一个快速、易用的 LLM 推理和 serving 库。

vLLM 最初由 UC Berkeley 的
[Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，现在已经发展成一个
活跃的开源 AI 项目，由来自众多学术机构和公司的贡献者共同维护。

vLLM 的性能能力包括：

- 高吞吐 serving
- 使用 [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
  高效管理 attention key/value memory
- continuous batching、chunked prefill、prefix caching
- piecewise/full CUDA/HIP graphs
- FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、
  compressed-tensors、ModelOpt、TorchAO 等量化路线
- FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA、Triton 等 attention kernel
- 使用 CUTLASS、TRTLLM-GEN、CuTeDSL 等优化 GEMM/MoE kernel
- n-gram、suffix、EAGLE、DFlash 等 speculative decoding
- 基于 `torch.compile` 的自动 kernel 生成和 graph-level transformation
- disaggregated prefill、decode、encode

vLLM 的易用性包括：

- 集成常见 Hugging Face 模型
- 支持 parallel sampling、beam search 等多种 decoding 算法
- 支持 tensor、pipeline、data、expert、context parallelism
- streaming output
- 使用 xgrammar 或 guidance 进行 structured output
- tool calling 和 reasoning parser
- OpenAI-compatible API server、Anthropic Messages API、gRPC
- 高效 multi-LoRA
- 支持 NVIDIA GPU、AMD GPU、x86/ARM/PowerPC CPU，以及 TPU、Intel Gaudi、
  IBM Spyre、Huawei Ascend、Rebellions NPU、Apple Silicon、MetaX GPU 等硬件插件

vLLM 支持 Hugging Face 上的 200+ 模型架构，包括：

- decoder-only LLM，例如 Llama、Qwen、Gemma
- MoE LLM，例如 Mixtral、DeepSeek-V3、Qwen-MoE、GPT-OSS
- hybrid attention 和 state-space 模型，例如 Mamba、Qwen3.5
- multi-modal 模型，例如 LLaVA、Qwen-VL、Pixtral
- embedding/retrieval 模型，例如 E5-Mistral、GTE、ColBERT
- reward/classification 模型，例如 Qwen-Math

完整支持列表见
[Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)。

## 快速开始

使用 [`uv`](https://docs.astral.sh/uv/)（推荐）或 `pip` 安装 vLLM：

```bash
uv pip install vllm
```

也可以参考文档从源码构建：
[Build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)。

更多信息见 [vLLM 文档](https://docs.vllm.ai/en/latest/)：

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## 贡献

vLLM 欢迎各类贡献与协作。参与方式见
[Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html)。

## 引用

如果你在研究中使用 vLLM，请引用论文：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系方式

- 技术问题和功能请求：GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- 用户讨论：[vLLM Forum](https://discuss.vllm.ai)
- 开发协作：[Slack](https://slack.vllm.ai)
- 安全披露：GitHub [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
- 合作：[collaboration@vllm.ai](mailto:collaboration@vllm.ai)

## Media Kit

- 如需使用 vLLM logo，请参考
  [media kit repo](https://github.com/vllm-project/media-kit)。
