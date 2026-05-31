<!-- markdownlint-disable MD001 MD041 -->
# VLLM-2080ti

This is a hardware-focused vLLM fork for a dual RTX 2080 Ti runtime.

Target runtime:

- GPU: dual RTX 2080 Ti, SM75, tensor parallel size 2
- CUDA/PyTorch: CUDA 12.8, `torch 2.11.0+cu128`
- Base vLLM: `0.21.0`
- Main stable identity: `vllm-sm75-tp2-cu128`
- Current purpose: preserve the patched source, launch profiles, and runtime
  notes needed to reproduce the working 2080 Ti vLLM stack.

This fork is not a general upstream replacement. It carries local patches for
SM75/Turing serving, FlashQLA, FlashInfer, TurboQuant KV, INT8 KV continuation,
MTP/CUDAGraph safety, and Qwen/Gemma production profiles.

Important local docs:

- `docs/2080ti-stable/README-STABLE.md`: stable runtime contract and benchmark
  evidence.
- `docs/2080ti-stable/README-cu128.md`: CUDA 12.8 migration and validation log.
- `bench_tools/stable_profile.sh`: named model/profile environment presets.
- `bench_tools/remote_start_vllm_cu128.sh`: environment-driven launcher template.

Core feature matrix:

Legend: 🟢 stable/recommended, 🟡 usable with limits or experimental, 🔴 not a
recommended path.

| Feature | Qwen-family 27B | Gemma4-family 31B |
|---|---|---|
| AWQ Marlin | 🟢 supported | 🔴 not current route |
| GPTQ Marlin | 🟢 supported | 🟢 supported |
| MTP speculative decoding | 🟢 Native MTP | 🟡 external draft tested, not a speed route |
| `turboquant_4bit_nc` KV | 🟢 supported | 🟢 supported |
| `int8_per_token_head` KV | 🟡 supported for MTP/capacity experiments | 🟡 tested, not recommended default |
| Native 262K context | 🟢 baseline target, capacity validated | 🟡 needs separate validation |
| YaRN 524K extension | 🟡 capacity/offline experiment | 🔴 not validated |
| No-eager/CUDAGraph | 🟢 supported | 🟢 noMTP / 🟡 MTP |
| Fast prefill path | 🟢 FlashQLA + FlashInfer/FA2 | 🟢 FlashInfer/FA2 + SDPA512 fallback controls |

Notes:

- Specific checkpoints and weight formats are deployment-profile choices under
  each model family; they are not separate rows in this feature matrix.
- Native MTP means the target checkpoint itself contains MTP/draft tensors.
  External draft MTP is a separate assistant model passed through
  `SPECULATIVE_CONFIG`.
- Native 262K means model-declared `262144` context. This is the baseline, not
  an ultra-long-context feature.
- YaRN 524K means RoPE/YaRN extension beyond native context. It is currently a
  capacity/offline route, not the default interactive service mode.
- Some Qwen-family BF16 draft checkpoints require a launch compatibility flag
  so the draft model does not inherit the target quantization config.
- The key local fixes are SM75 FlashQLA/GDN prefill, FlashInfer/FA2 TurboQuant
  prefill, INT8 KV continuation/cascade dequant, TurboQuant + MTP CUDAGraph
  safety, and FlashInfer sampler warmup compatibility.

Representative validation evidence:

- Qwen-family 27B AWQ `MTP3 + TQ4NC`, PP4096/TG128:
  `1736.42 / 77.65 tok/s`, with stable JSON/tool-style generation.
- Qwen-family 27B AWQ `noMTP + TQ4NC`, native context:
  READY at `262144`, max concurrency `3.79x` for 262K requests.
- Qwen-family 27B GPTQ `INT8 KV + YaRN 524K`:
  32K prefill `1494.20 tok/s`; 520K prefill completed at `497.86 tok/s`
  in capacity smoke testing.

Throughput is always written as `prefill / decode tok/s`.

## Upstream vLLM README

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
