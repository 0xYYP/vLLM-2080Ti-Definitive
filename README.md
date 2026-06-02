<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition cover](docs/assets/vllm-2080ti-cover.jpg)

![Live single-request speed demo](docs/assets/vllmspeed.gif)

The definitive vLLM runtime for dual RTX 2080 Ti / SM75 serving.

This is a hardware-focused fork that preserves the patched source, launch
profiles, and runtime notes needed to reproduce the working 2080 Ti vLLM stack.

Headline evidence: Qwen3.6 27B reaches `100+ tok/s` single-request decode, and
Gemma4 31B reaches `~100 tok/s` single-request decode on the same dual 2080 Ti
TP=2 runtime.

Language: English | [简体中文](README.zh-CN.md)

## 💡 Why RTX 2080 Ti for LLM Inference?

In August 2018, NVIDIA launched the RTX 2080 Ti and moved the enthusiast GPU
line from GTX into the RTX era. Years later, the card is still remembered as a
landmark Turing design. With 22GB memory mods, NVLink, high memory bandwidth,
and enough raw compute to remain relevant, dual 2080 Ti cards turn out to be a
surprisingly strong local AI inference platform.

| Metric | 2x 2080 Ti 22GB + NVLink | 3090 Ti 24GB baseline | Ratio |
|---|---:|---:|---:|
| Physical CUDA core count | 8,704 | 5,376 | 1.62x |
| SM count | 136 | 84 | 1.62x |
| Physical Tensor Core count | 1,088 | 336 | 3.24x |
| Dense Tensor FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| Total physical memory bandwidth | 1,232 GB/s | 1,008 GB/s | 1.22x |
| Total VRAM capacity | 44GB | 24GB | 1.83x |
| Secondary-market price anchor | about $550 with NVLink | about $1,100 | about 0.5x |

The project is built around a simple cost/performance bet: use roughly half the
secondary-market price of an RTX 3090 Ti to get a dual 22GB RTX 2080 Ti setup
that can match or exceed it on the physical resources that matter for LLM
serving, then use vLLM runtime work to turn those resources into real tokens.

That is the first value of this fork: take old but strong Turing silicon and
make it behave like a serious 27B/31B-class inference platform through Marlin,
FlashQLA/FlashInfer/FA2, TurboQuant/INT8 KV, MTP, and CUDAGraph integration.

## 🚀 MTP Is Acceptance-Bound

MTP/speculative decoding is not a fixed multiplier. Its speedup depends on how
many drafted tokens the target model accepts, so the best `MTP_K` changes with
task type. In our sweeps, code and scientific-analysis prompts benefited much
more than natural prose; very high K values could win synthetic throughput tests
but regress on realistic long-generation prompts.

For the current stable profiles, Qwen3.6 uses MTP3 as the conservative mixed
agent default, while Gemma4 uses higher MTP values only for specific FP16
speed-oriented profiles. See [MTP Task Sensitivity](docs/mtp-task-sensitivity.md)
for the measured Qwen/Gemma sweep tables.

## 🧩 Core Routes

Serving shape:

- This project optimizes for extreme single-concurrency performance on dual
  2080 Ti: one personal-agent style workload, one serious 27B/31B model, and
  the largest practical context window this hardware can sustain.
- It is not a multi-tenant serving stack. Multi-agent use is supported best as
  queued workspace isolation, not as parallel long-prefill throughput. Long
  prefill work is capacity-safe when tuned, but it is effectively serialized by
  the runtime scheduler on this TP=2 profile.

Status: 🟢 full support; 🟡 partial support; 🔴 performance regression; ⚪ not
supported.

### Qwen3.6 27B Mature Route

Qwen-family 27B is the primary production route for this fork. It has the most
complete coverage across Marlin weights, native MTP, FP16/INT8/TQ4NC KV, 262K
native context, YaRN capacity experiments, and image-serving compatibility.
Fast path: Qwen uses FlashQLA-SM70-SM75 for Gated DeltaNet / linear-attention
prefill, FlashInfer / FA2 for full-attention prefill, head_dim=256 fast-path
controls, and native MTP with CUDAGraph for decode.

| Feature | FP16 KV | INT8 KV | TQ4NC KV |
|---|---|---|---|
| Marlin weight route | 🟢 AWQ/GPTQ | 🟢 AWQ/GPTQ | 🟢 AWQ/GPTQ |
| Native MTP3 decoding | 🟢 short-context speed route | 🟢 capacity + speed route | 🟢 compressed-capacity route |
| Native 262K context | 🟢 noMTP real prompt supported | 🟡 capacity/speed candidate | 🟢 real prompt/service supported |
| YaRN 524K extension | ⚪ not the target route | 🟢 supported capacity route | 🟡 capacity candidate |
| No-eager / CUDAGraph | 🟢 supported | 🟢 supported | 🟢 graph-safety fixed |
| Fast prefill path | 🟢 FlashInfer / FA2 | 🟢 FlashInfer / INT8 path | 🟢 TurboQuant FlashInfer path |
| Multimodal image serving | 🟢 default-KV route | 🔴 output corruption observed | 🟢 recommended image route |
| Peak MTP3 PP4096/TG128 | 🟢 1747.52 / 100.98 tok/s | 🟢 1744.06 / 81.12 tok/s | 🟢 1746.32 / 85.94 tok/s |

Larger MTP values can produce higher synthetic throughput-only numbers: Qwen3.6
TQ4NC reached `90.75 tok/s` in a stable MTP5 row and `100.61 tok/s` in an
earlier candidate row. MTP3 is the practical deployment reference because it
balances acceptance rate and real workload throughput.

For interactive single-user chat, non-TQ FP16/default KV + native MTP5 is also
kept as a convenient speed-feel profile; it is useful for manual latency checks,
but it does not replace MTP3 as the conservative deployment reference.

Recommended Qwen profiles:

| Use case | KV precision | Context | Spec decoding | Message type | Concurrency limit |
|---|---|---|---|---|---|
| High-quality native-context route | FP16 | 262K native | None | text | 1 request |
| Peak short-context speed route | FP16 | 8K-16K | Native MTP3 | text | 1 request |
| High-compression route | TQ4NC | 262K native | Native MTP3 | text | 1 request / queued |
| Ultra-long context | INT8 | 524K YaRN | Native MTP3 | text | 1 offline request |
| Multi-workspace | INT8 or TQ4NC | 64K-262K caps | Native MTP3 | text | 4 x 64K queued / 2 x 262K queued |
| Multimodal | TQ4NC | 262K native | Native MTP3 | text + image | 1 request |

Qwen limits:

- INT8 KV is a text-serving capacity route. It is not recommended for image
  serving because the validated multimodal runs reached READY but produced
  corrupted punctuation/output instead of stable image answers.
- FP16/default KV has a real `PP262000/TG1` pass only in noMTP mode. The MTP3
  262K service can start, but the real 262K prompt OOMs during prefill, so MTP3
  stays a short-context speed route for FP16.
- The multi-workspace profile is for queued workspace isolation, not true
  parallel long-prefill throughput. This TP=2 runtime still serializes heavy
  long-context work in practice.
- YaRN 524K is an offline capacity profile. The native 262K profiles remain the
  default for normal interactive serving.

Common Qwen launcher presets:

- `qwen27-awq-mtp3`: regular Qwen-family FP16/default KV + native MTP3 route.
- `qwen27-awq-mtp3-peak`: short-context peak-speed text route.
- `qwen27-awq-mtp3-tq4nc-fi`: TQ4NC compressed-capacity route.
- `qwen27-awq-mtp3-int8kv`: INT8 KV capacity / YaRN route.
- `qwen27-awq-mm-*` and `heretic27-gptq-mm-*`: image-serving experiment
  presets for Qwen-family model variants.

### Gemma4 31B Experimental Route

Gemma4 31B is kept as a secondary experimental route. The FP16/default-KV path
is fast and useful, but Gemma's head_dim=512 and heterogeneous/GQA attention
make compressed-KV and long-context routes much less mature than Qwen.
Fast path: Gemma uses the default-KV fast route for short-context FP16 service;
compressed long-context paths still fall back to SDPA/GQA compatibility, and
assistant MTP is compatible but more workload-sensitive.

| Feature | FP16 KV | INT8 KV | TQ4NC KV |
|---|---|---|---|
| Marlin weight route | 🟢 GPTQ target | 🟢 GPTQ target | 🟢 GPTQ target |
| Assistant MTP decoding | 🟢 MTP5 peak route | 🔴 experimental | 🔴 MTP regression |
| Native 262K context | ⚪ capacity-negative | 🟡 slow offline route | ⚪ not enough practical capacity |
| YaRN 524K extension | ⚪ not supported | ⚪ not supported | ⚪ not supported |
| No-eager / CUDAGraph | 🟢 supported | 🔴 fallback-heavy | 🟢 repaired for compatibility |
| Fast prefill path | 🟢 default-KV fast path | 🔴 SDPA/GQA fallback cost | 🟡 short-context fast, long-context limited |
| Multimodal image serving | 🟢 default-KV route | ⚪ not supported | ⚪ not supported |
| Peak PP4096/TG128 | 🟢 MTP5 1655.65 / 99.64 tok/s | 🔴 not a serving profile | 🟡 noMTP 1596.15 / 31.70 tok/s |

Recommended Gemma profiles:

| Use case | KV precision | Context | Spec decoding | Message type | Concurrency limit |
|---|---|---|---|---|---|
| High-quality route | FP16 | 16K validated text service; 105K estimate only | Assistant MTP5 | text | 1 request |
| Fast compressed route | TQ4NC | 43K real-prompt practical edge | None | text | 1 request |
| Long-context route | INT8 | 262K native, slow offline | None | text | 1 slow offline request |
| Multimodal compatibility | FP16 | 8K image validated | Assistant MTP3 compatible | text + image | 1 request |

Gemma limits:

- Gemma4 uses head_dim=512 with heterogeneous/GQA attention. On SM75, the
  compressed-KV path does not have a validated FlashAttention/FlashInfer fast
  prefill route; the 262K INT8 path currently falls back to SDPA/GQA and is
  much slower than the FP16/default-KV route.
- The practical Gemma context window is much smaller than Qwen on the same dual
  22GB hardware. The combination of head_dim=512, GQA/heterogeneous attention,
  and less efficient compressed-KV grouping means that starting a compressed-KV
  profile is not the same as getting an efficient full-262K service profile.
- FP16/default KV is the best speed and quality route, but it cannot reach the
  full native 262K context on dual 22GB cards. The validated service profile is
  16K; the `105216` text and `97152` image figures are startup estimates from
  failed 262K probes, not proven practical request limits.
- TQ4NC is kept as a fast compressed short-context route. The useful real-prompt
  edge is about `43K`: a `43005`-token prompt passed, while `43505`/`44005`
  failed admission. `64K` is READY-only evidence, not a proven long-prompt pass.
- Gemma multimodal is kept on default KV. INT8/TQ4NC multimodal routes are not
  recommended because the heterogeneous-head multimodal backend rejects or
  breaks those compressed KV paths.

Common Gemma launcher presets:

- `gemma4-gptq-tq4nc-mtp3`: TQ4NC compatibility route with assistant MTP3.
- `gemma4-gptq-tq4nc-nomtp`: TQ4NC no-MTP short-context route.
- `gemma4-gptq-mm-nomtp`: default-KV image-serving compatibility route.
- `gemma4-gptq-mm-mtp3`: default-KV image-serving route with assistant MTP3.

## 🛠️ Target Hardware & Runtime

- Validated GPU profile: dual RTX 2080 Ti 22GB, SM75, NVLink, tensor parallel
  size 2
- CUDA/PyTorch: CUDA 12.8, `torch 2.11.0+cu128`
- Base vLLM: `0.21.0`
- Repository identity: `vllm-2080ti-definitive`
- Main stable runtime identity: `vllm-sm75-tp2-cu128`
- Compatibility target: NVIDIA Turing / SM75 GPUs. Other Turing cards still
  need profile validation for VRAM capacity, P2P/NVLink behavior, model
  head_dim, KV dtype, and CUDAGraph/MTP settings.

## 🚀 How To Use

This fork is intended to be used as a pinned runtime tree on a validated SM75
host, not as a drop-in `pip install vllm` replacement.

1. Put the patched runtime on the target machine and keep the stable root
   stable. Use your own install path:

   ```bash
   export STABLE_ROOT=/path/to/vllm-sm75-tp2-cu128
   ```

2. Put model checkpoints under a stable model root. The launcher expects profile
   aliases such as:

   ```text
   $MODEL_ROOT/qwen-family-27b-awq
   $MODEL_ROOT/gemma4-family-31b-gptq
   $MODEL_ROOT/gemma4-family-assistant
   ```

   You can also override `MODEL_DIR` directly when using a different checkpoint.

3. Start one of the validated profiles:

   ```bash
   cd "$STABLE_ROOT"

   PROFILE=qwen27-awq-mtp3-peak \
   MODEL_ROOT=/path/to/models \
   PORT=8000 \
   CUDA_VISIBLE_DEVICES=1,2 \
   RUN_USER=vllm \
   RUN_HOME=/var/lib/vllm \
   ./bench_tools/stable_profile.sh
   ```

4. Check the OpenAI-compatible endpoint:

   ```bash
   curl http://127.0.0.1:8000/v1/models
   ```

Common profiles:

- `qwen27-awq-mtp3-peak`: Qwen-family FP16/default KV peak-speed text route.
- `qwen27-awq-mtp3-tq4nc-fi`: Qwen-family TQ4NC compressed-capacity route.
- `qwen27-awq-mtp3-int8kv`: Qwen-family INT8 KV capacity/YaRN route.
- `gemma4-gptq-tq4nc-mtp3`: Gemma4 TQ4NC compatibility route.
- `gemma4-gptq-tq4nc-nomtp`: Gemma4 TQ4NC no-MTP short-context route.

## ❓ Hardware Q&A

**Q: What GPU interconnect is required?**

A: NVLink is recommended, but PCIe P2P is the real baseline requirement. The
validated system uses NVLink and an intentionally non-ideal PCIe topology, with
one card at PCIe 3.0 x1 and the other at PCIe 3.0 x4. With NVLink carrying
GPU-to-GPU traffic, PCIe slot bandwidth is not the main bottleneck. Without
NVLink, do not treat narrow PCIe links as proven sufficient; confirm P2P
behavior and benchmark the actual topology.

**Q: Does the host need a strong CPU or a lot of RAM?**

A: No. The validated path has run on a low-end desktop CPU with 16GB RAM. More
CPU/RAM mainly helps compile cache generation, downloads, and local build work,
not steady-state token generation.

**Q: Which Turing GPUs make sense? Can I mix 11GB and 22GB cards?**

A: The fully validated target is dual RTX 2080 Ti 22GB. Other good candidates
are high-VRAM TU102-class cards: TITAN RTX 24GB, Quadro RTX 6000 24GB, and
Quadro RTX 8000 48GB, preferably in pairs with NVLink or confirmed PCIe P2P.
Mixed 11GB + 22GB RTX 2080 Ti setups are not recommended for these 27B/31B
profiles because vLLM TP=2 is effectively constrained by the smaller rank.
Smaller Turing cards can run smaller models, but they are not the main target
for this stack.

**Q: Which CUDA, PyTorch, and driver versions are validated?**

A: The stable runtime is CUDA 12.8 + `torch 2.11.0+cu128`. Use a recent NVIDIA
driver that supports your host GPUs and is compatible with the CUDA runtime. Do
not mix build/runtime assumptions casually: keep the PyTorch CUDA lane, local
CUDA toolkit, FlashInfer/FlashQLA builds, and launch profile aligned.

**Q: What other hardware risks matter?**

A: Cooling, power stability, and enough SSD space for model files and compile
caches. Thermal throttling can hide as a software regression, especially during
long prefill or repeated CUDAGraph/AOT compilation runs.

## 🔗 Related Project

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox): companion
  toolbox for dual 2080 Ti model routes, benchmark summaries, model notes, and
  operational guidance. This repository focuses on the patched vLLM runtime
  itself.

## 🙏 Credits / Upstream Projects

This repository is a hardware-focused fork of upstream
[vLLM](https://github.com/vllm-project/vllm), licensed under Apache-2.0. The
fork keeps the upstream project structure and adds local SM75 runtime patches,
launch profiles, and validation notes for the dual 2080 Ti route.

Acceleration components used or integrated by the stable runtime include:

- [vLLM](https://github.com/vllm-project/vllm): base inference engine and
  serving stack.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): attention,
  sampling, and quantized kernel paths used by vLLM.
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA): upstream FlashQLA
  Gated DeltaNet / Qwen3.5 linear-attention implementation.
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75):
  SM70/SM75 adaptation used by the stable Qwen3.6 prefill profile.
- FlashAttention / FA2, TurboQuant, Marlin, CUTLASS, Triton, and related vLLM
  acceleration kernels: existing open-source acceleration work integrated and
  profiled for this hardware target.
