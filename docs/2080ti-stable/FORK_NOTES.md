# VLLM-2080ti Fork Notes

Created on 2026-05-31 from the Miniclaw stable runtime work.

This repository is intended to preserve the reproducible source and launch
surface for the dual RTX 2080 Ti vLLM stack. It should not include model
weights, virtual environments, torch compile caches, benchmark result
directories, or local build artifacts.

Canonical runtime source:

- Stable runtime tree: `/data/stable/vllm-sm75-tp2-cu128`
- Base source snapshot: `/data/experiments/vllm-0.21.0-marlin-sm75-src`
- Patched runtime Python files copied from:
  `/data/stable/vllm-sm75-tp2-cu128/.venv/lib/python3.11/site-packages/vllm`
- Launcher/profile scripts copied from:
  `/data/stable/vllm-sm75-tp2-cu128/bench_tools`

Important copied patch files:

- `vllm/v1/attention/backends/turboquant_attn.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/utils/flashinfer.py`

Current branch policy:

- `sm75-tp2-cu128-stable` is the stable 2080 Ti / CUDA 12.8 branch.
- Tags should use explicit runtime names, for example
  `miniclaw-sm75-tp2-cu128-20260531`.
- Upstream vLLM changes should be merged deliberately, not casually rebased over
  the production branch.

External repository policy:

- GitHub repository name: `VLLM-2080ti`.
- Create the remote only after deciding public vs private visibility.
- Keep model paths and performance reports in docs; keep large weights and
  machine-local caches out of git.
