# vLLM 2080 Ti Definitive Edition Fork Notes

Created on 2026-05-31 from the stable dual RTX 2080 Ti runtime work.

This repository is intended to preserve the reproducible source and launch
surface for the dual RTX 2080 Ti vLLM stack. It should not include model
weights, virtual environments, torch compile caches, benchmark result
directories, or local build artifacts.

Canonical runtime source:

- Stable runtime tree: `<STABLE_ROOT>`
- Base source snapshot: `<BASE_SOURCE_ROOT>`
- Patched runtime Python files copied from:
  `<PATCHED_SITE_PACKAGES>/vllm`
- Launcher/profile scripts copied from:
  `<STABLE_ROOT>/bench_tools`

Important copied patch files:

- `vllm/v1/attention/backends/turboquant_attn.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/utils/flashinfer.py`

Current branch policy:

- `sm75-tp2-cu128-stable` is the stable 2080 Ti / CUDA 12.8 branch.
- Tags should use explicit runtime names, for example
  `sm75-tp2-cu128-20260531`.
- Upstream vLLM changes should be merged deliberately, not casually rebased over
  the production branch.

External repository policy:

- GitHub repository name: `vllm-2080ti-definitive`.
- Create the remote only after deciding public vs private visibility.
- Use placeholders for model paths and report paths in public docs; keep large
  weights, benchmark results, machine-local caches, hostnames, service names,
  and absolute deployment paths out of git.
