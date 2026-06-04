# Changelog

This changelog tracks the fork release version for vLLM 2080 Ti Definitive
Edition. It is separate from the upstream vLLM package version.

## v0.1.2 - 2026-06-04

- Current public stable snapshot for the SM75 TP=2 CUDA 12.8 runtime.
- Includes the first three public fix rounds after the initial release.
- Keeps the upstream vLLM base at `0.21.0` while versioning this fork as an
  independent 2080 Ti runtime distribution.
- Updates the documented Qwen3.6 and Gemma4 runtime routes, tested checkpoint
  list, launcher profile guidance, and benchmark evidence links.

## v0.1.1

- Follow-up compatibility fixes for editable/source builds and optional CUDA
  extension imports on SM75 environments.

## v0.1.0

- Initial public stable snapshot of the dual 2080 Ti / SM75 TP=2 runtime.
- Published the stable runtime identity, launcher profiles, Qwen3.6 route,
  Gemma4 experimental route, and baseline benchmark documentation.
