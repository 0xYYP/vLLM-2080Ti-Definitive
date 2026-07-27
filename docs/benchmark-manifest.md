# Benchmark Manifest Contract

The benchmark manifest records the identity and result state of an existing
benchmark artifact. It is an offline validation contract, not a benchmark
runner, a GPU correctness claim, or a performance claim.

## Schema And Status

Every manifest has `schema_version: 2`, a `status`, a `decision`, a `reason`,
and an `artifacts` list.
The permitted status values are `captured`, `not-run`, `rejected`, and `failed`.
Only `captured` may use `decision: accepted` or declare measured metrics. A
captured manifest must reference the semantic JSONL, vLLM benchmark JSONL,
`results.csv`, evaluated profile snapshot, source closure, and metadata TSV raw
artifact files with their `sha256` checksums. The validator recomputes each
checksum offline.

`not-run`, `rejected`, and `failed` describe a terminal state without a measured
throughput result. They use `decision: excluded`, require a machine-readable
reason, and must not declare metrics. A manifest may therefore record a failed
attempt without presenting it as benchmark evidence.

## Case And Provenance Identity

Captured schema v2 manifests bind the case ID and group, profile path, mode,
model family, quantization, KV dtype, MTP value, compatible modes, workload
tokens, warmup count, and measured-run count. The validator parses the hashed
profile snapshot and rejects metadata drift. Summary generation also compares
the manifest identity with `cases.tsv`.

Checkpoint, tokenizer, and served alias are three independent model identities.
The evaluator defaults tokenizer paths to their checkpoint paths for existing
profiles, but accepts separate `FP8_TOKENIZER_DIR` and `INT4_TOKENIZER_DIR`
values and records the actual values passed to the benchmark command.

Git HEAD, Python implementation/version, vLLM runtime version, and build
identity are mandatory provenance. The build identity is bound to the exact hashed
source closure artifact, so replacing a closure after capture invalidates the
manifest.

## Source And Naming Closure

The evaluator also writes a source closure manifest. It records each synchronized
source path and its `sha256`, allowing the receiving checkout to prove that the
validated source closure is byte-identical before it evaluates a result.
`EVAL_SYNC=0` is rejected because it would otherwise bypass this validation.

## Timing And Aggregation

Timeout is part of the recorded benchmark procedure: a timeout produces a failed
or incomplete outcome, not a successful measurement. Keep every raw artifact,
including warmup records, for auditability. The aggregate summary is
measured-only: only labels matching `.*-run[1-9][0-9]*` contribute to medians,
run counts, and published summary fields. Unlabeled rows and warmups remain raw
artifact evidence and never inflate or change measured-only metrics.

Run the offline validator with the repository runtime Python, for example:

```bash
.venv/bin/python tools/validate_benchmark_manifest.py path/to/artifact-manifest.json
```

This validation neither starts vLLM nor downloads models, accesses GPUs, or runs
the remote evaluator.
