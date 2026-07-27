# Benchmark Manifest 契约

benchmark manifest 记录既有 benchmark artifact 的身份和结果状态。它是离线校验契约，
不是 benchmark runner，不代表 GPU correctness，也不代表 performance claim。

## Schema 与 Status

每个 manifest 都包含 `schema_version: 2`、`status`、`decision`、`reason` 和
`artifacts` 列表。允许的 status 为 `captured`、`not-run`、`rejected` 和 `failed`。
只有 `captured` 能使用 `decision: accepted` 并声明 measured metrics。captured manifest
必须以 `sha256` checksum 引用 semantic JSONL、vLLM benchmark JSONL、`results.csv`、
实际评测 profile snapshot、source closure 和 metadata TSV raw artifact。validator 会
离线重新计算每个 checksum。

`not-run`、`rejected` 和 `failed` 描述没有 measured throughput result 的终态，必须使用
`decision: excluded`、提供机器可读 reason，且不能声明 metrics。因此 manifest 可以记录
失败尝试，但不会把它表示为 benchmark evidence。

## Case 与 Provenance 身份

captured schema v2 manifest 会绑定 case ID/group、profile path、mode、model family、
quantization、KV dtype、MTP、compatible modes、workload tokens、warmup 数和 measured-run
数。validator 会解析已哈希的 profile snapshot，profile metadata 漂移会被拒绝；生成
summary 时还会把 manifest identity 与 `cases.tsv` 对照。

checkpoint、tokenizer 和 served alias 是三个独立的 model identity。现有 profile 默认让
tokenizer path 等于 checkpoint path，但 evaluator 支持独立的 `FP8_TOKENIZER_DIR` 和
`INT4_TOKENIZER_DIR`，并记录 benchmark 命令实际收到的值。

Git HEAD、Python implementation/version、vLLM runtime version 和 build identity 都是必填
provenance。build identity 与精确的 source closure artifact 哈希绑定，capture 后替换
closure 会使 manifest 失效。

## Source 与 Naming Closure

evaluator 还会写出 source closure manifest。它记录每个同步 source path 及其 `sha256`，
这样接收端 checkout 可在评估结果前证明已校验的 source closure 按字节一致。
`EVAL_SYNC=0` 会被明确拒绝，因为否则会绕过这项验证。

## Timeout 与 Aggregation

timeout 是已记录 benchmark procedure 的一部分：timeout 会生成 failed 或 incomplete
outcome，而不是成功测量。必须保留全部 raw artifacts，包括 warmup records，以便审计。
aggregate summary 是 measured-only：只有匹配 `.*-run[1-9][0-9]*` 的标签才参与 median、
run count 和发布 summary fields。无标签行和 warmup records 仍是 raw artifact evidence，
但绝不会增加或改变 measured-only metrics。

使用仓库 runtime Python 运行 offline validator，例如：

```bash
.venv/bin/python tools/validate_benchmark_manifest.py path/to/artifact-manifest.json
```

该校验不会启动 vLLM，也不会下载 model、访问 GPU 或运行 remote evaluator。
