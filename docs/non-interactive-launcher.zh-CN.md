# 非交互 Launcher

`launcher.sh` 现在支持完整的非交互启动路径，可用于脚本部署、CI 校验和可复现的
route 验证。

最终优先级为：

`CLI > ENV > PROFILE > default`

其中 `CLI` 包括直接参数（如 `--max-model-len 262144`）、`--set KEY=VALUE`
以及 `--unset KEY`。

## 参数命名

已纳入 launcher 管理面的配置，可以直接写成 `--lower-kebab-case`：

- `MODEL_DIR` -> `--model-dir`
- `PROFILE` -> `--profile`
- `MAX_MODEL_LEN` -> `--max-model-len`
- `MM_LIMIT_JSON` -> `--mm-limit-json`

对于不在直接 launcher 参数面上的高级环境变量，使用：

- `--set KEY=VALUE`
- `--unset KEY`

`--unset KEY` 会清掉继承自 profile 或环境变量的值，然后回落到更低优先级的
launcher 默认层。如果你需要保留“显式空字符串”而不是回落默认值，使用
`--set KEY=`。
例如，`--unset COMPILATION_CONFIG_JSON` 会选择 generated compilation config，而不会把
已经清掉的值当成非法的显式 JSON 输入。

## 常见示例

只打印最终启动摘要，不真正启动服务：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env
```

基于仓库内置 profile 启动，并只覆写少量字段：

```bash
./launcher.sh --non-interactive \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --mode normal \
  --service-scope lan \
  --max-model-len 196608 \
  --gpu-devices 0,1
```

环境变量仍然可用，但 CLI 优先级更高：

```bash
MAX_MODEL_LEN=131072 \
MODE=fast \
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --max-model-len 262144
```

用 `--set` 传递高级运行时环境变量：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set VLLM_TURBOQUANT_FLASHINFER_BACKEND=fa2 \
  --set CC=/usr/bin/gcc-12
```

清掉 profile 提供的字段，并让 launcher 回填更低层默认值：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --unset max-model-len
```

强制保留显式空值：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set REASONING_PARSER=
```

## 派生默认值

profile 加载后，launcher 仍会对部分字段做规范化：

- `MESSAGE_TYPE`、`MM_LIMIT_JSON`、`LANGUAGE_MODEL_ONLY`、
  `SKIP_MM_PROFILING`
- 模式派生运行时参数，如 `ENFORCE_EAGER`、
  `VLLM_SM75_SPEC_SYNC_MODE`、`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH`
- 已验证 Qwen3 路线上默认补齐的 `REASONING_PARSER=qwen3`
- Qwen prefix-cache 路线默认补齐的 `MAMBA_CACHE_MODE=align`

这些默认值只会补空缺字段，不会再反向覆盖显式 CLI 输入。

## 推荐校验方式

调整脚本或部署自动化时，优先先跑 `--print-config`。它会打印最终生效的启动摘要，
并在真正启动 vLLM 前退出，是确认每个覆写项是否真正落到最终配置上的最稳妥方式。

## 机器可读的解析契约

`--print-config` 会为影响最终启动的值输出 `resolved_<key>=<value>` 和
`resolved_<key>_source=<source>` 字段。source 只会是 `cli`、`env`、`profile`、
`default`、`generated` 或 `derived`。值仍遵守 `CLI > ENV > PROFILE > default`；
`generated` 或 `derived` 只会补齐上述层级没有提供的值。

`COMPILATION_CONFIG_JSON` 是边界值，不是可任意拼接的 launcher 文本。它必须是
JSON object。launcher 会在启动 server 前拒绝 malformed JSON、array、scalar 以及显式
empty 值。合法 object 会以 canonical JSON 形式输出，因此语义相同的输入有稳定的
机器可读输出。显式选择却不存在的 missing profile 是错误，绝不会静默回退到 default。

`GPU_DEVICES` 只接受以逗号分隔的 numeric、non-negative GPU ID 列表。numeric 项目两侧
的空白会被规范化。empty 输入或 empty 项、duplicate ID、UUID 值以及 MIG 值都会被拒绝。
UUID 和 MIG 设备选择有意不属于这份 numeric CPU contract。

未提供 `TP_SIZE` 时，launcher 会根据已解析 GPU ID 的数量 derived 出 TP。显式提供时，
`TP_SIZE` 必须是等于该数量的 positive integer。resolved 字段会标明 TP 是显式提供还是
derived。

`final_vllm_argv` 是经 shell 转义、完成全部规范化后原本将传给 vLLM 的最终 argument
vector。`--print-config` 会输出它；该命令不启动 service、不发送 network request、不执行
model download，也不进行 GPU probing。
