# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_VALIDATOR = _ROOT / "tools/validate_benchmark_manifest.py"
_EVALUATOR = _ROOT / "tools/evaluate_fast_modes.sh"
_REPOSITORY_PYTHON = _ROOT / ".venv/bin/python"
_VALIDATOR_PYTHON = (
    _REPOSITORY_PYTHON if _REPOSITORY_PYTHON.is_file() else Path(sys.executable)
)
_TEST_GIT_HEAD = "1" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_validator(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_VALIDATOR_PYTHON), str(_VALIDATOR), *arguments],
        check=False,
        text=True,
        capture_output=True,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_executable(path: Path, content: str) -> None:
    lines = content.splitlines()
    if lines[0] == "#!/usr/bin/env sh":
        lines[0] = "#!/bin/sh"
    path.write_text(
        lines[0] + "\n" + textwrap.dedent("\n".join(lines[1:])).lstrip() + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fake_remote_environment(
    tmp_path: Path,
    hang_profile: bool = False,
    warmups: str = "0",
    distinct_warmup: bool = False,
    prompt_tokens: str = "4096",
    gen_tokens: str = "128",
) -> dict[str, str]:
    timeout = shutil.which("timeout")
    assert timeout is not None
    fake_bin = tmp_path / "bin"
    remote_root = tmp_path / "remote"
    models = tmp_path / "models"
    fake_bin.mkdir()
    remote_root.mkdir(exist_ok=True)
    models.mkdir()
    model_dir = models / "checkpoint"
    model_dir.mkdir()
    tokenizer_dir = models / "tokenizer"
    tokenizer_dir.mkdir()
    arguments = tmp_path / "bench-arguments.json"
    launcher = tmp_path / "launcher.sh"
    python = fake_bin / "python"
    _write_executable(
        fake_bin / "runuser",
        """#!/usr/bin/env sh
        if [ "$1" = "-u" ]; then shift 2; fi
        if [ "$1" = "--" ]; then shift; fi
        if [ "$1" = "$REMOTE_ROOT/.venv/bin/python" ]; then
          shift
          exec "$FAKE_PYTHON" "$@"
        fi
        exec "$@"
        """,
    )
    _write_executable(
        fake_bin / "sleep",
        """#!/usr/bin/env sh
        exit 0
        """,
    )
    _write_executable(
        fake_bin / "timeout",
        """#!/usr/bin/env sh
        signal=TERM
        kill_after=10
        while [ "$#" -gt 0 ]; do
          case "$1" in
            --signal=*) signal=${1#*=}; shift ;;
            --kill-after=*) kill_after=${1#*=}; shift ;;
            *) break ;;
          esac
        done
        duration=$1
        shift
        if [ "$1" = "$REMOTE_ROOT/.venv/bin/python" ]; then
          shift
          exec "$FAKE_TIMEOUT" --signal="$signal" \\
            --kill-after="$kill_after" "$duration" "$FAKE_PYTHON" "$@"
        fi
        exec "$FAKE_TIMEOUT" --signal="$signal" \\
          --kill-after="$kill_after" "$duration" "$@"
        """,
    )
    _write_executable(
        fake_bin / "pgrep",
        """#!/usr/bin/env sh
        exit 1
        """,
    )
    _write_executable(
        fake_bin / "chown",
        """#!/usr/bin/env sh
        exit 0
        """,
    )
    _write_executable(
        fake_bin / "install",
        """#!/usr/bin/env sh
        while [ "$#" -gt 2 ]; do shift; done
        mkdir -p "$(dirname "$2")"
        cp "$1" "$2"
        """,
    )
    _write_executable(
        fake_bin / "sed",
        """#!/usr/bin/env sh
        if [ "$1" = "-i" ]; then
          shift
          if /usr/bin/sed --version >/dev/null 2>&1; then
            exec /usr/bin/sed -i "$@"
          fi
          exec /usr/bin/sed -i '' "$@"
        fi
        exec /usr/bin/sed "$@"
        """,
    )
    _write_executable(
        launcher,
        """#!/usr/bin/env sh
        exit "${FAKE_LAUNCH_EXIT:-0}"
        """,
    )
    _write_executable(
        python,
        """#!/usr/bin/env sh
        if [ "$1" = "-" ]; then
          content=$(cat)
          case "$content" in
            *sysconfig.get_paths*) printf '%s\\n' "$REMOTE_ROOT/fake-site"; exit 0 ;;
          esac
          printf '%s' "$content" | exec "$FAKE_REAL_PY" "$@"
        fi
        if [ "$1" = "-m" ] && [ "$2" = "vllm" ]; then
          shift 3
          result_dir= result_name= model= tokenizer= served=
          while [ "$#" -gt 0 ]; do
            case "$1" in
              --result-dir) result_dir=$2; shift 2 ;;
              --result-filename) result_name=$2; shift 2 ;;
              --model) model=$2; shift 2 ;;
              --tokenizer) tokenizer=$2; shift 2 ;;
              --served-model-name) served=$2; shift 2 ;;
              *) shift ;;
            esac
          done
          printf '{"model":"%s","tokenizer":"%s","served":"%s"}\\n' \\
            "$model" "$tokenizer" "$served" > "$FAKE_ARGUMENTS"
          printf '{"output_throughput": 80.0}\\n' >> "$result_dir/$result_name"
          exit 0
        fi
        if [ "${1##*/}" = "profile_request.py" ]; then
          if [ "$FAKE_PROFILE_HANG" = "1" ]; then /bin/sleep 30; exit 0; fi
          out= label=
          while [ "$#" -gt 0 ]; do
            case "$1" in
              --out) out=$2; shift 2 ;;
              --label) label=$2; shift 2 ;;
              *) shift ;;
            esac
          done
          if [ "$FAKE_DISTINCT_WARMUP" = "1" ] && \
             case "$label" in *-warmup*) true ;; *) false ;; esac; then
            prefix='{"label":"'
            suffix='","content_sample":"drift","chunks":99,'
            suffix="$suffix"'"decode_tok_s":1.0,"prefill_tok_s":2.0}'
          else
            prefix='{"label":"'
            suffix='","content_sample":" the the the the","chunks":6,'
            suffix="$suffix"'"decode_tok_s":80.0,"prefill_tok_s":40.0}'
          fi
          printf '%s%s%s\\n' "$prefix" "$label" "$suffix" >> "$out"
          exit 0
        fi
        exec "$FAKE_REAL_PY" "$@"
        """,
    )
    _write_executable(
        fake_bin / "rsync",
        """#!/usr/bin/env python3
        import os
        import shutil
        import subprocess
        import sys

        arguments = sys.argv[1:]
        destination = arguments[-1].split(":", 1)[-1]
        result = subprocess.run(
            ["/usr/bin/rsync", *arguments[:-1], destination], check=False
        )
        raise SystemExit(result.returncode)
        """,
    )
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env sh
        if [ "$1" = "-o" ]; then shift 2; fi
        shift
        exec /bin/sh -c "$1"
        """,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "REMOTE_HOST": "fake-host",
            "REMOTE_ROOT": str(remote_root),
            "REMOTE_USER": os.environ.get("USER", "yyp"),
            "REMOTE_RESULT_DIR": str(remote_root / "results"),
            "FP8_MODEL_DIR": str(model_dir),
            "INT4_MODEL_DIR": str(model_dir),
            "FP8_TOKENIZER_DIR": str(tokenizer_dir),
            "INT4_TOKENIZER_DIR": str(tokenizer_dir),
            "CASE_FILTER": "fp8_int8kv",
            "FAST_ONLY": "1",
            "FAST_COMPARE_REQUIRED": "0",
            "MEASURED_RUNS": "1",
            "WARMUPS": warmups,
            "PROMPT_TOKENS": prompt_tokens,
            "GEN_TOKENS": gen_tokens,
            "BENCHMARK_TIMEOUT_SEC": "1",
            "FAKE_REAL_PY": str(_VALIDATOR_PYTHON),
            "FAKE_TIMEOUT": timeout,
            "FAKE_PYTHON": str(python),
            "FAKE_ARGUMENTS": str(arguments),
            "EVAL_LAUNCHER": str(launcher),
            "FAKE_PROFILE_HANG": "1" if hang_profile else "0",
            "FAKE_DISTINCT_WARMUP": "1" if distinct_warmup else "0",
        }
    )
    environment["REMOTE_ROOT"] = str(remote_root)
    return environment


def _run_fake_evaluator(
    tmp_path: Path,
    hang_profile: bool = False,
    warmups: str = "0",
    distinct_warmup: bool = False,
    prompt_tokens: str = "4096",
    gen_tokens: str = "128",
    eval_sync: str = "1",
) -> subprocess.CompletedProcess[str]:
    environment = _fake_remote_environment(
        tmp_path,
        hang_profile,
        warmups,
        distinct_warmup,
        prompt_tokens,
        gen_tokens,
    )
    environment["EVAL_SYNC"] = eval_sync
    return subprocess.run(
        ["bash", str(_EVALUATOR)],
        check=False,
        text=True,
        capture_output=True,
        cwd=_ROOT,
        env=environment,
        timeout=20,
    )


def _write_captured_manifest(
    case_dir: Path,
    *,
    result_name: str = "results.jsonl",
    semantic_text: str,
    csv_text: str,
    metrics: dict[str, object],
    group: str = "captured",
    mode: str = "fast",
    role: str = "fast_guard",
    threshold: float = 0.0,
    profile_path: str = "profile.env",
    checkpoint: str = "model",
    tokenizer: str = "model",
    prompt_tokens: int = 4096,
    generation_tokens: int = 128,
    warmups: int = 0,
    measured_runs: int = 1,
) -> Path:
    result = case_dir / result_name
    result.write_text(semantic_text, encoding="utf-8")
    csv = case_dir / "results.csv"
    csv.write_text(csv_text, encoding="utf-8")
    bench = case_dir / "bench-results.jsonl"
    bench.write_text('{"output_throughput": 12.5}\n', encoding="utf-8")
    profile = case_dir / "profile.env"
    profile.write_text(
        "SERVED_NAME=test-served-alias\n"
        "COMPATIBLE_MODES=fast,normal,safe\n"
        "MODEL_FAMILY=qwen\n"
        "QUANTIZATION=fp8\n"
        "KV_CACHE_DTYPE=int8_per_token_head\n"
        "MTP_K=3\n",
        encoding="utf-8",
    )
    closure = case_dir / "source-closure-manifest.json"
    _write_json(
        closure,
        {
            "schema_version": 1,
            "files": [{"path": "launcher.sh", "sha256": "2" * 64}],
        },
    )
    metadata = case_dir / "meta.tsv"
    metadata.write_text(f"group\t{group}\nmode\t{mode}\n", encoding="utf-8")
    manifest = case_dir / "artifact-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 2,
            "status": "captured",
            "decision": "accepted",
            "reason": None,
            "case": {
                "id": f"{group}_{mode}",
                "group": group,
                "mode": mode,
                "role": role,
                "threshold": threshold,
            },
            "profile": {
                "path": profile_path,
                "model_family": "qwen",
                "quantization": "fp8",
                "kv_cache_dtype": "int8_per_token_head",
                "mtp_k": 3,
                "compatible_modes": ["fast", "normal", "safe"],
            },
            "workload": {
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "warmups": warmups,
                "measured_runs": measured_runs,
            },
            "model": {
                "checkpoint": checkpoint,
                "tokenizer": tokenizer,
                "served_alias": "test-served-alias",
            },
            "provenance": {
                "git_head": _TEST_GIT_HEAD,
                "runtime": {
                    "python_version": "3.12.0",
                    "python_implementation": "CPython",
                    "vllm_version": "test",
                },
                "build": {
                    "source_closure_sha256": _sha256(closure),
                    "source_closure_schema_version": 1,
                },
            },
            "artifacts": [
                {
                    "path": result_name,
                    "role": "semantic_jsonl",
                    "sha256": _sha256(result),
                },
                {
                    "path": "bench-results.jsonl",
                    "role": "bench_jsonl",
                    "sha256": _sha256(bench),
                },
                {
                    "path": "results.csv",
                    "role": "summary_csv",
                    "sha256": _sha256(csv),
                },
                {
                    "path": "profile.env",
                    "role": "profile_snapshot",
                    "sha256": _sha256(profile),
                },
                {
                    "path": closure.name,
                    "role": "source_closure",
                    "sha256": _sha256(closure),
                },
                {
                    "path": metadata.name,
                    "role": "metadata_tsv",
                    "sha256": _sha256(metadata),
                },
            ],
            "metrics": metrics,
        },
    )
    return manifest


def _captured_manifest(case_dir: Path, result_name: str = "results.jsonl") -> Path:
    return _write_captured_manifest(
        case_dir,
        result_name=result_name,
        semantic_text=(
            '{"label":"captured_fast-run1", '
            '"content_sample": " the the the the", "chunks": 3, '
            '"decode_tok_s": 12.5, "prefill_tok_s": 8.0}\n'
        ),
        csv_text=(
            "prefill_tok_s,decode_tok_s,chunks,filler_valid\n"
            "8.0,12.5,3.0,True\n"
        ),
        metrics={
            "chunks_median": 3.0,
            "decode_median": 12.5,
            "filler_valid": True,
            "prefill_median": 8.0,
        },
    )


def test_validator_accepts_captured_artifact_and_rejects_checksum_mismatch(
    tmp_path: Path,
) -> None:
    # Given: a captured vLLM benchmark JSONL artifact and its manifest.
    case_dir = tmp_path / "captured"
    case_dir.mkdir()
    manifest = _captured_manifest(case_dir)

    # When: the offline validator reads the matching manifest.
    valid = _run_validator(str(manifest))

    # Then: it accepts the artifact, but fails once the referenced bytes change.
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.strip() == "OK captured"
    (case_dir / "results.jsonl").write_text("changed\n", encoding="utf-8")
    mismatch = _run_validator(str(manifest))
    assert mismatch.returncode == 1
    assert mismatch.stderr.strip() == "ERROR artifact checksum mismatch: results.jsonl"
    artifact = case_dir / "results.jsonl"
    artifact.write_text(
        '{"label":"captured_fast-run1", '
        '"content_sample": " the the the the", "chunks": 3, '
        '"decode_tok_s": 12.5, "prefill_tok_s": 8.0}\n',
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    semantic_artifact = next(
        entry
        for entry in payload["artifacts"]
        if entry["role"] == "semantic_jsonl"
    )
    semantic_artifact["sha256"] = _sha256(artifact)
    payload["metrics"]["decode_median"] = 13.0
    _write_json(manifest, payload)
    misleading = _run_validator(str(manifest))
    assert misleading.returncode == 1
    assert (
        misleading.stderr.strip()
        == "ERROR captured decode_median does not match artifact"
    )


def test_validator_enforces_non_captured_state_semantics(tmp_path: Path) -> None:
    # Given: every terminal non-captured state.
    for status in ("not-run", "rejected", "failed"):
        manifest = tmp_path / f"{status}.json"
        _write_json(
            manifest,
            {
                "schema_version": 2,
                "status": status,
                "decision": "excluded",
                "reason": "test_terminal_state",
                "artifacts": [],
                "metrics": {"output_throughput": 99.0},
            },
        )

        # When: it tries to claim throughput without a captured artifact.
        result = _run_validator(str(manifest))

        # Then: each state is rejected with a stable error.
        assert result.returncode == 1
        assert result.stderr.strip() == f"ERROR {status} must not declare metrics"


def test_validator_accepts_non_captured_states_without_claimed_metrics(
    tmp_path: Path,
) -> None:
    # Given: terminal non-captured manifests without throughput claims.
    for status in ("not-run", "rejected", "failed"):
        manifest = tmp_path / f"{status}.json"
        _write_json(
            manifest,
            {
                "schema_version": 2,
                "status": status,
                "decision": "excluded",
                "reason": "test_terminal_state",
                "artifacts": [],
            },
        )

        # When: the offline validator reads the terminal status.
        result = _run_validator(str(manifest))

        # Then: the manifest is structurally valid without being benchmark success.
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"OK {status}"


def test_validator_reports_malformed_manifest_with_stable_error(tmp_path: Path) -> None:
    # Given: an artifact manifest that is not JSON.
    manifest = tmp_path / "malformed.json"
    manifest.write_text("{", encoding="utf-8")

    # When: the offline validator parses the boundary input.
    result = _run_validator(str(manifest))

    # Then: it fails deterministically instead of reporting success.
    assert result.returncode == 1
    assert result.stderr.strip() == f"ERROR invalid JSON: {manifest}"


def test_fake_remote_source_closure_match_mismatch_and_missing(tmp_path: Path) -> None:
    # Given: a source closure and a separately materialized fake remote tree.
    source = tmp_path / "source"
    remote = tmp_path / "fake-remote"
    relative = Path("vllm/v1/attention/sm75_attention_planner.py")
    for root in (source, remote):
        target = root / relative
        target.parent.mkdir(parents=True)
        target.write_text("planner bytes\n", encoding="utf-8")
    closure = tmp_path / "source-closure-manifest.json"
    _write_json(
        closure,
        {
            "schema_version": 1,
            "files": [{"path": str(relative), "sha256": _sha256(source / relative)}],
        },
    )

    # When: the remote closure has matching bytes, then mismatched and missing bytes.
    matching = _run_validator(
        "--source-closure", str(closure), "--source-root", str(remote)
    )
    (remote / relative).write_text("different bytes\n", encoding="utf-8")
    mismatch = _run_validator(
        "--source-closure", str(closure), "--source-root", str(remote)
    )
    (remote / relative).unlink()
    missing = _run_validator(
        "--source-closure", str(closure), "--source-root", str(remote)
    )

    # Then: fake-remote validation never accepts local-only hashes.
    assert matching.returncode == 0, matching.stderr
    assert matching.stdout.strip() == "OK source closure"
    assert mismatch.returncode == 1
    assert mismatch.stderr.strip() == f"ERROR source checksum mismatch: {relative}"
    assert missing.returncode == 1
    assert missing.stderr.strip() == f"ERROR source missing: {relative}"


def test_summary_consumes_actual_non_default_case_result_path(tmp_path: Path) -> None:
    # Given: only a non-default workload result, plus a stale default-named file.
    root = tmp_path / "results"
    case_dir = root / "fp8_int8kv_fast"
    case_dir.mkdir(parents=True)
    _write_captured_manifest(
        case_dir,
        result_name="pp8192_tg256.jsonl",
        semantic_text=(
            '{"label":"fp8_int8kv_fast-run1", '
            '"content_sample": " the the the the", "chunks": 9, '
            '"decode_tok_s": 23.0, "prefill_tok_s": 17.0}\n'
        ),
        csv_text=(
            "prefill_tok_s,decode_tok_s,chunks,filler_valid\n"
            "17.0,23.0,9.0,True\n"
        ),
        metrics={
            "chunks_median": 9.0,
            "decode_median": 23.0,
            "filler_valid": True,
            "prefill_median": 17.0,
        },
        group="fp8_int8kv",
        mode="fast",
        role="fast_guard",
        threshold=20.0,
        prompt_tokens=8192,
        generation_tokens=256,
    )
    (case_dir / "pp4096_tg128.jsonl").write_text(
        '{"content_sample": "drift", "chunks": 1, '
        '"decode_tok_s": 1.0, "prefill_tok_s": 1.0}\n',
        encoding="utf-8",
    )
    (root / "cases.tsv").write_text(
        "group\tprofile\tmodel_dir\tmode\trole\tthreshold\n"
        "fp8_int8kv\tprofile.env\tmodel\tfast\tfast_guard\t20\n",
        encoding="utf-8",
    )

    # When: the offline summary is built from manifests.
    summary = _run_validator("--summarize", str(root), "--expected-runs", "1")

    # Then: it records the non-default result path without stale-default pollution.
    assert summary.returncode == 0, summary.stderr
    row = (root / "summary.tsv").read_text(encoding="utf-8").splitlines()[1]
    assert "pp8192_tg256.jsonl" in row
    assert "pp4096_tg128.jsonl" not in row
    assert "23.00" in row


def test_summary_excludes_warmup_records_from_measured_metrics(tmp_path: Path) -> None:
    # Given: production labels contain one warmup and one measured run.
    root = tmp_path / "results"
    case_dir = root / "fp8_int8kv_fast"
    case_dir.mkdir(parents=True)
    _write_captured_manifest(
        case_dir,
        result_name="pp8192_tg256.jsonl",
        semantic_text=(
            '{"label":"fp8_int8kv_fast-warmup1", '
            '"content_sample": "drift", "chunks": 99, '
            '"decode_tok_s": 1.0, "prefill_tok_s": 2.0}\n'
            '{"label":"fp8_int8kv_fast-run1", '
            '"content_sample": " the the the the", "chunks": 9, '
            '"decode_tok_s": 23.0, "prefill_tok_s": 17.0}\n'
        ),
        csv_text=(
            "prefill_tok_s,decode_tok_s,chunks,filler_valid\n"
            "17.0,23.0,9.0,True\n"
        ),
        metrics={
            "chunks_median": 9.0,
            "decode_median": 23.0,
            "filler_valid": True,
            "prefill_median": 17.0,
        },
        group="fp8_int8kv",
        mode="fast",
        role="fast_guard",
        threshold=20.0,
        prompt_tokens=8192,
        generation_tokens=256,
        warmups=1,
    )
    (root / "cases.tsv").write_text(
        "group\tprofile\tmodel_dir\tmode\trole\tthreshold\n"
        "fp8_int8kv\tprofile.env\tmodel\tfast\tfast_guard\t20\n",
        encoding="utf-8",
    )

    # When: the offline summary uses production defaults of one warmup and one run.
    summary = _run_validator("--summarize", str(root), "--expected-runs", "1")

    # Then: only the measured run satisfies completeness and all summary metrics.
    assert summary.returncode == 0, summary.stderr
    row = (root / "summary.tsv").read_text(encoding="utf-8").splitlines()[1]
    header = (root / "summary.tsv").read_text(encoding="utf-8").splitlines()[0]
    fields = dict(zip(header.split("\t"), row.split("\t")))
    assert fields["runs"] == "1"
    assert fields["prefill_median"] == "17.00"
    assert fields["decode_median"] == "23.00"
    assert fields["chunks_median"] == "9.00"
    assert fields["filler_valid"] == "True"


def test_validator_rejects_captured_manifest_without_compatible_csv(
    tmp_path: Path,
) -> None:
    # Given: a captured manifest which only points to JSONL.
    case_dir = tmp_path / "captured"
    case_dir.mkdir()
    manifest = _captured_manifest(case_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"] = [payload["artifacts"][0]]
    _write_json(manifest, payload)

    # When: the offline validator reads the incomplete artifact contract.
    result = _run_validator(str(manifest))

    # Then: captured cannot claim a JSONL-only result.
    assert result.returncode == 1
    assert result.stderr.strip() == "ERROR captured artifact set is incomplete"


def test_summary_rejects_not_run_case_instead_of_reporting_success(
    tmp_path: Path,
) -> None:
    # Given: a structurally valid terminal not-run case.
    root = tmp_path / "results"
    case_dir = root / "fp8_int8kv_fast"
    case_dir.mkdir(parents=True)
    _write_json(
        case_dir / "artifact-manifest.json",
        {
            "schema_version": 2,
            "status": "not-run",
            "decision": "excluded",
            "reason": "not_started",
            "artifacts": [],
        },
    )
    (root / "cases.tsv").write_text(
        "group\tprofile\tmodel_dir\tmode\trole\tthreshold\n"
        "fp8_int8kv\tprofile.env\tmodel\tfast\tfast_guard\t20\n",
        encoding="utf-8",
    )

    # When: summary consumes the terminal status.
    summary = _run_validator("--summarize", str(root))

    # Then: it writes a failure verdict rather than misleading benchmark success.
    assert summary.returncode == 1
    assert summary.stderr.strip() == "ERROR summary validation failed"
    assert (root / "verdict.txt").read_text(encoding="utf-8").startswith("FAIL\n")


def test_validator_rejects_unlabeled_rows_in_measured_only_mode(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "captured"
    case_dir.mkdir()
    manifest = _captured_manifest(case_dir)
    semantic = case_dir / "results.jsonl"
    semantic.write_text(
        '{"content_sample":" the the the the","chunks":3,'
        '"decode_tok_s":12.5,"prefill_tok_s":8.0}\n',
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    next(
        entry
        for entry in payload["artifacts"]
        if entry["role"] == "semantic_jsonl"
    )["sha256"] = _sha256(semantic)
    _write_json(manifest, payload)

    result = _run_validator(str(manifest))

    assert result.returncode == 1
    assert result.stderr.strip() == "ERROR semantic JSONL is empty: results.jsonl"


def test_validator_rejects_missing_provenance_and_profile_snapshot_drift(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "captured"
    case_dir.mkdir()
    manifest = _captured_manifest(case_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    provenance = payload.pop("provenance")
    _write_json(manifest, payload)

    missing = _run_validator(str(manifest))

    assert missing.returncode == 1
    assert missing.stderr.strip() == "ERROR provenance is invalid"

    payload["provenance"] = provenance
    profile = case_dir / "profile.env"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace("MTP_K=3", "MTP_K=0"),
        encoding="utf-8",
    )
    next(
        entry
        for entry in payload["artifacts"]
        if entry["role"] == "profile_snapshot"
    )["sha256"] = _sha256(profile)
    _write_json(manifest, payload)

    drift = _run_validator(str(manifest))

    assert drift.returncode == 1
    assert drift.stderr.strip() == (
        "ERROR profile metadata does not match profile snapshot"
    )


def test_validator_binds_build_provenance_to_source_closure(tmp_path: Path) -> None:
    case_dir = tmp_path / "captured"
    case_dir.mkdir()
    manifest = _captured_manifest(case_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    closure = case_dir / "source-closure-manifest.json"
    closure.write_text(
        closure.read_text(encoding="utf-8").replace("2" * 64, "3" * 64),
        encoding="utf-8",
    )
    next(
        entry
        for entry in payload["artifacts"]
        if entry["role"] == "source_closure"
    )["sha256"] = _sha256(closure)
    _write_json(manifest, payload)

    result = _run_validator(str(manifest))

    assert result.returncode == 1
    assert result.stderr.strip() == (
        "ERROR build provenance does not match source closure"
    )


def test_evaluator_rejects_sync_bypass_before_remote_execution(tmp_path: Path) -> None:
    result = _run_fake_evaluator(tmp_path, eval_sync="0")

    assert result.returncode == 1
    assert "source closure cannot be verified" in result.stderr
    assert not (tmp_path / "bench-arguments.json").exists()


def test_fake_remote_evaluator_preserves_checkpoint_and_api_identities(
    tmp_path: Path,
) -> None:
    # Given: a fake remote with a deterministic benchmark CLI receiver.
    result = _run_fake_evaluator(tmp_path)
    remote_root = tmp_path / "remote"

    # When: the production evaluator completes a selected fast guard case.
    assert result.returncode == 0, result.stderr
    arguments = json.loads(
        (tmp_path / "bench-arguments.json").read_text(encoding="utf-8")
    )
    manifest = remote_root / "results/fp8_int8kv_fast/artifact-manifest.json"

    # Then: checkpoint, tokenizer, and API identities remain distinct.
    assert arguments == {
        "model": str(tmp_path / "models/checkpoint"),
        "served": "qwen27b-fp8-int8kv-252K-mtp3-text-only-cu128",
        "tokenizer": str(tmp_path / "models/tokenizer"),
    }
    validation = _run_validator(str(manifest))
    assert validation.returncode == 0, validation.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["model"] == {
        "checkpoint": str(tmp_path / "models/checkpoint"),
        "served_alias": "qwen27b-fp8-int8kv-252K-mtp3-text-only-cu128",
        "tokenizer": str(tmp_path / "models/tokenizer"),
    }
    assert payload["workload"] == {
        "generation_tokens": 128,
        "measured_runs": 1,
        "prompt_tokens": 4096,
        "warmups": 0,
    }
    assert {entry["role"] for entry in payload["artifacts"]} == {
        "semantic_jsonl",
        "bench_jsonl",
        "summary_csv",
        "profile_snapshot",
        "source_closure",
        "metadata_tsv",
    }
    assert (remote_root / "results/summary.tsv").read_text(encoding="utf-8").count(
        "pp4096_tg128.jsonl"
    ) == 1


def test_fake_remote_default_warmup_is_not_a_measured_run(tmp_path: Path) -> None:
    result = _run_fake_evaluator(tmp_path, warmups="1")
    remote_root = tmp_path / "remote"
    case_dir = remote_root / "results/fp8_int8kv_fast"

    assert result.returncode == 0, result.stderr
    semantic_rows = (case_dir / "pp4096_tg128.jsonl").read_text(encoding="utf-8")
    assert semantic_rows.count('"label":"fp8_int8kv_fast-warmup1"') == 1
    assert semantic_rows.count('"label":"fp8_int8kv_fast-run1"') == 1
    summary_path = remote_root / "results/summary.tsv"
    summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    summary_fields = dict(
        zip(
            summary_lines[0].split("\t"),
            summary_lines[1].split("\t"),
        )
    )
    assert summary_fields["runs"] == "1"


def test_fake_remote_evaluator_excludes_warmups_from_aggregate_artifacts(
    tmp_path: Path,
) -> None:
    # Given: a production fake remote with deliberately divergent warmup data.
    result = _run_fake_evaluator(
        tmp_path,
        warmups="1",
        distinct_warmup=True,
        prompt_tokens="8192",
        gen_tokens="256",
    )
    case_dir = tmp_path / "remote/results/fp8_int8kv_fast"

    # When: the evaluator captures one warmup followed by one measured run.
    manifest = json.loads(
        (case_dir / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    csv_rows = (case_dir / "results.csv").read_text(encoding="utf-8").splitlines()
    summary_lines = (
        tmp_path / "remote/results/summary.tsv"
    ).read_text(encoding="utf-8").splitlines()
    summary = dict(zip(summary_lines[0].split("\t"), summary_lines[1].split("\t")))

    # Then: raw semantics retain both records, but derived artifacts use run1 only.
    assert result.returncode == 0, result.stderr
    assert (case_dir / "pp8192_tg256.jsonl").read_text(encoding="utf-8").count(
        '"label":"fp8_int8kv_fast-'
    ) == 2
    assert manifest["metrics"] == {
        "chunks_median": 6.0,
        "decode_median": 80.0,
        "filler_valid": True,
        "prefill_median": 40.0,
    }
    assert csv_rows == [
        "prefill_tok_s,decode_tok_s,chunks,filler_valid",
        "40.0,80.0,6,True",
    ]
    assert summary["runs"] == "1"
    assert summary["prefill_median"] == "40.00"
    assert summary["decode_median"] == "80.00"
    assert summary["chunks_median"] == "6.00"
    assert summary["filler_valid"] == "True"


def test_fake_remote_timeout_writes_failed_manifest_and_preserves_output(
    tmp_path: Path,
) -> None:
    # Given: a fake semantic runner that exceeds the production timeout.
    result = _run_fake_evaluator(tmp_path, hang_profile=True)
    case_dir = tmp_path / "remote/results/fp8_int8kv_fast"

    # When: the bounded profile request command times out.
    manifest = json.loads(
        (case_dir / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    # Then: it writes failed status, preserves logs, and summary cannot pass.
    assert result.returncode == 1
    assert manifest["status"] == "failed"
    assert manifest["reason"] == "benchmark_timeout_or_failure"
    assert (case_dir / "bench.out").is_file()
    assert (case_dir / "bench.err").is_file()
    assert (
        (tmp_path / "remote/results/verdict.txt")
        .read_text(encoding="utf-8")
        .startswith("FAIL\n")
    )


def test_fake_remote_stale_case_writes_rejected_manifest(tmp_path: Path) -> None:
    # Given: a previously created selected case directory.
    stale_case = tmp_path / "remote/results/fp8_int8kv_fast"
    stale_case.mkdir(parents=True)

    # When: the evaluator encounters stale state before running a benchmark.
    result = _run_fake_evaluator(tmp_path)
    rejected = tmp_path / "remote/results/fp8_int8kv_fast.rejected.json"

    # Then: production writes a rejected receipt and exits nonzero.
    assert result.returncode == 1
    assert json.loads(rejected.read_text(encoding="utf-8")) == {
        "artifacts": [],
        "decision": "excluded",
        "reason": "stale_case_directory",
        "schema_version": 2,
        "status": "rejected",
    }
