#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, TypedDict

_SCHEMA_VERSION: Final = 2
_STATUSES: Final = frozenset({"captured", "not-run", "rejected", "failed"})
_ROLES: Final = frozenset(
    {
        "semantic_jsonl",
        "bench_jsonl",
        "summary_csv",
        "profile_snapshot",
        "source_closure",
        "metadata_tsv",
    }
)
_CAPTURED_ROLES: Final = _ROLES
_FIELDS: Final = [
    "group",
    "profile",
    "mode",
    "role",
    "threshold",
    "status",
    "runs",
    "prefill_median",
    "decode_median",
    "chunks_median",
    "filler_valid",
    "baseline_target",
    "result",
]


class ValidationError(Exception):
    pass


class SemanticRecord(TypedDict):
    chunks: float
    decode_tok_s: float
    prefill_tok_s: float
    content_sample: str


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    expected_runs: int
    margin: float
    fast_only: bool
    baseline_root: Path | None
    compare_required: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValidationError(f"manifest missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"manifest must be an object: {path}")
    return value


def _relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationError(f"{label} path is not relative: {value}")
    return Path(*path.parts)


def _number(value: object, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field} is invalid: {path}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationError(f"{field} is invalid: {path}")
    return numeric


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field} is invalid")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} is invalid")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{field} is invalid")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValidationError(f"{field} is invalid")
    return value


def _profile_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _artifacts(manifest: dict[str, object], base: Path) -> dict[str, Path]:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ValidationError("artifacts are invalid")
    artifacts: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValidationError("artifact entry must be an object")
        role = entry.get("role")
        relative = _relative(entry.get("path"), "artifact")
        expected = entry.get("sha256")
        if not isinstance(role, str) or role not in _ROLES:
            raise ValidationError(f"artifact role is invalid: {relative}")
        if role in artifacts:
            raise ValidationError(f"artifact role is duplicated: {role}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValidationError(f"artifact checksum is invalid: {relative}")
        target = base / relative
        if not target.is_file():
            raise ValidationError(f"artifact missing: {relative}")
        if _sha256(target) != expected:
            raise ValidationError(f"artifact checksum mismatch: {relative}")
        artifacts[role] = target
    return artifacts


def _semantic_records(path: Path, measured_only: bool = False) -> list[SemanticRecord]:
    records: list[SemanticRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(f"invalid JSONL: {path}") from error
        if not isinstance(raw, dict):
            raise ValidationError(f"JSONL row must be an object: {path}")
        sample = raw.get("content_sample")
        if not isinstance(sample, str):
            raise ValidationError(f"content_sample is invalid: {path}")
        label = raw.get("label")
        if label is not None and not isinstance(label, str):
            raise ValidationError(f"label is invalid: {path}")
        record = {
            "chunks": _number(raw.get("chunks"), "chunks", path),
            "decode_tok_s": _number(raw.get("decode_tok_s"), "decode_tok_s", path),
            "prefill_tok_s": _number(
                raw.get("prefill_tok_s"), "prefill_tok_s", path
            ),
            "content_sample": sample,
        }
        if not measured_only or (
            isinstance(label, str) and re.fullmatch(r".+-run[1-9][0-9]*", label)
        ):
            records.append(record)
    if not records:
        raise ValidationError(f"semantic JSONL is empty: {path.name}")
    return records


def _filler_valid(records: list[SemanticRecord]) -> bool:
    return all(
        " the the the the" in record["content_sample"].lower()
        and "climate change" not in record["content_sample"].lower()
        and "introduction" not in record["content_sample"].lower()
        for record in records
    )


def _metrics(records: list[SemanticRecord]) -> dict[str, float | bool]:
    return {
        "prefill_median": statistics.median(
            record["prefill_tok_s"] for record in records
        ),
        "decode_median": statistics.median(
            record["decode_tok_s"] for record in records
        ),
        "chunks_median": statistics.median(record["chunks"] for record in records),
        "filler_valid": _filler_valid(records),
    }


def _validate_case_contract(
    manifest: dict[str, object], artifacts: dict[str, Path]
) -> None:
    case = _object(manifest.get("case"), "case")
    case_id = _string(case.get("id"), "case.id")
    group = _string(case.get("group"), "case.group")
    mode = _string(case.get("mode"), "case.mode")
    _string(case.get("role"), "case.role")
    _number(case.get("threshold"), "case.threshold", artifacts["metadata_tsv"])
    if case_id != _case_id({"group": group, "mode": mode}):
        raise ValidationError("case.id does not match group and mode")

    profile = _object(manifest.get("profile"), "profile")
    _relative(profile.get("path"), "profile")
    expected_profile = {
        "model_family": _string(profile.get("model_family"), "profile.model_family"),
        "quantization": _string(profile.get("quantization"), "profile.quantization"),
        "kv_cache_dtype": _string(
            profile.get("kv_cache_dtype"), "profile.kv_cache_dtype"
        ),
        "mtp_k": _integer(profile.get("mtp_k"), "profile.mtp_k"),
        "compatible_modes": _string_list(
            profile.get("compatible_modes"), "profile.compatible_modes"
        ),
    }
    snapshot = _profile_values(artifacts["profile_snapshot"])
    try:
        snapshot_mtp_k = int(snapshot.get("MTP_K", "0"))
    except ValueError as error:
        raise ValidationError("profile snapshot MTP_K is invalid") from error
    actual_profile: dict[str, object] = {
        "model_family": snapshot.get("MODEL_FAMILY", ""),
        "quantization": snapshot.get("QUANTIZATION", ""),
        "kv_cache_dtype": snapshot.get("KV_CACHE_DTYPE") or "fp16",
        "mtp_k": snapshot_mtp_k,
        "compatible_modes": sorted(
            item.strip()
            for item in snapshot.get("COMPATIBLE_MODES", "").split(",")
            if item.strip()
        ),
    }
    expected_profile["compatible_modes"] = sorted(
        expected_profile["compatible_modes"]
    )
    if expected_profile != actual_profile:
        raise ValidationError("profile metadata does not match profile snapshot")
    if mode not in actual_profile["compatible_modes"]:
        raise ValidationError("case.mode is not compatible with profile snapshot")

    workload = _object(manifest.get("workload"), "workload")
    _integer(workload.get("prompt_tokens"), "workload.prompt_tokens", minimum=1)
    _integer(
        workload.get("generation_tokens"),
        "workload.generation_tokens",
        minimum=1,
    )
    _integer(workload.get("warmups"), "workload.warmups")
    _integer(workload.get("measured_runs"), "workload.measured_runs", minimum=1)

    model = _object(manifest.get("model"), "model")
    _string(model.get("checkpoint"), "model.checkpoint")
    _string(model.get("tokenizer"), "model.tokenizer")
    _string(model.get("served_alias"), "model.served_alias")

    provenance = _object(manifest.get("provenance"), "provenance")
    git_head = _string(provenance.get("git_head"), "provenance.git_head")
    if not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise ValidationError("provenance.git_head is invalid")
    runtime = _object(provenance.get("runtime"), "provenance.runtime")
    for field in ("python_version", "python_implementation", "vllm_version"):
        _string(runtime.get(field), f"provenance.runtime.{field}")
    build = _object(provenance.get("build"), "provenance.build")
    closure_sha = _string(
        build.get("source_closure_sha256"),
        "provenance.build.source_closure_sha256",
    )
    if closure_sha != _sha256(artifacts["source_closure"]):
        raise ValidationError("build provenance does not match source closure")
    if _integer(
        build.get("source_closure_schema_version"),
        "provenance.build.source_closure_schema_version",
        minimum=1,
    ) != 1:
        raise ValidationError("source closure schema_version is invalid")
    closure = _json(artifacts["source_closure"])
    if closure.get("schema_version") != 1 or not isinstance(closure.get("files"), list):
        raise ValidationError("source closure artifact is invalid")


def _validate_captured(manifest: dict[str, object], base: Path) -> None:
    artifacts = _artifacts(manifest, base)
    if set(artifacts) != _CAPTURED_ROLES:
        raise ValidationError(
            "captured artifact set is incomplete"
        )
    if manifest.get("decision") != "accepted":
        raise ValidationError("captured decision is invalid")
    if manifest.get("reason") is not None:
        raise ValidationError("captured reason must be null")
    _validate_case_contract(manifest, artifacts)
    if artifacts["semantic_jsonl"].suffix != ".jsonl":
        raise ValidationError("semantic JSONL artifact is invalid")
    if artifacts["bench_jsonl"].suffix != ".jsonl":
        raise ValidationError("bench JSONL artifact is invalid")
    if artifacts["summary_csv"].name != "results.csv":
        raise ValidationError("summary CSV artifact is invalid")
    with artifacts["summary_csv"].open(encoding="utf-8", newline="") as handle:
        headers = next(csv.reader(handle), [])
    if headers != ["prefill_tok_s", "decode_tok_s", "chunks", "filler_valid"]:
        raise ValidationError("summary CSV header is invalid")
    expected = _metrics(
        _semantic_records(artifacts["semantic_jsonl"], measured_only=True)
    )
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValidationError("captured metrics are invalid")
    for field, value in expected.items():
        actual = metrics.get(field)
        if isinstance(value, bool):
            if actual is not value:
                raise ValidationError(f"captured {field} does not match artifact")
        elif _number(actual, field, base) != value:
            raise ValidationError(f"captured {field} does not match artifact")


def validate_artifact_manifest(path: Path) -> str:
    manifest = _json(path)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValidationError("schema_version is invalid")
    status = manifest.get("status")
    if not isinstance(status, str) or status not in _STATUSES:
        raise ValidationError("status is invalid")
    match status:
        case "captured":
            _validate_captured(manifest, path.parent)
        case "not-run" | "rejected" | "failed":
            if manifest.get("decision") != "excluded":
                raise ValidationError(f"{status} decision is invalid")
            _string(manifest.get("reason"), f"{status} reason")
            if manifest.get("metrics"):
                raise ValidationError(f"{status} must not declare metrics")
            if manifest.get("artifacts") != []:
                _artifacts(manifest, path.parent)
        case unreachable:
            raise AssertionError(f"unreachable status: {unreachable}")
    return status


def validate_source_closure(manifest_path: Path, source_root: Path) -> None:
    manifest = _json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("source closure files are invalid")
    for entry in files:
        if not isinstance(entry, dict):
            raise ValidationError("source entry must be an object")
        relative = _relative(entry.get("path"), "source")
        expected = entry.get("sha256")
        if not isinstance(expected, str):
            raise ValidationError(f"source checksum is invalid: {relative}")
        target = source_root / relative
        if not target.is_file():
            raise ValidationError(f"source missing: {relative}")
        if _sha256(target) != expected:
            raise ValidationError(f"source checksum mismatch: {relative}")


def _case_id(case: dict[str, str]) -> str:
    raw = f"{case['group']}_{case['mode']}"
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in raw)


def _validate_summary_identity(
    manifest: dict[str, object], case: dict[str, str]
) -> None:
    manifest_case = _object(manifest.get("case"), "case")
    manifest_profile = _object(manifest.get("profile"), "profile")
    manifest_model = _object(manifest.get("model"), "model")
    expected = {
        "id": _case_id(case),
        "group": case["group"],
        "mode": case["mode"],
        "role": case["role"],
        "profile": case["profile"],
        "checkpoint": case["model_dir"],
        "tokenizer": case.get("tokenizer_dir") or case["model_dir"],
    }
    actual = {
        "id": manifest_case.get("id"),
        "group": manifest_case.get("group"),
        "mode": manifest_case.get("mode"),
        "role": manifest_case.get("role"),
        "profile": manifest_profile.get("path"),
        "checkpoint": manifest_model.get("checkpoint"),
        "tokenizer": manifest_model.get("tokenizer"),
    }
    if actual != expected:
        raise ValidationError("case manifest identity does not match cases.tsv")
    if _number(
        manifest_case.get("threshold"), "case.threshold", Path("cases.tsv")
    ) != float(case["threshold"]):
        raise ValidationError("case threshold does not match cases.tsv")


def _baseline_targets(root: Path | None, margin: float) -> dict[str, float]:
    summary = root / "summary.tsv" if root else None
    if summary is None or not summary.is_file():
        return {}
    groups: dict[str, list[float]] = {}
    with summary.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            value = row.get("decode_median", "")
            if (
                row.get("role") == "fp16_compare"
                and row.get("mode") in {"safe", "normal"}
                and value
            ):
                groups.setdefault(row["group"], []).append(float(value))
    return {group: max(values) + margin for group, values in groups.items()}


def _summary_row(
    case: dict[str, str], root: Path, targets: dict[str, float]
) -> tuple[dict[str, str], str | None]:
    manifest_path = root / _case_id(case) / "artifact-manifest.json"
    status, records, result, error = "missing", [], "", None
    try:
        status = validate_artifact_manifest(manifest_path)
        if status == "captured":
            manifest = _json(manifest_path)
            _validate_summary_identity(manifest, case)
            artifacts = _artifacts(manifest, manifest_path.parent)
            result = str(artifacts["semantic_jsonl"])
            records = _semantic_records(artifacts["semantic_jsonl"], measured_only=True)
    except ValidationError as caught:
        error = f"{case['group']} {case['mode']}: {caught}"
    metrics = _metrics(records) if records else None
    row = {
        **case,
        "status": status,
        "runs": str(len(records)),
        "prefill_median": f"{metrics['prefill_median']:.2f}" if metrics else "",
        "decode_median": f"{metrics['decode_median']:.2f}" if metrics else "",
        "chunks_median": f"{metrics['chunks_median']:.2f}" if metrics else "",
        "filler_valid": str(metrics["filler_valid"]) if metrics else "False",
        "baseline_target": f"{targets[case['group']]:.2f}"
        if case["group"] in targets
        else "",
        "result": result,
    }
    return row, error


def summarize(root: Path, config: SummaryConfig) -> None:
    cases_path = root / "cases.tsv"
    if not cases_path.is_file():
        raise ValidationError("cases.tsv missing")
    with cases_path.open(encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle, delimiter="\t"))
    targets = _baseline_targets(config.baseline_root, config.margin)
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    groups: dict[str, dict[str, dict[str, str]]] = {}
    for case in cases:
        row, error = _summary_row(case, root, targets)
        rows.append(row)
        groups.setdefault(case["group"], {})[case["mode"]] = row
        if error:
            failures.append(error)
        if row["status"] != "captured" or int(row["runs"]) < config.expected_runs:
            failures.append(
                f"{case['group']} {case['mode']}: incomplete "
                f"status={row['status']} runs={row['runs']}"
            )
        if row["filler_valid"] != "True":
            failures.append(f"{case['group']} {case['mode']}: filler output drift")
        if (
            case["role"] == "fast_guard"
            and row["decode_median"]
            and float(row["decode_median"]) < float(case["threshold"])
        ):
            failures.append(
                f"{case['group']}: fast guard decode "
                f"{float(row['decode_median']):.2f} < threshold "
                f"{float(case['threshold']):.2f}"
            )
    _compare_fast(cases, groups, targets, config, failures)
    _write_summary(root, rows, failures)


def _compare_fast(
    cases: list[dict[str, str]],
    groups: dict[str, dict[str, dict[str, str]]],
    targets: dict[str, float],
    config: SummaryConfig,
    failures: list[str],
) -> None:
    for group in sorted(
        {case["group"] for case in cases if case["role"] == "fp16_compare"}
    ):
        modes = groups[group]
        fast = modes.get("fast")
        if not config.compare_required and fast is None:
            continue
        if fast is None or not fast["decode_median"]:
            failures.append(f"{group}: missing fast decode median")
            continue
        if config.fast_only:
            target = targets.get(group)
            if target is None:
                failures.append(f"{group}: missing baseline safe/normal target")
                continue
        elif not all(
            mode in modes and modes[mode]["decode_median"]
            for mode in ("safe", "normal", "fast")
        ):
            failures.append(f"{group}: missing safe/normal/fast decode medians")
            continue
        else:
            target = (
                max(
                    float(modes["safe"]["decode_median"]),
                    float(modes["normal"]["decode_median"]),
                )
                + config.margin
            )
        if float(fast["decode_median"]) < target:
            failures.append(
                f"{group}: fast decode {float(fast['decode_median']):.2f} "
                f"< target {target:.2f}"
            )


def _write_summary(root: Path, rows: list[dict[str, str]], failures: list[str]) -> None:
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row[field] for field in _FIELDS} for row in rows)
    verdict = root / "verdict.txt"
    text = "FAIL\n" + "\n".join(failures) + "\n" if failures else "PASS\n"
    verdict.write_text(text, encoding="utf-8")
    if failures:
        raise ValidationError("summary validation failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--source-closure", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--expected-runs", type=int, default=1)
    parser.add_argument("--fast-margin", type=float, default=1.0)
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--baseline-result-dir", type=Path)
    parser.add_argument("--fast-compare-required", action="store_true")
    args = parser.parse_args()
    try:
        if args.source_closure:
            if not args.source_root:
                raise ValidationError("--source-root is required with --source-closure")
            validate_source_closure(args.source_closure, args.source_root)
            print("OK source closure")
        elif args.summarize:
            summarize(
                args.summarize,
                SummaryConfig(
                    args.expected_runs,
                    args.fast_margin,
                    args.fast_only,
                    args.baseline_result_dir,
                    args.fast_compare_required,
                ),
            )
            print(f"OK summary {args.summarize}")
        elif args.manifest:
            print(f"OK {validate_artifact_manifest(Path(args.manifest))}")
        else:
            raise ValidationError(
                "manifest, --source-closure, or --summarize is required"
            )
    except ValidationError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
