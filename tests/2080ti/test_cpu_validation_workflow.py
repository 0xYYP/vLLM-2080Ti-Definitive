# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).parents[2]
_WORKFLOW: Final = _ROOT / ".github/workflows/cpu-validation.yml"
_LAUNCHER_DOCS: Final = (
    _ROOT / "docs/non-interactive-launcher.md",
    _ROOT / "docs/non-interactive-launcher.zh-CN.md",
)
_MANIFEST_DOCS: Final = (
    _ROOT / "docs/benchmark-manifest.md",
    _ROOT / "docs/benchmark-manifest.zh-CN.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_cpu_workflow_enforces_the_cpu_validation_contract() -> None:
    # Given: the repository CPU validation workflow.
    workflow = _read(_WORKFLOW)

    # When: its static CI contract is inspected without executing GitHub Actions.
    # Then: it has pinned, non-optional Ubuntu/Bash/Python policy gates.
    for required in (
        "workflow_dispatch:",
        "permissions:\n  contents: read",
        "concurrency:",
        "cancel-in-progress: true",
        "shell-contracts:",
        "python-policy-tests:",
        "runs-on: ubuntu-24.04",
        "timeout-minutes:",
        "python-version: \"3.12\"",
        "fetch-depth: 0",
        "bash --version",
        "BASH_VERSINFO[0]",
        "set -o pipefail",
        (
            "bash -n build.sh launcher.sh tools/validate_profiles.sh "
            "tools/evaluate_fast_modes.sh"
        ),
        "bash tools/validate_profiles.sh",
        "pytest -q",
        "test_sm75_attention_trace.py",
        "test_sm75_attention_planner.py",
        "test_sm75_int8kv_integration.py",
        "test_sm75_turboquant_integration.py",
        "test_sm75_spec_sync_integration.py",
        "test_launcher_profile_contract.py",
        "test_benchmark_manifest.py",
        "test_cpu_validation_workflow.py",
        "ruff check",
        "python -m py_compile",
        "git diff --check",
        "git diff --name-only --diff-filter=ACMR",
        "VALIDATION_BASE...HEAD",
        "if: failure()",
        "github.run_id",
    ):
        assert required in workflow

    assert "continue-on-error" not in workflow
    python_policy_job = workflow.split("  python-policy-tests:", maxsplit=1)[1]
    assert "pytest -q tests/2080ti/test_sm75_attention_trace.py" in python_policy_job
    assert "--ignore" not in python_policy_job
    assert 'ruff check --select E9,F,I "${changed_python_files[@]}"' in python_policy_job
    assert 'python -m py_compile "${changed_python_files[@]}"' in python_policy_job
    assert 'git diff --check "$VALIDATION_BASE...HEAD"' in python_policy_job
    for forbidden in (
        "pip install .",
        "requirements/",
        "torch",
        "cuda",
        "nvidia-smi",
        "docker",
        "ssh ",
        "model download",
    ):
        assert forbidden not in workflow.lower()
    assert not re.search(r"^\s*bash tools/evaluate_fast_modes\.sh\s*$", workflow, re.M)

    actions = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", workflow)
    assert actions
    assert all(re.fullmatch(r"[0-9a-f]{40}", action) for action in actions)
    artifact_names = re.findall(r"name:\s*(cpu-[^\n]+)", workflow)
    assert len(artifact_names) == len(set(artifact_names))
    assert all("${{ github.run_id }}" in name for name in artifact_names)


def test_bilingual_docs_describe_launcher_and_manifest_contracts() -> None:
    # Given: English and Simplified Chinese user-facing contract documents.
    # When: their machine-consumed terms are checked for equivalent coverage.
    # Then: both explain the stable boundaries that CPU CI protects.
    launcher_terms = (
        "CLI > ENV > PROFILE > default",
        "resolved_",
        "COMPILATION_CONFIG_JSON",
        "JSON object",
        "canonical",
        "missing profile",
        "GPU_DEVICES",
        "numeric",
        "empty",
        "duplicate",
        "UUID",
        "MIG",
        "TP_SIZE",
        "derived",
        "final_vllm_argv",
    )
    manifest_terms = (
        "schema_version",
        "status",
        "sha256",
        "source closure",
        "checkpoint",
        "tokenizer",
        "served alias",
        "profile snapshot",
        "Git HEAD",
        "runtime version",
        "build identity",
        "EVAL_SYNC=0",
        "timeout",
        "raw artifact",
        "warmup",
        "measured-only",
    )
    for path in _LAUNCHER_DOCS:
        text = _read(path)
        for term in launcher_terms:
            assert term in text, f"{path.name} is missing {term}"
    for path in _MANIFEST_DOCS:
        text = _read(path)
        for term in manifest_terms:
            assert term in text, f"{path.name} is missing {term}"
