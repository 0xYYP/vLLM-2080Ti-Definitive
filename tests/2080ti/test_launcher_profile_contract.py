# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_LAUNCHER = _ROOT / "launcher.sh"
_VALIDATOR = _ROOT / "tools/validate_profiles.sh"


@pytest.fixture
def launcher_environment(tmp_path: Path) -> dict[str, str]:
    # Given: an isolated runtime that can only satisfy the launcher's preflight.
    runtime_python = tmp_path / "runtime/.venv/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(Path(sys.executable))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    log_dir = tmp_path / "logs"
    environment = os.environ.copy()
    environment.update(
        {
            "RUNTIME_ROOT": str(tmp_path / "runtime"),
            "MODEL_DIR": str(model_dir),
            "LOG_DIR": str(log_dir),
            "STATE_FILE": str(log_dir / "state"),
            "PROFILE_DIR": str(tmp_path / "profiles"),
            "GPU_DEVICES": "4,7",
            "TP_SIZE": "2",
        }
    )
    return environment


def _run_launcher(
    environment: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_LAUNCHER), "--print-config", *arguments],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )


def _resolved_field(output: str, name: str) -> str:
    return next(
        line.removeprefix(f"{name}=")
        for line in output.splitlines()
        if line.startswith(f"{name}=")
    )


def _final_vllm_argv(output: str) -> list[str]:
    return shlex.split(_resolved_field(output, "final_vllm_argv"))


@pytest.mark.parametrize(
    ("value", "expected_returncode"),
    [
        ('{"cudagraph_mode":"PIECEWISE"}', 0),
        ("{bad-json", 1),
        ("[]", 1),
        ("1", 1),
        ("", 1),
    ],
    ids=["object", "malformed", "array", "scalar", "empty"],
)
def test_profile_validator_allows_only_compilation_config_objects(
    tmp_path: Path, value: str, expected_returncode: int
) -> None:
    # Given: a route profile with one allowed or rejected compilation JSON value.
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "route.env").write_text(
        "COMPATIBLE_MODES=normal\n"
        f"COMPILATION_CONFIG_JSON={value}\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PROFILE_DIR"] = str(profile_dir)

    # When: the real profile validator evaluates the temporary route.
    result = subprocess.run(
        ["bash", str(_VALIDATOR)],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    # Then: only a JSON object passes the route-profile boundary.
    assert result.returncode == expected_returncode, result.stderr
    if expected_returncode == 0:
        assert "profile_validation_ok total=1" in result.stdout
    else:
        assert "COMPILATION_CONFIG_JSON" in result.stderr


def test_print_config_emits_final_argv_and_canonical_profile_json(
    launcher_environment: dict[str, str], tmp_path: Path
) -> None:
    # Given: a profile that supplies an intentionally non-canonical JSON object.
    profile = tmp_path / "profile.env"
    profile.write_text(
        "COMPATIBLE_MODES=normal\n"
        "MODEL_FAMILY=qwen\n"
        "SERVED_NAME=profile-name\n"
        'COMPILATION_CONFIG_JSON={"z":1,"a":[true,false]}\n',
        encoding="utf-8",
    )

    # When: the real non-interactive launcher resolves the profile.
    result = _run_launcher(launcher_environment, "--profile-file", str(profile))

    # Then: machine-readable values describe the same argv that would launch.
    assert result.returncode == 0, result.stderr
    assert "final_vllm_argv=" in result.stdout
    assert "resolved_tp_size=2" in result.stdout
    assert "resolved_gpu_devices=4,7" in result.stdout
    assert 'resolved_compilation_config_json={"a":[true,false],"z":1}' in result.stdout
    argv = _final_vllm_argv(result.stdout)
    assert argv[argv.index("--compilation-config") + 1] == '{"a":[true,false],"z":1}'


@pytest.mark.parametrize(
    "value",
    ["{bad-json", "[]", "1", '"value"', ""],
    ids=["malformed", "array", "scalar-number", "scalar-string", "empty"],
)
def test_explicit_compilation_config_json_rejects_non_object_values(
    launcher_environment: dict[str, str], value: str
) -> None:
    # Given: each malformed or non-object explicit boundary value.
    launcher_environment["COMPILATION_CONFIG_JSON"] = value

    # When: the launcher resolves its final configuration.
    result = _run_launcher(launcher_environment)

    # Then: it fails before a server can start.
    assert result.returncode != 0
    assert "COMPILATION_CONFIG_JSON" in result.stderr


@pytest.mark.parametrize(
    "variable",
    ["PROFILE", "PROFILE_FILE"],
)
def test_explicit_missing_profile_fails(
    launcher_environment: dict[str, str], variable: str, tmp_path: Path
) -> None:
    # Given: an explicitly selected profile path that does not exist.
    launcher_environment[variable] = str(tmp_path / "missing-profile.env")

    # When: the launcher resolves the configuration.
    result = _run_launcher(launcher_environment)

    # Then: it rejects the explicit selector instead of silently using defaults.
    assert result.returncode != 0
    assert "profile" in result.stderr.lower()


def test_no_profile_retains_manual_default_path(
    launcher_environment: dict[str, str]
) -> None:
    # Given: an existing manual launch configuration without a profile selector.

    # When: the launcher resolves it for printing.
    result = _run_launcher(launcher_environment)

    # Then: the legacy default path remains available.
    assert result.returncode == 0, result.stderr
    assert "final_vllm_argv=" in result.stdout
    argv = _final_vllm_argv(result.stdout)
    assert argv[argv.index("--max-model-len") + 1] == "131072"
    assert _resolved_field(result.stdout, "resolved_max_model_len") == "131072"
    assert _resolved_field(result.stdout, "resolved_max_model_len_source") == "default"


def test_checkpoint_quantization_method_overrides_filename_guess(
    launcher_environment: dict[str, str], tmp_path: Path
) -> None:
    model_dir = tmp_path / "Example-FP8-Checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"quantization_config":{"quant_method":"compressed-tensors"}}',
        encoding="utf-8",
    )
    launcher_environment["MODEL_DIR"] = str(model_dir)

    result = _run_launcher(launcher_environment)

    assert result.returncode == 0, result.stderr
    argv = _final_vllm_argv(result.stdout)
    assert argv[argv.index("--quantization") + 1] == "compressed-tensors"
    assert _resolved_field(result.stdout, "resolved_quantization_source") == "checkpoint"


def test_missing_checkpoint_quantization_falls_back_to_filename_guess(
    launcher_environment: dict[str, str], tmp_path: Path
) -> None:
    model_dir = tmp_path / "Example-FP8-Checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"example"}',
        encoding="utf-8",
    )
    launcher_environment["MODEL_DIR"] = str(model_dir)

    result = _run_launcher(launcher_environment)

    assert result.returncode == 0, result.stderr
    argv = _final_vllm_argv(result.stdout)
    assert argv[argv.index("--quantization") + 1] == "fp8"
    assert _resolved_field(result.stdout, "resolved_quantization_source") == "guessed"


def test_checkpoint_quantization_conflicting_profile_is_rejected(
    launcher_environment: dict[str, str], tmp_path: Path
) -> None:
    model_dir = tmp_path / "Example-FP8-Checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"quantization_config":{"quant_method":"compressed-tensors"}}',
        encoding="utf-8",
    )
    launcher_environment["MODEL_DIR"] = str(model_dir)
    profile = tmp_path / "generic-fp8.env"
    profile.write_text(
        "COMPATIBLE_MODES=normal\n"
        "MODEL_FAMILY=qwen\n"
        "QUANTIZATION=fp8\n",
        encoding="utf-8",
    )

    result = _run_launcher(launcher_environment, "--profile-file", str(profile))

    assert result.returncode != 0
    assert "Profile QUANTIZATION=fp8 conflicts with checkpoint config" in result.stderr


def test_checkpoint_quantization_matching_profile_is_accepted(
    launcher_environment: dict[str, str], tmp_path: Path
) -> None:
    model_dir = tmp_path / "Example-FP8-Checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"quantization_config":{"quant_method":"compressed-tensors"}}',
        encoding="utf-8",
    )
    launcher_environment["MODEL_DIR"] = str(model_dir)
    profile = tmp_path / "compressed-tensors.env"
    profile.write_text(
        "COMPATIBLE_MODES=normal\n"
        "MODEL_FAMILY=qwen\n"
        "QUANTIZATION=compressed-tensors\n",
        encoding="utf-8",
    )

    result = _run_launcher(launcher_environment, "--profile-file", str(profile))

    assert result.returncode == 0, result.stderr
    argv = _final_vllm_argv(result.stdout)
    assert argv[argv.index("--quantization") + 1] == "compressed-tensors"
    assert _resolved_field(result.stdout, "resolved_quantization_source") == "profile"


def test_explicit_quantization_conflict_is_rejected(
    launcher_environment: dict[str, str], tmp_path: Path
) -> None:
    model_dir = tmp_path / "Example-FP8-Checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"quantization_config":{"quant_method":"compressed-tensors"}}',
        encoding="utf-8",
    )
    launcher_environment["MODEL_DIR"] = str(model_dir)

    result = _run_launcher(launcher_environment, "--set", "QUANTIZATION=fp8")

    assert result.returncode != 0
    assert "conflicts with checkpoint config" in result.stderr


@pytest.mark.parametrize(
    ("environment_value", "cli_value", "expected_value", "expected_source"),
    [
        (None, None, "111", "profile"),
        ("222", None, "222", "env"),
        ("222", "333", "333", "cli"),
    ],
    ids=["profile", "environment", "cli"],
)
def test_profile_precedence_selects_highest_explicit_value(
    launcher_environment: dict[str, str],
    tmp_path: Path,
    environment_value: str | None,
    cli_value: str | None,
    expected_value: str,
    expected_source: str,
) -> None:
    # Given: different profile, environment, and CLI values for one route key.
    profile = tmp_path / "precedence.env"
    profile.write_text(
        "COMPATIBLE_MODES=normal\n"
        "MODEL_FAMILY=qwen\n"
        "MAX_MODEL_LEN=111\n",
        encoding="utf-8",
    )
    if environment_value is not None:
        launcher_environment["MAX_MODEL_LEN"] = environment_value

    # When: the selected higher-precedence source is resolved.
    arguments = ["--profile-file", str(profile)]
    if cli_value is not None:
        arguments.extend(["--max-model-len", cli_value])
    result = _run_launcher(launcher_environment, *arguments)

    # Then: CLI > ENV > PROFILE > default selects the expected value.
    assert result.returncode == 0, result.stderr
    argv = _final_vllm_argv(result.stdout)
    assert argv[argv.index("--max-model-len") + 1] == expected_value
    assert (
        _resolved_field(result.stdout, "resolved_max_model_len_source")
        == expected_source
    )


@pytest.mark.parametrize(
    ("profile_value", "environment_value", "cli_value", "expected_source"),
    [
        ('{"profile":11}', None, None, "profile"),
        ('{"profile":11}', '{"environment":22}', None, "env"),
        ('{"profile":11}', '{"environment":22}', '{"cli":33}', "cli"),
        (None, None, None, "generated"),
    ],
    ids=["profile", "environment", "cli", "generated"],
)
def test_print_config_reports_compilation_config_source(
    launcher_environment: dict[str, str],
    tmp_path: Path,
    profile_value: str | None,
    environment_value: str | None,
    cli_value: str | None,
    expected_source: str,
) -> None:
    # Given: each distinct compilation-config source or generated fallback.
    arguments: list[str] = []
    if profile_value is not None:
        profile = tmp_path / "source-compilation.env"
        profile.write_text(
            "COMPATIBLE_MODES=normal\n"
            "MODEL_FAMILY=qwen\n"
            f"COMPILATION_CONFIG_JSON={profile_value}\n",
            encoding="utf-8",
        )
        arguments.extend(["--profile-file", str(profile)])
    if environment_value is not None:
        launcher_environment["COMPILATION_CONFIG_JSON"] = environment_value
    if cli_value is not None:
        arguments.extend(["--compilation-config-json", cli_value])

    # When: the launcher builds the final vLLM argv.
    result = _run_launcher(launcher_environment, *arguments)

    # Then: source reporting is independent from the canonical final JSON value.
    assert result.returncode == 0, result.stderr
    assert (
        _resolved_field(result.stdout, "resolved_compilation_config_json_source")
        == expected_source
    )


def test_unset_compilation_config_json_falls_back_to_generated(
    launcher_environment: dict[str, str],
) -> None:
    # Given: an inherited malformed value that is explicitly unset by the CLI.
    launcher_environment["COMPILATION_CONFIG_JSON"] = "{bad-json"

    # When: generic --unset clears the inherited boundary value.
    result = _run_launcher(
        launcher_environment,
        "--unset",
        "COMPILATION_CONFIG_JSON",
    )

    # Then: generated compilation config is used instead of validating absence.
    assert result.returncode == 0, result.stderr
    assert (
        _resolved_field(result.stdout, "resolved_compilation_config_json_source")
        == "generated"
    )
    assert _resolved_field(result.stdout, "resolved_compilation_config_json")


@pytest.mark.parametrize(
    (
        "gpu_devices",
        "tp_size",
        "expected_gpu_devices",
        "expected_tp_size",
        "expected_tp_source",
    ),
    [
        ("0,1", None, "0,1", "2", "derived"),
        ("0, 1", None, "0,1", "2", "derived"),
        (" 0 , 1 ", "2", "0,1", "2", "env"),
    ],
    ids=["canonical", "whitespace", "explicit-tp"],
)
def test_print_config_normalizes_resolved_gpu_devices_and_tp_size(
    launcher_environment: dict[str, str],
    gpu_devices: str,
    tp_size: str | None,
    expected_gpu_devices: str,
    expected_tp_size: str,
    expected_tp_source: str,
) -> None:
    # Given: legal GPU lists with derived or explicit tensor parallelism.
    launcher_environment["GPU_DEVICES"] = gpu_devices
    if tp_size is None:
        launcher_environment.pop("TP_SIZE")
    else:
        launcher_environment["TP_SIZE"] = tp_size

    # When: the launcher resolves the final non-interactive configuration.
    result = _run_launcher(launcher_environment)

    # Then: the printed machine contract contains canonical devices and TP provenance.
    assert result.returncode == 0, result.stderr
    assert (
        _resolved_field(result.stdout, "resolved_gpu_devices") == expected_gpu_devices
    )
    assert _resolved_field(result.stdout, "resolved_tp_size") == expected_tp_size
    assert (
        _resolved_field(result.stdout, "resolved_tp_size_source") == expected_tp_source
    )


@pytest.mark.parametrize(
    "gpu_devices",
    [
        "",
        ",0,1",
        "0,,1",
        "0,1,",
        "-1",
        "x",
        "0,0",
        "GPU-uuid",
        "MIG-GPU-uuid/1/0",
    ],
    ids=[
        "empty",
        "leading-empty-item",
        "empty-item",
        "trailing-empty-item",
        "negative",
        "non-numeric",
        "duplicate",
        "uuid",
        "mig",
    ],
)
def test_print_config_rejects_invalid_resolved_gpu_devices(
    launcher_environment: dict[str, str], gpu_devices: str
) -> None:
    # Given: each unsupported resolved GPU device-list form.
    launcher_environment["GPU_DEVICES"] = gpu_devices
    launcher_environment.pop("TP_SIZE")

    # When: the launcher resolves configuration before building server arguments.
    result = _run_launcher(launcher_environment)

    # Then: every form fails at the stable GPU contract boundary.
    assert result.returncode != 0
    assert "GPU_DEVICES must be" in result.stderr
    assert "Starting vLLM server" not in result.stdout


@pytest.mark.parametrize(
    "tp_size",
    ["", "0", "two", "1"],
    ids=["empty", "zero", "non-numeric", "mismatch"],
)
def test_print_config_rejects_invalid_explicit_tp_size(
    launcher_environment: dict[str, str], tp_size: str
) -> None:
    # Given: invalid explicit tensor-parallel values for two resolved GPUs.
    launcher_environment["GPU_DEVICES"] = "0,1"
    launcher_environment["TP_SIZE"] = tp_size

    # When: the launcher resolves configuration before building server arguments.
    result = _run_launcher(launcher_environment)

    # Then: it rejects invalid format, zero, and device-count mismatch uniformly.
    assert result.returncode != 0
    assert "TP_SIZE must be" in result.stderr
    assert "Starting vLLM server" not in result.stdout


def test_print_config_prefers_cli_gpu_devices_over_environment_and_default(
    launcher_environment: dict[str, str]
) -> None:
    # Given: distinct legal environment and CLI GPU values with no explicit TP.
    launcher_environment["GPU_DEVICES"] = "4,7"
    launcher_environment.pop("TP_SIZE")

    # When: CLI provides a higher-precedence GPU selection.
    result = _run_launcher(launcher_environment, "--gpu-devices", "0, 1")

    # Then: CLI wins, the list is canonical, and TP is derived from that selection.
    assert result.returncode == 0, result.stderr
    assert _resolved_field(result.stdout, "resolved_gpu_devices") == "0,1"
    assert _resolved_field(result.stdout, "resolved_gpu_devices_source") == "cli"
    assert _resolved_field(result.stdout, "resolved_tp_size") == "2"
    assert _resolved_field(result.stdout, "resolved_tp_size_source") == "derived"


def test_print_config_uses_default_gpu_devices_when_unset(
    launcher_environment: dict[str, str]
) -> None:
    # Given: no explicit GPU or TP environment values.
    launcher_environment.pop("GPU_DEVICES")
    launcher_environment.pop("TP_SIZE")

    # When: the launcher resolves its CPU-only default device selection.
    result = _run_launcher(launcher_environment)

    # Then: the default list is normalized and TP is derived from its cardinality.
    assert result.returncode == 0, result.stderr
    assert _resolved_field(result.stdout, "resolved_gpu_devices") == "0,1"
    assert _resolved_field(result.stdout, "resolved_gpu_devices_source") == "default"
    assert _resolved_field(result.stdout, "resolved_tp_size") == "2"
    assert _resolved_field(result.stdout, "resolved_tp_size_source") == "derived"


def test_sourcing_launcher_does_not_start_a_service(
    launcher_environment: dict[str, str]
) -> None:
    # Given: the launcher sourced as a shell library.

    # When: Bash evaluates the source command.
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "--", str(_LAUNCHER)],
        cwd=_ROOT,
        env=launcher_environment,
        capture_output=True,
        check=False,
        text=True,
    )

    # Then: no server launch path runs.
    assert result.returncode == 0, result.stderr
    assert "Starting vLLM server" not in result.stdout
