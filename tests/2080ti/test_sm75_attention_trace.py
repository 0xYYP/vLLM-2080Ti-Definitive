# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_TRACE_MODULE_NAME = "vllm.sm75_attention_trace"
_TRACE_MODULE_PATH = Path(__file__).parents[2] / "vllm/sm75_attention_trace.py"
_ENVS_MODULE_NAME = "vllm.envs"
_ENVS_MODULE_PATH = Path(__file__).parents[2] / "vllm/envs.py"


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_vllm(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Given
    vllm_package = ModuleType("vllm")
    vllm_package.__path__ = []
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    envs_module = ModuleType(_ENVS_MODULE_NAME)
    envs_module.VLLM_SM75_ATTENTION_TRACE = False
    envs_module.VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS = 64
    vllm_package.envs = envs_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_package)
    monkeypatch.setitem(sys.modules, _ENVS_MODULE_NAME, envs_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    # When
    yield envs_module

    # Then
    sys.modules.pop(_TRACE_MODULE_NAME, None)
    sys.modules.pop(_ENVS_MODULE_NAME, None)


def _read_trace_environment(variable: str, value: str | None) -> str:
    environment = os.environ.copy()
    if value is None:
        environment.pop(variable, None)
    else:
        environment[variable] = value

    script = (
        "import importlib.util; "
        "spec = importlib.util.spec_from_file_location("
        f"'isolated_envs', {_ENVS_MODULE_PATH.as_posix()!r}); "
        "envs = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(envs); "
        f"print(envs.environment_variables[{variable!r}]())"
    )
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True,
        check=False,
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def trace_module(isolated_vllm: ModuleType) -> ModuleType:
    del isolated_vllm
    if not _TRACE_MODULE_PATH.is_file():
        pytest.fail("SM75 trace helper 缺失")

    module = _load_module(_TRACE_MODULE_NAME, _TRACE_MODULE_PATH)
    module._reset_sm75_attention_trace_for_tests()
    yield module
    module._reset_sm75_attention_trace_for_tests()
    sys.modules.pop(_TRACE_MODULE_NAME, None)


def test_trace_is_silent_when_disabled(
    caplog: pytest.LogCaptureFixture,
    trace_module: ModuleType,
) -> None:
    # Given
    caplog.set_level("INFO", logger=_TRACE_MODULE_NAME)

    # When
    trace_module.sm75_attention_trace("decode", path="triton")

    # Then
    assert caplog.records == []


def test_trace_deduplicates_fields_regardless_of_order(
    caplog: pytest.LogCaptureFixture,
    trace_module: ModuleType,
) -> None:
    # Given
    trace_module.envs.VLLM_SM75_ATTENTION_TRACE = True
    trace_module.envs.VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS = 4
    caplog.set_level("INFO", logger=_TRACE_MODULE_NAME)

    # When
    trace_module.sm75_attention_trace("decode", path="triton", splits=16)
    trace_module.sm75_attention_trace("decode", splits=16, path="triton")

    # Then
    assert len(caplog.records) == 1
    trace_event = caplog.records[0].sm75_attention_trace
    assert trace_event.event == "decode"
    assert trace_event.fields == (("path", "triton"), ("splits", 16))
    assert caplog.records[0].getMessage() == (
        "SM75 attention trace event=decode path=triton splits=16"
    )


def test_trace_ignores_new_unique_events_at_its_limit(
    caplog: pytest.LogCaptureFixture,
    trace_module: ModuleType,
) -> None:
    # Given
    trace_module.envs.VLLM_SM75_ATTENTION_TRACE = True
    trace_module.envs.VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS = 2
    caplog.set_level("INFO", logger=_TRACE_MODULE_NAME)

    # When
    trace_module.sm75_attention_trace("decode", path="triton")
    trace_module.sm75_attention_trace("prefill", path="flashinfer")
    trace_module.sm75_attention_trace("sync", point="before_slot_mapping")

    # Then
    assert [record.sm75_attention_trace.event for record in caplog.records] == [
        "decode",
        "prefill",
    ]


def test_trace_reset_allows_a_previous_event_again(
    caplog: pytest.LogCaptureFixture,
    trace_module: ModuleType,
) -> None:
    # Given
    trace_module.envs.VLLM_SM75_ATTENTION_TRACE = True
    caplog.set_level("INFO", logger=_TRACE_MODULE_NAME)
    trace_module.sm75_attention_trace("decode", path="triton")

    # When
    trace_module._reset_sm75_attention_trace_for_tests()
    trace_module.sm75_attention_trace("decode", path="triton")

    # Then
    assert len(caplog.records) == 2


def test_trace_environment_uses_historical_default_and_lower_bound(
) -> None:
    # Given

    # When
    enabled = _read_trace_environment("VLLM_SM75_ATTENTION_TRACE", None)
    max_events = _read_trace_environment("VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS", "-3")

    # Then
    assert enabled == "False"
    assert max_events == "1"


def test_trace_environment_rejects_malformed_event_limit(
) -> None:
    # Given

    # When / Then
    environment = os.environ.copy()
    environment["VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS"] = "not-an-int"
    result = subprocess.run(
        [
            "python3",
            "-c",
            "import importlib.util; "
            "spec = importlib.util.spec_from_file_location("
            f"'isolated_envs', {_ENVS_MODULE_PATH.as_posix()!r}); "
            "envs = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(envs); "
            "envs.environment_variables['VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS']()",
        ],
        capture_output=True,
        check=False,
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr
