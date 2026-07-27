# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).parents[2]
_ATTENTION = _ROOT / "vllm/v1/attention"
_RUNNER = _ROOT / "vllm/v1/worker/gpu_model_runner.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name, path in {
    "vllm": _ROOT / "vllm",
    "vllm.v1": _ROOT / "vllm/v1",
    "vllm.v1.attention": _ATTENTION,
}.items():
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

_load(
    "vllm.v1.attention.sm75_attention_planner_types",
    _ATTENTION / "sm75_attention_planner_types.py",
)
planner = _load(
    "sm75_spec_sync_planner_for_test", _ATTENTION / "sm75_attention_planner.py"
)
_TRACE_MODULE_NAME = "vllm.sm75_attention_trace"
_TRACE_MODULE_PATH = _ROOT / "vllm/sm75_attention_trace.py"


def _call_name(call: ast.Call) -> str | None:
    match call.func:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=name):
            return name
        case _:
            return None


def _trace_calls(module: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and _call_name(node) == "sm75_attention_trace"
    ]


def _method(module: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef) and (
                    member.name == method_name
                ):
                    return member
    raise AssertionError(f"missing {class_name}.{method_name}")


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _load_isolated_trace(enabled: bool) -> ModuleType:
    envs_module = ModuleType("vllm.envs")
    envs_module.VLLM_SM75_ATTENTION_TRACE = enabled
    envs_module.VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS = 64
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    sys.modules["vllm"].envs = envs_module
    sys.modules["vllm.logger"] = logger_module
    sys.modules["vllm.envs"] = envs_module
    trace_module = _load(_TRACE_MODULE_NAME, _TRACE_MODULE_PATH)
    trace_module._reset_sm75_attention_trace_for_tests()
    return trace_module


@pytest.mark.parametrize(
    ("mode", "cache_dtype", "expected"),
    [
        ("auto", "turboquant_k8v4", True),
        ("auto", "fp8", False),
        ("safe", "fp8", True),
        ("nosync", "turboquant_k8v4", False),
    ],
)
def test_spec_sync_decision_matches_current_runtime_truth_table(
    mode: str, cache_dtype: str, expected: bool
) -> None:
    # Given: each current legal mode and cache-dtype combination.
    spec_mode = planner.SpecSyncMode(mode)

    # When: the pure policy plans the runtime decision.
    decision = planner.SM75AttentionPlanner.plan_spec_sync(spec_mode, cache_dtype)

    # Then: it retains the inline runtime policy truth table.
    assert decision.enabled is expected


def test_gpu_model_runner_routes_spec_sync_through_planner_and_trace() -> None:
    # Given: the GPU-heavy runner is parsed without importing CUDA dependencies.
    module = ast.parse(_RUNNER.read_text(encoding="utf-8"), filename=str(_RUNNER))

    # When: the runtime seam is inspected.
    runtime_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and _call_name(node) == "plan_spec_sync"
    ]
    trace_calls = _trace_calls(module)

    # Then: planner output owns the boolean, and both retained sync points trace it.
    assert len(runtime_calls) == 1
    assert len(trace_calls) == 2
    assert {keyword.arg for call in trace_calls for keyword in call.keywords} >= {
        "mode",
        "is_turboquant",
        "enabled",
        "sync_point",
    }

    wait_lines = [
        node.lineno
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and _call_name(node) == "wait_stream"
    ]
    synchronize_lines = [
        node.lineno
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and _call_name(node) == "synchronize"
    ]
    trace_lines = [call.lineno for call in trace_calls]
    assert any(
        trace_line < wait_line for trace_line in trace_lines for wait_line in wait_lines
    )
    assert any(
        trace_line < synchronize_line
        for trace_line in trace_lines
        for synchronize_line in synchronize_lines
    )


def test_runner_stores_planner_enabled_value() -> None:
    module = ast.parse(_RUNNER.read_text(encoding="utf-8"), filename=str(_RUNNER))
    initializer = _method(module, "GPUModelRunner", "__init__")

    assignments = [
        node
        for node in ast.walk(initializer)
        if isinstance(node, ast.Assign)
        and any(
            _attribute_chain(target) == ("self", "sm75_spec_syncs_enabled")
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    assignment = assignments[0]
    assert isinstance(assignment.value, ast.Attribute)
    assert _attribute_chain(assignment.value) == ("spec_sync_plan", "enabled")


def test_runner_waits_on_copy_stream_from_current_stream() -> None:
    module = ast.parse(_RUNNER.read_text(encoding="utf-8"), filename=str(_RUNNER))
    prepare_inputs = _method(module, "GPUModelRunner", "_prepare_inputs")
    waits = [
        node
        for node in ast.walk(prepare_inputs)
        if isinstance(node, ast.Call) and _call_name(node) == "wait_stream"
    ]

    assert len(waits) == 1
    wait = waits[0]
    assert len(wait.args) == 1
    assert _attribute_chain(wait.args[0]) == (
        "self",
        "valid_sampled_token_count_copy_stream",
    )
    assert isinstance(wait.func, ast.Attribute)
    assert isinstance(wait.func.value, ast.Call)
    assert _call_name(wait.func.value) == "current_stream"


def test_runner_copy_wait_binds_flag_stream_trace_and_wait() -> None:
    module = ast.parse(_RUNNER.read_text(encoding="utf-8"), filename=str(_RUNNER))
    prepare_inputs = _method(module, "GPUModelRunner", "_prepare_inputs")
    guarded_waits = [
        node
        for node in ast.walk(prepare_inputs)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call) and _call_name(call) == "wait_stream"
            for call in ast.walk(node)
        )
        and any(
            _attribute_chain(name) == ("self", "sm75_spec_syncs_enabled")
            for name in ast.walk(node.test)
            if isinstance(name, ast.Attribute)
        )
        and any(
            _attribute_chain(name)
            == ("self", "valid_sampled_token_count_copy_stream")
            for name in ast.walk(node.test)
            if isinstance(name, ast.Attribute)
        )
        and any(
            isinstance(call, ast.Call)
            and _call_name(call) == "sm75_attention_trace"
            for call in ast.walk(node)
        )
    ]

    assert len(guarded_waits) == 1
    wait = next(
        call
        for call in ast.walk(guarded_waits[0])
        if isinstance(call, ast.Call) and _call_name(call) == "wait_stream"
    )
    assert len(wait.args) == 1
    assert _attribute_chain(wait.args[0]) == (
        "self",
        "valid_sampled_token_count_copy_stream",
    )


def test_runner_guards_current_stream_sync_with_enabled_flag() -> None:
    module = ast.parse(_RUNNER.read_text(encoding="utf-8"), filename=str(_RUNNER))
    prepare_inputs = _method(module, "GPUModelRunner", "_prepare_inputs")
    guarded_syncs = [
        node
        for node in ast.walk(prepare_inputs)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call) and _call_name(call) == "synchronize"
            for call in ast.walk(node)
        )
        and any(
            _attribute_chain(name) == ("self", "sm75_spec_syncs_enabled")
            for name in ast.walk(node.test)
            if isinstance(name, ast.Attribute)
        )
    ]

    assert len(guarded_syncs) == 1


def test_spec_sync_trace_toggle_preserves_decision_and_emits_two_sync_points(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: an enabled and disabled isolated trace surface for the same inputs.
    spec_mode = planner.SpecSyncMode.AUTO
    disabled_trace = _load_isolated_trace(False)
    caplog.set_level("INFO", logger=_TRACE_MODULE_NAME)
    disabled_decision = planner.SM75AttentionPlanner.plan_spec_sync(
        spec_mode, "turboquant_k8v4"
    )
    for sync_point in ("copy_stream_wait", "current_stream_synchronize"):
        disabled_trace.sm75_attention_trace(
            "spec_sync",
            mode=spec_mode.value,
            is_turboquant=True,
            enabled=disabled_decision.enabled,
            sync_point=sync_point,
        )
    assert caplog.records == []

    # When: tracing is enabled without changing the planner inputs.
    sys.modules.pop(_TRACE_MODULE_NAME, None)
    enabled_trace = _load_isolated_trace(True)
    enabled_decision = planner.SM75AttentionPlanner.plan_spec_sync(
        spec_mode, "turboquant_k8v4"
    )
    for sync_point in ("copy_stream_wait", "current_stream_synchronize"):
        enabled_trace.sm75_attention_trace(
            "spec_sync",
            mode=spec_mode.value,
            is_turboquant=True,
            enabled=enabled_decision.enabled,
            sync_point=sync_point,
        )

    # Then: the decision is identical and enabled tracing exposes both sync points.
    assert enabled_decision.enabled is disabled_decision.enabled
    events = [record.sm75_attention_trace for record in caplog.records]
    assert len(events) == 2
    assert {dict(event.fields)["sync_point"] for event in events} == {
        "copy_stream_wait",
        "current_stream_synchronize",
    }
    for event in events:
        fields = dict(event.fields)
        assert fields["mode"] == "auto"
        assert fields["is_turboquant"] is True
        assert fields["enabled"] is True
    sys.modules.pop(_TRACE_MODULE_NAME, None)
