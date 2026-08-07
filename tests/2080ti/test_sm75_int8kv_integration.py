# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).parents[2]
_ATTN = _ROOT / "vllm/v1/attention"
_BACKEND = _ATTN / "backends/triton_attn.py"
_TRACE = _ROOT / "vllm/sm75_attention_trace.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package_name, package_path in {
    "vllm": _ROOT / "vllm",
    "vllm.v1": _ROOT / "vllm/v1",
    "vllm.v1.attention": _ATTN,
}.items():
    package = ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

_load(
    "vllm.v1.attention.sm75_attention_planner_types",
    _ATTN / "sm75_attention_planner_types.py",
)
planner = _load("sm75_int8kv_planner_for_test", _ATTN / "sm75_attention_planner.py")


def _int8_input(**overrides: bool | int | str):
    inputs = planner.Int8KVRouteInput(
        True,
        True,
        "int8_per_token_head",
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        1,
        128,
        4096,
        1,
        False,
        True,
        128,
        65_536,
        False,
        False,
        True,
    )
    return inputs._replace(**overrides)


def test_runtime_consumes_planner_and_structured_trace_seam() -> None:
    tree = ast.parse(_BACKEND.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_try_int8kv_fa_prefill"
    )
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name | ast.Attribute)
    }
    trace_fields = {
        keyword.arg
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sm75_attention_trace"
        for keyword in node.keywords
        if keyword.arg is not None
    }

    assert "SM75AttentionPlanner" in imported
    assert "Int8KVRouteInput" in imported
    assert "sm75_attention_trace" in imported
    assert "plan_int8kv_route" in calls
    assert "sm75_attention_trace" in calls
    assert {
        "route",
        "reason",
        "first_chunk",
        "request_count",
        "query_len",
        "sequence_len",
        "direct_paged_attempt",
    } <= trace_fields


@pytest.mark.parametrize(
    ("feature_enabled", "per_token_head"),
    [(False, True), (True, False)],
)
def test_int8kv_entry_gate_does_not_touch_poison_metadata(
    feature_enabled: bool,
    per_token_head: bool,
) -> None:
    tree = ast.parse(_BACKEND.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_try_int8kv_fa_prefill"
    )
    namespace = {
        "_INT8KV_FA_PREFILL": feature_enabled,
    }
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                method,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, str(_BACKEND), "exec"), namespace)
    entry = namespace["_try_int8kv_fa_prefill"]

    class PoisonMetadata:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"feature gate touched metadata: {name}")

    result = entry(
        type("Stub", (), {"_is_per_token_head_quant": per_token_head})(),
        None,
        None,
        None,
        None,
        None,
        PoisonMetadata(),
        0,
        None,
        None,
    )

    assert result is False


def test_trace_is_observational_for_each_int8kv_plan() -> None:
    disabled = planner.SM75AttentionPlanner.plan_int8kv_route(
        _int8_input(has_alibi=True)
    )
    direct_attempt = planner.SM75AttentionPlanner.plan_int8kv_route(
        _int8_input(direct_paged=True)
    )
    cascade = planner.SM75AttentionPlanner.plan_int8kv_route(
        _int8_input(sequence_len=65_537, cascade_dequant=True)
    )

    for trace_enabled in (False, True):
        observed = [
            (disabled.route.value, disabled.reason, disabled.direct_paged_attempt),
            (
                direct_attempt.route.value,
                direct_attempt.reason,
                direct_attempt.direct_paged_attempt,
            ),
            (cascade.route.value, cascade.reason, cascade.direct_paged_attempt),
        ]
        assert trace_enabled in (False, True)
        assert observed == [
            ("disabled", "alibi", False),
            ("continuation", None, False),
            ("cascade", None, False),
        ]


def test_trace_route_payload_is_bounded_and_machine_observable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    package = ModuleType("vllm")
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    envs_module = ModuleType("vllm.envs")
    envs_module.VLLM_SM75_ATTENTION_TRACE = True
    envs_module.VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS = 1
    package.envs = envs_module
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_module)
    trace = _load("sm75_int8kv_trace_for_test", _TRACE)
    caplog.set_level("INFO", logger="sm75_int8kv_trace_for_test")

    plan = planner.SM75AttentionPlanner.plan_int8kv_route(
        _int8_input(direct_paged=True)
    )
    trace.sm75_attention_trace(
        "int8kv_prefill_route",
        route=plan.route.value,
        reason=plan.reason,
        first_chunk=False,
        request_count=1,
        query_len=128,
        sequence_len=4096,
        direct_paged_attempt=plan.direct_paged_attempt,
    )
    trace.sm75_attention_trace(
        "int8kv_prefill_route",
        route="disabled",
        reason="alibi",
        first_chunk=False,
        request_count=1,
        query_len=128,
        sequence_len=4096,
        direct_paged_attempt=False,
    )

    assert len(caplog.records) == 1
    fields = dict(caplog.records[0].sm75_attention_trace.fields)
    assert fields == {
        "direct_paged_attempt": False,
        "first_chunk": False,
        "query_len": 128,
        "reason": None,
        "request_count": 1,
        "route": "continuation",
        "sequence_len": 4096,
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"enabled": False}, ("disabled", "disabled", False, False)),
        (
            {"kv_cache_dtype": "fp8"},
            ("disabled", "not_int8_per_token_head", False, False),
        ),
        (
            {"decoder_attention": False},
            ("disabled", "not_decoder_attention", False, False),
        ),
        ({"has_alibi": True}, ("disabled", "alibi", False, False)),
        ({"has_sinks": True}, ("disabled", "attention_sinks", False, False)),
        ({"has_sliding_window": True}, ("disabled", "sliding_window", False, False)),
        ({"has_logits_soft_cap": True}, ("disabled", "logits_soft_cap", False, False)),
        ({"qkv_fp16": False}, ("disabled", "non_fp16_qkv", False, False)),
        (
            {"has_computed_tokens": False},
            ("disabled", "missing_num_computed_tokens", False, False),
        ),
        ({"computed_tokens": 0}, ("ragged", None, False, False)),
        (
            {"computed_tokens": 0, "first_chunk_dequant": True},
            ("continuation", None, False, True),
        ),
        ({"request_count": 2}, ("disabled", "continuation_batch_not_1", False, False)),
        ({"query_len": 127}, ("continuation", None, False, False)),
        (
            {"has_sequence_len": False},
            ("disabled", "continuation_missing_seq_lens_cpu", False, False),
        ),
        ({"sequence_len": 65_537}, ("disabled", "continuation_too_long", False, False)),
        (
            {"sequence_len": 65_537, "cascade_dequant": True},
            ("cascade", None, False, False),
        ),
        ({"direct_paged": True}, ("continuation", None, False, False)),
        (
            {"ragged_enabled": False},
            ("disabled", "ragged_prefill_disabled", False, False),
        ),
    ],
)
def test_current_int8kv_truth_table(
    overrides: dict[str, bool | int | str],
    expected: tuple[str, str | None, bool, bool],
) -> None:
    plan = planner.SM75AttentionPlanner.plan_int8kv_route(_int8_input(**overrides))
    assert tuple(plan) == expected
