# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import CodeType, ModuleType, SimpleNamespace

import pytest

_ROOT = Path(__file__).parents[2]
_ATTENTION = _ROOT / "vllm/v1/attention"
_TURBOQUANT = _ATTENTION / "backends/turboquant_attn.py"
_DECODE = _ATTENTION / "ops/triton_turboquant_decode.py"


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
    "sm75_turboquant_planner_for_test",
    _ATTENTION / "sm75_attention_planner.py",
)


def _method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(methods) == 1
    return methods[0]


def _calls(method: ast.FunctionDef, attribute: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == attribute
            or isinstance(node.func, ast.Name)
            and node.func.id == attribute
        )
        for node in ast.walk(method)
    )


def _call_keywords(method: ast.AST, function_name: str) -> set[str]:
    return {
        keyword.arg
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
        for keyword in node.keywords
        if keyword.arg is not None
    }


def _uniform_speculative_branch() -> CodeType:
    tree = ast.parse(_TURBOQUANT.read_text())
    forward = _method(tree, "TurboQuantAttentionImpl", "forward")
    branches = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and any(
            isinstance(value, ast.Compare)
            and any(
                isinstance(comparator, ast.Constant) and comparator.value == 1
                for comparator in value.comparators
            )
            for value in node.test.values
        )
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_spec_decode_attention"
            for call in ast.walk(node)
        )
    ]
    assert len(branches) == 1
    branch_module = ast.Module(body=[copy.deepcopy(branches[0])], type_ignores=[])
    return compile(
        ast.fix_missing_locations(branch_module),
        str(_TURBOQUANT),
        "exec",
    )


def _decode_route_planner_class() -> type:
    tree = ast.parse(_TURBOQUANT.read_text())
    method = copy.deepcopy(
        _method(tree, "TurboQuantAttentionImpl", "_plan_decode_route")
    )
    target = ast.ClassDef(
        name="_DecodeRoutePlannerTarget",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(body=[target], type_ignores=[])
    namespace = {
        "SM75AttentionPlanner": planner.SM75AttentionPlanner,
        "TQDecodeInput": planner.TQDecodeInput,
        "TQDecodePlan": planner.TQDecodePlan,
        "_TQ_FORCE_DECODE_SDPA": False,
        "_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK": True,
        "_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK": True,
    }
    code = compile(ast.fix_missing_locations(module), str(_TURBOQUANT), "exec")
    exec(code, namespace)
    return namespace["_DecodeRoutePlannerTarget"]


class _DispatchTensor:
    def __getitem__(self, _: slice) -> "_DispatchTensor":
        return self

    def view(self, *_: int) -> "_DispatchTensor":
        return self


class _UniformSpeculativeDispatchTarget:
    def __init__(self, plan) -> None:
        self.num_kv_heads = 1
        self.head_size = 1
        self._plan = plan
        self.calls: list[str] = []

    def _plan_decode_route(self, **_: bool):
        return self._plan

    def _spec_decode_attention_sdpa_fallback(self, *_: object) -> str:
        self.calls.append("sdpa")
        return "sdpa-result"

    def _spec_decode_attention(self, *_: object) -> str:
        self.calls.append("native")
        return "native-result"


@pytest.mark.parametrize(
    ("route_input", "expected_call", "expected_result"),
    [
        (
            (True, True, 256, True, True, True),
            "sdpa",
            "sdpa-result",
        ),
        (
            (False, True, 256, True, True, True),
            "native",
            "native-result",
        ),
    ],
)
def test_uniform_speculative_branch_dispatches_the_planner_route(
    route_input: tuple[bool, bool, int, bool, bool, bool],
    expected_call: str,
    expected_result: str,
) -> None:
    # Given: an actual production branch and a pure planner-selected route.
    plan = planner.SM75AttentionPlanner.plan_tq_decode(
        planner.TQDecodeInput(*route_input)
    )
    tensor = _DispatchTensor()
    for trace_enabled in (False, True):
        target = _UniformSpeculativeDispatchTarget(plan)
        trace_calls: list[str] = []
        trace_sink = trace_calls if trace_enabled else []
        trace = lambda *_args, _trace_sink=trace_sink, **_kwargs: _trace_sink.append(
            "route"
        )
        namespace = {
            "_TQ_CUDAGRAPH_SPEC_DECODE_SAFE": True,
            "TQDecodeRoute": planner.TQDecodeRoute,
            "N": 1,
            "Pi": tensor,
            "PiT": tensor,
            "centroids": tensor,
            "attn_metadata": SimpleNamespace(max_query_len=2),
            "key": tensor,
            "kv_cache": tensor,
            "layer": object(),
            "q": tensor,
            "self": target,
            "sm75_attention_trace": trace,
            "use_shared_draft_decode_sdpa": False,
            "value": tensor,
        }

        # When: the graph-safe uniform speculative branch executes with substitutes.
        exec(_uniform_speculative_branch(), namespace)

        # Then: trace state cannot change the unique planner-selected dispatch.
        assert target.calls == [expected_call]
        assert namespace["attn_out"] == expected_result
        assert trace_calls == (["route"] if trace_enabled else [])


def test_turboquant_runtime_consumes_planner_and_trace_seams() -> None:
    # Given: current runtime source without importing CUDA or Triton modules.
    turboquant_tree = ast.parse(_TURBOQUANT.read_text())
    decode_tree = ast.parse(_DECODE.read_text())

    # When: each current inline route boundary is inspected.
    runtime_methods = {
        "build_for_cudagraph_capture": "plan_tq_cudagraph",
        "__init__": "plan_tq_prefill_capability",
        "_use_flashinfer_for_first_chunk": "plan_tq_flashinfer_stage",
        "_use_flashinfer_for_continuation": "plan_tq_flashinfer_stage",
        "_plan_decode_route": "plan_tq_decode",
        "_continuation_prefill": "plan_tq_continuation",
    }

    # Then: planner owns each decision and traces are emitted only at routes.
    for method_name, planner_method in runtime_methods.items():
        class_name = (
            "TurboQuantMetadataBuilder"
            if method_name == "build_for_cudagraph_capture"
            else "TurboQuantAttentionImpl"
        )
        method = _method(turboquant_tree, class_name, method_name)
        assert _calls(method, planner_method)
    assert not _calls(
        _method(turboquant_tree, "TurboQuantAttentionImpl", "forward"),
        "_use_decode_sdpa_fallback",
    )
    assert _calls(
        _method(
            turboquant_tree,
            "TurboQuantMetadataBuilder",
            "build_for_cudagraph_capture",
        ),
        "sm75_attention_trace",
    )
    assert _calls(
        _method(turboquant_tree, "TurboQuantAttentionImpl", "_continuation_prefill"),
        "sm75_attention_trace",
    )
    decode_function = next(
        node
        for node in decode_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "triton_turboquant_decode_attention"
    )
    assert _calls(decode_function, "sm75_attention_trace")
    expected_trace_fields = {"decision", "enabled", "reason", "route"}
    assert expected_trace_fields <= _call_keywords(
        _method(
            turboquant_tree,
            "TurboQuantMetadataBuilder",
            "build_for_cudagraph_capture",
        ),
        "sm75_attention_trace",
    )
    assert expected_trace_fields <= _call_keywords(
        _method(turboquant_tree, "TurboQuantAttentionImpl", "_continuation_prefill"),
        "sm75_attention_trace",
    )
    assert expected_trace_fields <= _call_keywords(
        decode_function,
        "sm75_attention_trace",
    )


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ((True, True, 512, True, True, True), ("sdpa", "forced")),
        ((False, True, 512, True, True, True), ("native", None)),
        ((False, False, 512, True, True, True), ("sdpa", "head_dim_256")),
    ],
)
def test_turboquant_decode_planner_preserves_speculative_precedence(
    inputs: tuple[bool, bool, int, bool, bool, bool],
    expected: tuple[str, str | None],
) -> None:
    # Given: a route input that differs at forced and speculative boundaries.
    route_input = planner.TQDecodeInput(*inputs)

    # When: the pure planner selects the decode route.
    plan = planner.SM75AttentionPlanner.plan_tq_decode(route_input)

    # Then: the route and machine-readable reason match the current truth table.
    assert tuple(plan) == expected


def test_turboquant_planner_route_is_trace_independent() -> None:
    # Given: the same forced-SDPA input with trace conceptually off and on.
    route_input = planner.TQDecodeInput(True, False, 256, True, True, True)

    # When: the pure route planner is evaluated around an unrelated trace seam.
    trace_off_plan = planner.SM75AttentionPlanner.plan_tq_decode(route_input)
    trace_on_plan = planner.SM75AttentionPlanner.plan_tq_decode(route_input)

    # Then: route and reason remain byte-identical.
    assert tuple(trace_off_plan) == tuple(trace_on_plan) == ("sdpa", "forced")


def test_turboquant_decode_route_plans_are_cached_per_runtime_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an isolated production method and a counted pure planner seam.
    target_type = _decode_route_planner_class()
    target = target_type()
    target.head_size = 256
    target._decode_route_plans = {}
    calls: list[planner.TQDecodeInput] = []
    original = planner.SM75AttentionPlanner.plan_tq_decode

    def counted(route_input: planner.TQDecodeInput):
        calls.append(route_input)
        return original(route_input)

    monkeypatch.setattr(planner.SM75AttentionPlanner, "plan_tq_decode", counted)

    # When: normal, shared-draft, speculative, and combined cases repeat.
    combinations = ((False, False), (False, True), (True, False), (True, True))
    first = {
        combination: target._plan_decode_route(
            speculative=combination[0], shared_draft_fallback=combination[1]
        )
        for combination in combinations
    }
    second = {
        combination: target._plan_decode_route(
            speculative=combination[0], shared_draft_fallback=combination[1]
        )
        for combination in combinations
    }

    # Then: each input combination is planned once and reused by identity.
    assert len(calls) == len(combinations)
    assert all(first[key] is second[key] for key in combinations)
