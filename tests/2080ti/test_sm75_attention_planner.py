# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).parents[2]
_ATTN = _ROOT / "vllm/v1/attention"


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
    "vllm.v1.attention": _ATTN,
}.items():
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

_load(
    "vllm.v1.attention.sm75_attention_planner_types",
    _ATTN / "sm75_attention_planner_types.py",
)
planner = _load("sm75_planner_for_test", _ATTN / "sm75_attention_planner.py")
P = planner.SM75AttentionPlanner

_TQ_GRAPH = "turboquant_attn.py:502-513 safe and 1<q<=threshold"
_TQ_CAP = "turboquant_attn.py:623-634 capability predicates"
_TQ_FIRST = "turboquant_attn.py:867-872 first q>=min"
_TQ_CONT = "turboquant_attn.py:874-879 continuation q>=min"
_TQ_SPEC = "turboquant_attn.py:1181-1197 spec precedence"
_TQ_DECODE = "turboquant_attn.py:1217-1227,2555-2560 fallbacks"
_TQ_PREFIX = "turboquant_attn.py:2021-2028 prefix-combine predicates"
_I_ELIG = "triton_attn.py:1219-1264 eligibility precedence"
_I_FIRST = "triton_attn.py:1274-1289 first-chunk route"
_I_BATCH = "triton_attn.py:1292 batch before query/seq"
_I_SMALL = "triton_attn.py:1294 q_len<min before seq/direct"
_I_SEQ = "triton_attn.py:1296 seq-lens after small query"
_I_CASCADE = "triton_attn.py:1300-1308 length-only cascade"
_I_RAGGED = "triton_attn.py:1320-1332,1483 ragged fallback"
_SYNC = "gpu_model_runner.py:430-434 safe or auto+TQ"


@pytest.mark.parametrize(
    ("source", "inputs", "expected"),
    [
        (_TQ_GRAPH, (True, 1, 8), (False, 1)),
        (_TQ_GRAPH, (True, 2, 8), (True, 2)),
        (_TQ_GRAPH, (True, 8, 8), (True, 8)),
        (_TQ_GRAPH, (True, 9, 8), (False, 1)),
    ],
)
def test_tq_cudagraph_truth_table(source, inputs, expected) -> None:
    plan = P.plan_tq_cudagraph(planner.TQCUDAGraphInput(*inputs))
    assert source and tuple(plan) == expected


@pytest.mark.parametrize(
    ("source", "inputs", "expected"),
    [
        (_TQ_CAP, (True, False, True, True, 128, 128), (False, "wrapper_unavailable")),
        (_TQ_CAP, (True, True, True, True, 127, 128),
         (False, "head_dim_below_sm75_threshold")),
        (_TQ_CAP, (True, True, True, True, 128, 128), (True, None)),
        # sm75_min_head_size<=0 means "no SM75 head-dim lower bound"
        # (flashinfer 0.6.8.x defaults MIN_HEAD_DIM to 0): any head size
        # is allowed on SM75, including ones below a positive threshold.
        (_TQ_CAP, (True, True, True, True, 128, 0), (True, None)),
        (_TQ_CAP, (True, True, True, True, 1, 0), (True, None)),
        (_TQ_CAP, (True, True, True, True, 256, -1), (True, None)),
        (_TQ_CAP, (True, True, True, False, 128, 0), (True, None)),
    ],
)
def test_tq_capability_truth_table(source, inputs, expected) -> None:
    plan = P.plan_tq_prefill_capability(planner.TQPrefillCapabilityInput(*inputs))
    assert source and tuple(plan) == expected


def test_tq_capability_rejects_invalid_head_size() -> None:
    """head_size <= 0 stays an invalid input even when the SM75 bound is 0."""
    with pytest.raises(planner.PlannerInputError):
        P.plan_tq_prefill_capability(
            planner.TQPrefillCapabilityInput(
                True, True, True, True, 0, 0
            )
        )


@pytest.mark.parametrize(
    ("source", "stage", "query_len", "expected"),
    [
        (_TQ_FIRST, planner.TQPrefillStage.FIRST_CHUNK, 255,
         (False, "query_len_below_sm75_threshold")),
        (_TQ_FIRST, planner.TQPrefillStage.FIRST_CHUNK, 256, (True, None)),
        (_TQ_CONT, planner.TQPrefillStage.CONTINUATION, 127,
         (False, "query_len_below_sm75_threshold")),
        (_TQ_CONT, planner.TQPrefillStage.CONTINUATION, 128, (True, None)),
    ],
)
def test_tq_flashinfer_threshold_truth_table(
    source, stage, query_len, expected
) -> None:
    inputs = planner.TQFlashInferInput(True, True, query_len, 256, 128)
    assert source and tuple(P.plan_tq_flashinfer_stage(inputs, stage)) == expected


@pytest.mark.parametrize(
    ("source", "inputs", "expected"),
    [
        (_TQ_SPEC, (True, True, 512, True, True, True), ("sdpa", "forced")),
        (_TQ_SPEC, (False, True, 512, True, True, True), ("native", None)),
        (_TQ_DECODE, (False, False, 512, True, True, True), ("sdpa", "head_dim_256")),
        (_TQ_DECODE, (False, False, 512, False, True, True), ("sdpa", "head_dim_512")),
        (_TQ_DECODE, (False, False, 128, False, False, True), ("sdpa", "shared_draft")),
    ],
)
def test_tq_decode_truth_table(source, inputs, expected) -> None:
    plan = P.plan_tq_decode(planner.TQDecodeInput(*inputs))
    assert source and tuple(plan) == expected


def _tq():
    return planner.TQContinuationInput(
        planner.TQPrefixCombineMode.AUTO,
        20_480,
        20_480,
        False,
        4,
        20_224,
        0,
        True,
    )


@pytest.mark.parametrize(
    ("source", "overrides", "expected"),
    [
        (_TQ_PREFIX, {"prefix_combine_mode": planner.TQPrefixCombineMode.OFF},
         (False, "none")),
        (_TQ_PREFIX, {"prefix_combine_mode": planner.TQPrefixCombineMode.ON,
         "sequence_len": 1}, (True, "shared")),
        (_TQ_PREFIX, {"sequence_len": 20_479}, (False, "none")),
        (_TQ_PREFIX, {"kv_cache_dim": 5}, (False, "none")),
        (_TQ_PREFIX, {"cached_len": 0}, (False, "none")),
        (_TQ_PREFIX, {"sliding_window": 1}, (False, "none")),
        (_TQ_PREFIX, {"force_sdpa": True}, (False, "none")),
        (_TQ_PREFIX, {"flashinfer_continuation": False}, (False, "none")),
        (_TQ_PREFIX, {}, (True, "shared")),
    ],
)
def test_tq_continuation_truth_table(source, overrides, expected) -> None:
    plan = P.plan_tq_continuation(_tq()._replace(**overrides))
    assert source and tuple(plan) == expected


def _int8():
    return planner.Int8KVRouteInput(
        True, True, "int8_per_token_head", True, False, False, False, False,
        True, True, True, 1, 4096, 4096, 0, False, True, 128, 65_536,
        False, True, True,
    )


@pytest.mark.parametrize(
    ("source", "field", "value", "reason"),
    [
        (_I_ELIG, "enabled", False, "disabled"),
        (_I_ELIG, "kv_cache_dtype", "fp8", "not_int8_per_token_head"),
        (_I_ELIG, "decoder_attention", False, "not_decoder_attention"),
        (_I_ELIG, "has_alibi", True, "alibi"),
        (_I_ELIG, "has_sinks", True, "attention_sinks"),
        (_I_ELIG, "has_sliding_window", True, "sliding_window"),
        (_I_ELIG, "has_logits_soft_cap", True, "logits_soft_cap"),
        (_I_ELIG, "qkv_fp16", False, "non_fp16_qkv"),
        (_I_ELIG, "has_computed_tokens", False, "missing_num_computed_tokens"),
    ],
)
def test_int8_eligibility_truth_table(source, field, value, reason) -> None:
    plan = P.plan_int8kv_route(_int8()._replace(**{field: value}))
    assert source and tuple(plan) == ("disabled", reason, False, False)


@pytest.mark.parametrize(
    ("source", "overrides", "expected"),
    [
        (_I_FIRST, {}, ("ragged", None, True, False)),
        (_I_FIRST, {"first_chunk_dequant": True}, ("continuation", None, False, True)),
        (_I_BATCH, {"computed_tokens": 1, "continuation_dequant": False},
         ("disabled", "prefix_or_cached_kv", False, False)),
        (_I_BATCH, {"computed_tokens": 1, "request_count": 2,
         "query_len": 127, "has_sequence_len": False},
         ("disabled", "continuation_batch_not_1", False, False)),
        (_I_SMALL, {"computed_tokens": 1, "query_len": 127,
         "has_sequence_len": False},
         ("disabled", "continuation_q_too_small", False, False)),
        (_I_SMALL, {"computed_tokens": 1, "query_len": 127,
         "direct_paged": False},
         ("disabled", "continuation_q_too_small", False, False)),
        (_I_SEQ, {"computed_tokens": 1, "has_sequence_len": False},
         ("disabled", "continuation_missing_seq_lens_cpu", False, False)),
        (_I_CASCADE, {"computed_tokens": 1, "sequence_len": 65_537},
         ("disabled", "continuation_too_long", False, False)),
        (_I_CASCADE, {"computed_tokens": 1, "sequence_len": 65_537,
         "cascade_dequant": True}, ("cascade", None, True, False)),
        (_I_CASCADE, {"computed_tokens": 1}, ("continuation", None, True, False)),
        (_I_CASCADE, {"computed_tokens": 1, "cascade_dequant": True},
         ("continuation", None, True, False)),
        (_I_RAGGED, {"ragged_enabled": False},
         ("disabled", "ragged_prefill_disabled", True, False)),
    ],
)
def test_int8_route_truth_table(source, overrides, expected) -> None:
    plan = P.plan_int8kv_route(_int8()._replace(**overrides))
    assert source and tuple(plan) == expected


@pytest.mark.parametrize(
    ("source", "mode", "dtype", "expected"),
    [
        (_SYNC, planner.SpecSyncMode.AUTO, "turboquant_k8v4",
         (True, "turboquant_auto")),
        (_SYNC, planner.SpecSyncMode.AUTO, "int8_per_token_head",
         (False, "non_turboquant_auto")),
        (_SYNC, planner.SpecSyncMode.SAFE, "int8_per_token_head", (True, "safe_mode")),
        (_SYNC, planner.SpecSyncMode.NOSYNC, "turboquant_k8v4", (False, "nosync_mode")),
    ],
)
def test_spec_sync_truth_table(source, mode, dtype, expected) -> None:
    assert source and tuple(P.plan_spec_sync(mode, dtype)) == expected


def test_current_surfaces_and_malformed_inputs() -> None:
    assert set(planner.TQContinuationInput._fields) == {
        "prefix_combine_mode", "prefix_combine_min_tokens", "sequence_len",
        "force_sdpa", "kv_cache_dim", "cached_len", "sliding_window",
        "flashinfer_continuation",
    }
    assert planner.Int8KVRoutePlan._fields == (
        "route", "reason", "direct_paged_attempt", "force_first_chunk_dequant"
    )
    with pytest.raises(planner.PlannerInputError):
        P.plan_tq_cudagraph(planner.TQCUDAGraphInput(True, -1, 8))
    with pytest.raises(planner.PlannerInputError):
        P.plan_int8kv_route(_int8()._replace(computed_tokens=-1))
    with pytest.raises(TypeError):
        planner.TQCUDAGraphInput._make((True, 1))
    with pytest.raises(ValueError):
        planner.SpecSyncMode("unsafe")
