# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import assert_never

from vllm.v1.attention.sm75_attention_planner_types import (
    CapabilityPlan,
    Int8KVRoute,
    Int8KVRouteInput,
    Int8KVRoutePlan,
    PlannerInputError,
    SpecSyncMode,
    SpecSyncPlan,
    TQContinuationInput,
    TQContinuationPlan,
    TQCUDAGraphInput,
    TQCUDAGraphPlan,
    TQDecodeInput,
    TQDecodePlan,
    TQDecodeRoute,
    TQFlashInferInput,
    TQPrefillCapabilityInput,
    TQPrefillStage,
    TQPrefixCombineMode,
    WorkspacePolicy,
)


class SM75AttentionPlanner:
    @staticmethod
    def plan_tq_cudagraph(inputs: TQCUDAGraphInput) -> TQCUDAGraphPlan:
        if inputs.max_query_len < 0 or inputs.continuation_threshold < 1:
            raise PlannerInputError("invalid TurboQuant cudagraph shape")
        force_spec = (
            inputs.spec_decode_safe
            and 1 < inputs.max_query_len <= inputs.continuation_threshold
        )
        return TQCUDAGraphPlan(force_spec, inputs.max_query_len if force_spec else 1)

    @staticmethod
    def plan_tq_prefill_capability(
        inputs: TQPrefillCapabilityInput,
    ) -> CapabilityPlan:
        if inputs.head_size <= 0 or inputs.sm75_min_head_size <= 0:
            raise PlannerInputError("invalid TurboQuant head size")
        checks = (
            (not inputs.requested, "disabled"),
            (not inputs.wrapper_available, "wrapper_unavailable"),
            (not inputs.is_cuda, "non_cuda"),
            (
                inputs.is_sm75 and inputs.head_size < inputs.sm75_min_head_size,
                "head_dim_below_sm75_threshold",
            ),
        )
        for failed, reason in checks:
            if failed:
                return CapabilityPlan(False, reason)
        return CapabilityPlan(True, None)

    @staticmethod
    def plan_tq_flashinfer_stage(
        inputs: TQFlashInferInput, stage: TQPrefillStage
    ) -> CapabilityPlan:
        if inputs.query_len < 0:
            raise PlannerInputError("invalid TurboQuant query length")
        match stage:
            case TQPrefillStage.FIRST_CHUNK:
                threshold = inputs.first_chunk_min_query
            case TQPrefillStage.CONTINUATION:
                threshold = inputs.continuation_min_query
            case unreachable:
                assert_never(unreachable)
        if not inputs.available:
            return CapabilityPlan(False, "disabled")
        if inputs.is_sm75 and inputs.query_len < threshold:
            return CapabilityPlan(False, "query_len_below_sm75_threshold")
        return CapabilityPlan(True, None)

    @staticmethod
    def plan_tq_decode(inputs: TQDecodeInput) -> TQDecodePlan:
        if inputs.head_size <= 0:
            raise PlannerInputError("invalid TurboQuant head size")
        if inputs.force_sdpa:
            return TQDecodePlan(TQDecodeRoute.SDPA, "forced")
        if inputs.speculative:
            return TQDecodePlan(TQDecodeRoute.NATIVE, None)
        checks = (
            (inputs.d256_fallback and inputs.head_size >= 256, "head_dim_256"),
            (inputs.d512_fallback and inputs.head_size >= 512, "head_dim_512"),
            (inputs.shared_draft_fallback, "shared_draft"),
        )
        for enabled, reason in checks:
            if enabled:
                return TQDecodePlan(TQDecodeRoute.SDPA, reason)
        return TQDecodePlan(TQDecodeRoute.NATIVE, None)

    @staticmethod
    def plan_tq_continuation(inputs: TQContinuationInput) -> TQContinuationPlan:
        if (
            min(
                inputs.sequence_len,
                inputs.kv_cache_dim,
                inputs.cached_len,
                inputs.prefix_combine_min_tokens,
            )
            < 0
        ):
            raise PlannerInputError("invalid TurboQuant continuation shape")
        match inputs.prefix_combine_mode:
            case TQPrefixCombineMode.OFF:
                requested = False
            case TQPrefixCombineMode.ON:
                requested = True
            case TQPrefixCombineMode.AUTO:
                requested = inputs.sequence_len >= inputs.prefix_combine_min_tokens
            case unreachable:
                assert_never(unreachable)
        enabled = (
            requested
            and not inputs.force_sdpa
            and inputs.kv_cache_dim != 5
            and inputs.cached_len > 0
            and inputs.sliding_window <= 0
            and inputs.flashinfer_continuation
        )
        workspace = WorkspacePolicy.SHARED if enabled else WorkspacePolicy.NONE
        return TQContinuationPlan(enabled, workspace)

    @staticmethod
    def plan_int8kv_route(inputs: Int8KVRouteInput) -> Int8KVRoutePlan:
        if (
            min(
                inputs.request_count,
                inputs.query_len,
                inputs.sequence_len,
                inputs.computed_tokens,
                inputs.continuation_min_query,
                inputs.continuation_max_tokens,
            )
            < 0
        ):
            raise PlannerInputError("invalid INT8KV shape")
        reason = SM75AttentionPlanner._int8kv_eligibility(inputs)
        if reason is not None:
            return Int8KVRoutePlan(Int8KVRoute.DISABLED, reason, False, False)
        first_chunk = inputs.computed_tokens == 0
        if first_chunk:
            if inputs.first_chunk_dequant and inputs.request_count != 1:
                return Int8KVRoutePlan(
                    Int8KVRoute.DISABLED,
                    "first_chunk_batch_not_1",
                    False,
                    False,
                )
            bridge = inputs.first_chunk_dequant
            route = Int8KVRoute.CONTINUATION if bridge else Int8KVRoute.RAGGED
            direct = inputs.direct_paged and not bridge
            return SM75AttentionPlanner._finish_int8kv(inputs, route, direct, bridge)
        return SM75AttentionPlanner._plan_int8kv_continuation(inputs)

    @staticmethod
    def _int8kv_eligibility(inputs: Int8KVRouteInput) -> str | None:
        checks = (
            (not inputs.enabled or not inputs.per_token_head, "disabled"),
            (inputs.kv_cache_dtype != "int8_per_token_head", "not_int8_per_token_head"),
            (not inputs.decoder_attention, "not_decoder_attention"),
            (inputs.has_alibi, "alibi"),
            (inputs.has_sinks, "attention_sinks"),
            (inputs.has_sliding_window, "sliding_window"),
            (inputs.has_logits_soft_cap, "logits_soft_cap"),
            (not inputs.qkv_fp16, "non_fp16_qkv"),
            (not inputs.has_computed_tokens, "missing_num_computed_tokens"),
        )
        return next((reason for failed, reason in checks if failed), None)

    @staticmethod
    def _plan_int8kv_continuation(inputs: Int8KVRouteInput) -> Int8KVRoutePlan:
        if not inputs.continuation_dequant:
            reason = "prefix_or_cached_kv"
        elif inputs.request_count != 1:
            reason = "continuation_batch_not_1"
        elif inputs.query_len < inputs.continuation_min_query:
            reason = "continuation_q_too_small"
        elif not inputs.has_sequence_len:
            reason = "continuation_missing_seq_lens_cpu"
        elif (
            inputs.sequence_len > inputs.continuation_max_tokens
            and not inputs.cascade_dequant
        ):
            reason = "continuation_too_long"
        else:
            route = (
                Int8KVRoute.CASCADE
                if inputs.sequence_len > inputs.continuation_max_tokens
                else Int8KVRoute.CONTINUATION
            )
            return SM75AttentionPlanner._finish_int8kv(
                inputs, route, inputs.direct_paged, False
            )
        return Int8KVRoutePlan(Int8KVRoute.DISABLED, reason, False, False)

    @staticmethod
    def _finish_int8kv(
        inputs: Int8KVRouteInput,
        route: Int8KVRoute,
        direct: bool,
        force_first_chunk: bool,
    ) -> Int8KVRoutePlan:
        if route is not Int8KVRoute.CASCADE and not inputs.ragged_enabled:
            return Int8KVRoutePlan(
                Int8KVRoute.DISABLED,
                "ragged_prefill_disabled",
                direct,
                force_first_chunk,
            )
        return Int8KVRoutePlan(route, None, direct, force_first_chunk)

    @staticmethod
    def plan_spec_sync(mode: SpecSyncMode, kv_cache_dtype: str) -> SpecSyncPlan:
        match mode:
            case SpecSyncMode.SAFE:
                return SpecSyncPlan(True, "safe_mode")
            case SpecSyncMode.NOSYNC:
                return SpecSyncPlan(False, "nosync_mode")
            case SpecSyncMode.AUTO:
                turboquant = kv_cache_dtype.lower().startswith("turboquant_")
                return SpecSyncPlan(
                    turboquant,
                    "turboquant_auto" if turboquant else "non_turboquant_auto",
                )
            case unreachable:
                assert_never(unreachable)
