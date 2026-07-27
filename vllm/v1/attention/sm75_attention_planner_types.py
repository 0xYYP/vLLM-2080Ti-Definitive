# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import NamedTuple


class PlannerInputError(ValueError):
    pass


class TQPrefillStage(str, Enum):
    FIRST_CHUNK = "first_chunk"
    CONTINUATION = "continuation"


class TQPrefixCombineMode(str, Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


class SpecSyncMode(str, Enum):
    AUTO = "auto"
    SAFE = "safe"
    NOSYNC = "nosync"


class TQDecodeRoute(str, Enum):
    NATIVE = "native"
    SDPA = "sdpa"


class Int8KVRoute(str, Enum):
    DISABLED = "disabled"
    RAGGED = "ragged"
    CONTINUATION = "continuation"
    CASCADE = "cascade"


class WorkspacePolicy(str, Enum):
    NONE = "none"
    SHARED = "shared"


class TQCUDAGraphInput(NamedTuple):
    spec_decode_safe: bool
    max_query_len: int
    continuation_threshold: int


class TQCUDAGraphPlan(NamedTuple):
    force_spec_decode: bool
    capture_seq_len: int


class TQPrefillCapabilityInput(NamedTuple):
    requested: bool
    wrapper_available: bool
    is_cuda: bool
    is_sm75: bool
    head_size: int
    sm75_min_head_size: int


class CapabilityPlan(NamedTuple):
    enabled: bool
    reason: str | None


class TQFlashInferInput(NamedTuple):
    available: bool
    is_sm75: bool
    query_len: int
    first_chunk_min_query: int
    continuation_min_query: int


class TQDecodeInput(NamedTuple):
    force_sdpa: bool
    speculative: bool
    head_size: int
    d256_fallback: bool
    d512_fallback: bool
    shared_draft_fallback: bool


class TQDecodePlan(NamedTuple):
    route: TQDecodeRoute
    reason: str | None


class TQContinuationInput(NamedTuple):
    prefix_combine_mode: TQPrefixCombineMode
    prefix_combine_min_tokens: int
    sequence_len: int
    force_sdpa: bool
    kv_cache_dim: int
    cached_len: int
    sliding_window: int
    flashinfer_continuation: bool


class TQContinuationPlan(NamedTuple):
    prefix_combine: bool
    workspace: WorkspacePolicy


class Int8KVRouteInput(NamedTuple):
    enabled: bool
    per_token_head: bool
    kv_cache_dtype: str
    decoder_attention: bool
    has_alibi: bool
    has_sinks: bool
    has_sliding_window: bool
    has_logits_soft_cap: bool
    qkv_fp16: bool
    has_computed_tokens: bool
    has_sequence_len: bool
    request_count: int
    query_len: int
    sequence_len: int
    computed_tokens: int
    first_chunk_dequant: bool
    continuation_dequant: bool
    continuation_min_query: int
    continuation_max_tokens: int
    cascade_dequant: bool
    direct_paged: bool
    ragged_enabled: bool


class Int8KVRoutePlan(NamedTuple):
    route: Int8KVRoute
    reason: str | None
    direct_paged_attempt: bool
    force_first_chunk_dequant: bool


class SpecSyncPlan(NamedTuple):
    enabled: bool
    reason: str
