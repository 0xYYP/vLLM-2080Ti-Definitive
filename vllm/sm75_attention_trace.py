# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 attention route tracing with process-local event bounds."""

from __future__ import annotations

from threading import Lock
from typing import NamedTuple

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

TraceFieldValue = str | int | float | bool | None
TraceFields = tuple[tuple[str, TraceFieldValue], ...]


class SM75AttentionTraceEvent(NamedTuple):
    event: str
    fields: TraceFields


_TRACE_LOCK = Lock()
_TRACE_SEEN: set[SM75AttentionTraceEvent] = set()


def sm75_attention_trace_enabled() -> bool:
    return bool(envs.VLLM_SM75_ATTENTION_TRACE)


def _render_trace_value(value: TraceFieldValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def sm75_attention_trace(event: str, **fields: TraceFieldValue) -> None:
    if not sm75_attention_trace_enabled():
        return

    trace_event = SM75AttentionTraceEvent(event, tuple(sorted(fields.items())))
    with _TRACE_LOCK:
        if trace_event in _TRACE_SEEN:
            return
        if len(_TRACE_SEEN) >= envs.VLLM_SM75_ATTENTION_TRACE_MAX_EVENTS:
            return
        _TRACE_SEEN.add(trace_event)

    rendered_fields = " ".join(
        f"{key}={_render_trace_value(value)}" for key, value in trace_event.fields
    )
    message = f"SM75 attention trace event={event}"
    if rendered_fields:
        message = f"{message} {rendered_fields}"
    logger.info(message, extra={"sm75_attention_trace": trace_event})


def _reset_sm75_attention_trace_for_tests() -> None:
    with _TRACE_LOCK:
        _TRACE_SEEN.clear()
