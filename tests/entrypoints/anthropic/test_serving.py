import asyncio
import json

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall, UsageInfo


def _tool_response(*, finish_reason: str = "stop") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                finish_reason=finish_reason,
                message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCall(
                                name="shell",
                                arguments='{"command":"true"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )


def test_inline_system_message_is_promoted_before_conversation_turns():
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "test-model",
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "follow the repository rules"},
                {"role": "assistant", "content": "ack"},
            ],
        }
    )

    converted = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assert converted.messages == [
        {"role": "system", "content": "follow the repository rules"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ack"},
    ]


def test_inline_system_message_is_merged_with_top_level_system():
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "test-model",
            "max_tokens": 32,
            "system": "base instructions",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "follow the repository rules"},
                {"role": "assistant", "content": "ack"},
            ],
        }
    )

    converted = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assert converted.messages == [
        {
            "role": "system",
            "content": "base instructions\nfollow the repository rules",
        },
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ack"},
    ]


def _tool_definition(name: str = "shell"):
    return {
        "name": name,
        "description": f"run the {name} tool",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }


def test_anthropic_requests_allow_parallel_tool_calls_by_default():
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "inspect the project"}],
            "tools": [_tool_definition()],
        }
    )

    converted = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assert converted.parallel_tool_calls is True


def test_anthropic_disable_parallel_tool_use_is_forwarded():
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "inspect the project"}],
            "tools": [_tool_definition()],
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": True,
            },
        }
    )

    converted = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assert converted.parallel_tool_calls is False


def test_adjacent_assistant_tool_turns_are_folded_into_one_openai_message():
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "test-model",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "shell",
                            "input": {"command": "pwd"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_2",
                            "name": "shell",
                            "input": {"command": "git status"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": " /workspace",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_2",
                            "content": "clean",
                        },
                    ],
                },
            ],
        }
    )

    converted = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assistant_messages = [
        message for message in converted.messages if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert [call["id"] for call in assistant_messages[0]["tool_calls"]] == [
        "call_1",
        "call_2",
    ]
    tool_messages = [
        message for message in converted.messages if message["role"] == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call_1",
        "call_2",
    ]


def test_full_tool_response_uses_anthropic_tool_use_stop_reason():
    response = AnthropicServingMessages.messages_full_converter(
        None, _tool_response(finish_reason="stop")
    )

    assert response.stop_reason == "tool_use"
    assert response.content[0].type == "tool_use"


def test_full_response_preserves_parallel_tool_calls():
    response = _tool_response(finish_reason="stop")
    response.choices[0].message.tool_calls.append(
        ToolCall(
            id="call_2",
            function=FunctionCall(
                name="read",
                arguments='{"path":"README.md"}',
            ),
        )
    )

    converted = AnthropicServingMessages.messages_full_converter(None, response)

    assert converted.stop_reason == "tool_use"
    assert [block.id for block in converted.content] == ["call_1", "call_2"]
    assert [block.name for block in converted.content] == ["shell", "read"]


def test_stream_tool_response_uses_anthropic_tool_use_stop_reason():
    chunks = [
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "shell", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        ),
    ]

    async def source():
        for chunk in chunks:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    service = object.__new__(AnthropicServingMessages)
    service.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    events = asyncio.run(collect_events(service.message_stream_converter(source())))

    assert '"type":"tool_use"' in "".join(events)
    assert '"stop_reason":"tool_use"' in "".join(events)
    assert sum('"type":"message_delta"' in event for event in events) == 1
    assert sum('"type":"message_stop"' in event for event in events) == 1


def test_stream_preserves_interleaved_parallel_tool_calls():
    chunks = [
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":"pw',
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":"READ',
                                },
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {"arguments": 'ME.md"}'},
                            },
                            {
                                "index": 0,
                                "function": {"arguments": 'd"}'},
                            },
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        ),
    ]

    async def source():
        for chunk in chunks:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    service = object.__new__(AnthropicServingMessages)
    service.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    raw_events = asyncio.run(collect_events(service.message_stream_converter(source())))
    events = [
        json.loads(event.split("data: ", 1)[1])
        for event in raw_events
        if "data: " in event
    ]

    starts = [event for event in events if event["type"] == "content_block_start"]
    deltas = [
        event
        for event in events
        if event["type"] == "content_block_delta"
        and event["delta"]["type"] == "input_json_delta"
    ]
    assert [event["index"] for event in starts] == [0, 1]
    assert [event["content_block"]["id"] for event in starts] == [
        "call_1",
        "call_2",
    ]
    assert [json.loads(event["delta"]["partial_json"]) for event in deltas] == [
        {"command": "pwd"},
        {"path": "README.md"},
    ]
    message_deltas = [event for event in events if event["type"] == "message_delta"]
    assert message_deltas[-1]["delta"]["stop_reason"] == "tool_use"


def test_stream_done_without_usage_chunk_flushes_tools_and_stop_reason():
    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        model="test-model",
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ]
                },
                "finish_reason": "stop",
            }
        ],
    )

    async def source():
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    service = object.__new__(AnthropicServingMessages)
    service.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    raw_events = asyncio.run(collect_events(service.message_stream_converter(source())))
    events = [
        json.loads(event.split("data: ", 1)[1])
        for event in raw_events
        if "data: " in event
    ]

    event_types = [event["type"] for event in events]
    assert event_types[-2:] == ["message_delta", "message_stop"]
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_stream_preserves_tool_then_text_block_order():
    chunks = [
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {"content": "after tool"},
                    "finish_reason": "stop",
                }
            ],
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        ),
    ]

    async def source():
        for chunk in chunks:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    service = object.__new__(AnthropicServingMessages)
    service.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    raw_events = asyncio.run(collect_events(service.message_stream_converter(source())))
    events = [
        json.loads(event.split("data: ", 1)[1])
        for event in raw_events
        if "data: " in event
    ]

    starts = [event for event in events if event["type"] == "content_block_start"]
    assert [event["content_block"]["type"] for event in starts] == [
        "tool_use",
        "text",
    ]


def test_stream_reports_incomplete_tool_call_instead_of_dropping_it():
    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        model="test-model",
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"command":"pwd"}'},
                        }
                    ]
                },
                "finish_reason": "stop",
            }
        ],
    )

    async def source():
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    service = object.__new__(AnthropicServingMessages)
    service.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    raw_events = asyncio.run(collect_events(service.message_stream_converter(source())))
    events = [
        json.loads(event.split("data: ", 1)[1])
        for event in raw_events
        if "data: " in event
    ]

    assert events[-1]["type"] == "error"
    assert "missing id or name" in events[-1]["error"]["message"]


async def collect_events(generator):
    return [event async for event in generator]
