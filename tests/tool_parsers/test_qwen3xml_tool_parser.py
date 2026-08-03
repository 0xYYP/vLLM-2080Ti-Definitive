import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


def _bash_tool() -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    )


def _parse_in_chunks(model_output: str, chunk_size: int):
    parser = StreamingXMLToolCallParser()
    parser.set_tools([_bash_tool()])
    deltas = []
    for start in range(0, len(model_output), chunk_size):
        delta = parser.parse_single_streaming_chunks(
            model_output[start : start + chunk_size]
        )
        if delta.tool_calls:
            deltas.extend(delta.tool_calls)
    return deltas


def test_qwen3_xml_keeps_markup_like_bash_text_inside_parameter():
    command = "grep -E '<(foo|bar)>' file && echo 'a & b'"
    output = (
        "<tool_call><function=Bash><parameter=command>"
        f"{command}</parameter></function></tool_call>"
    )

    deltas = _parse_in_chunks(output, len(output))
    arguments = "".join(
        delta.function.arguments
        for delta in deltas
        if delta.function and delta.function.arguments
    )

    assert json.loads(arguments) == {"command": command}


def test_qwen3_xml_keeps_markup_like_bash_text_across_stream_chunks():
    command = "python3 -c 'print(1 < 2)' && cat <<'EOF'\nhello\nEOF"
    output = (
        "<tool_call><function=Bash><parameter=command>"
        f"{command}</parameter></function></tool_call>"
    )

    deltas = _parse_in_chunks(output, 3)
    arguments = "".join(
        delta.function.arguments
        for delta in deltas
        if delta.function and delta.function.arguments
    )

    assert json.loads(arguments) == {"command": command}
