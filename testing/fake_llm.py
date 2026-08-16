"""Scripted LLMPort double: deterministic loop testing, no network, no cost."""

from __future__ import annotations

import json
import uuid
from typing import Any

from core.llm import LLMPort, LLMResponse, ToolCall, Usage


class ScriptExhaustedError(AssertionError):
    """The loop asked for more responses than the script provides."""


class FakeLLM(LLMPort):
    """Returns a scripted sequence of LLMResponses and records every call.

    ``repeat_last=True`` keeps returning the final scripted response forever
    — the way to simulate a model that never stops calling tools (for
    iteration-ceiling tests).
    """

    def __init__(self, script: list[LLMResponse], repeat_last: bool = False) -> None:
        self._script = list(script)
        self._index = 0
        self._repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tool_schemas": list(tool_schemas)}
        )
        if self._index >= len(self._script):
            if self._repeat_last and self._script:
                return self._script[-1]
            raise ScriptExhaustedError(
                f"FakeLLM script exhausted after {len(self._script)} responses"
            )
        response = self._script[self._index]
        self._index += 1
        return response

    def count_tokens(self, messages: list[dict], tool_schemas: list[dict]) -> int:
        return max(1, len(json.dumps(messages) + json.dumps(tool_schemas)) // 4)


# Script-building helpers ---------------------------------------------------

def call(name: str, params: dict[str, Any], id: str | None = None) -> ToolCall:
    return ToolCall(id=id or f"tc_{uuid.uuid4().hex[:8]}", name=name, params=params)


def text_response(text: str, input_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=(),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def tool_response(
    *calls: ToolCall,
    text: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tuple(calls),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="tool_use",
    )
