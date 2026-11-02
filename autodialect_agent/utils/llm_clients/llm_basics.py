# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT


import json
from dataclasses import dataclass
from typing import Any

from autodialect_agent.tools.base import ToolCall, ToolResult


@dataclass
class LLMMessage:
    """Standard message format."""

    role: str
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass
class LLMUsage:
    """LLM usage format."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def __str__(self) -> str:
        return f"LLMUsage(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, cache_creation_input_tokens={self.cache_creation_input_tokens}, cache_read_input_tokens={self.cache_read_input_tokens}, reasoning_tokens={self.reasoning_tokens})"


@dataclass
class LLMResponse:
    """Standard LLM response format."""

    content: str
    usage: LLMUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None


def parse_tool_call_arguments(raw_arguments: str | None) -> dict[str, Any]:
    """Parse model-emitted tool arguments without crashing the agent loop."""
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return {
            "_raw_arguments": raw_arguments,
            "_json_parse_error": f"{type(exc).__name__}: {exc}",
        }
    if isinstance(parsed, dict):
        return parsed
    return {
        "_raw_arguments": raw_arguments,
        "_json_parse_error": f"Tool arguments must be a JSON object, got {type(parsed).__name__}.",
    }
