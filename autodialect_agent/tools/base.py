# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""Base classes for tools and tool calling."""

import asyncio
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TypeAlias, override

ParamSchemaValue: TypeAlias = str | list[str] | bool | dict[str, object]
Property: TypeAlias = dict[str, ParamSchemaValue]


class ToolError(Exception):
    """Base class for tool errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message: str = message


@dataclass
class ToolExecResult:
    """Intermediate result of a tool execution."""

    output: str | None = None
    error: str | None = None
    error_code: int = 0


@dataclass
class ToolResult:
    """Result of a tool execution."""

    call_id: str
    name: str
    success: bool
    result: str | None = None
    error: str | None = None
    id: str | None = None  # OpenAI-specific field


ToolCallArguments = dict[str, str | int | float | dict[str, object] | list[object] | None]


@dataclass
class ToolCall:
    """Represents a parsed tool call."""

    name: str
    call_id: str
    arguments: ToolCallArguments = field(default_factory=dict)
    id: str | None = None

    @override
    def __str__(self) -> str:
        return f"ToolCall(name={self.name}, arguments={self.arguments}, call_id={self.call_id}, id={self.id})"


@dataclass
class ToolParameter:
    """Tool parameter definition."""

    name: str
    type: str | list[str]
    description: str
    enum: list[str] | None = None
    items: dict[str, object] | None = None
    required: bool = True


class Tool(ABC):
    """Base class for all tools."""

    def __init__(self, model_provider: str | None = None):
        self._model_provider = model_provider
        self._project_path: Path | None = None

    def set_project_path(self, project_path: str) -> None:
        """Scope mutating tool operations to the active task worktree."""
        path = Path(project_path)
        if not project_path or not path.is_absolute():
            raise ValueError(f"Project path must be a non-empty absolute path: {project_path!r}")
        self._project_path = path.resolve()

    @property
    def project_path(self) -> Path | None:
        return self._project_path

    def is_path_in_project_or_temp(self, path: Path) -> bool:
        """Return whether a write target belongs to the task worktree or /tmp.

        Resolving both paths also prevents an in-worktree symlink from being
        used to write into the orchestration repository or another case.
        Standalone tool instances remain unrestricted until an agent assigns
        their task project path.
        """
        if self._project_path is None:
            return True
        resolved = path.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        return (
            resolved == self._project_path
            or self._project_path in resolved.parents
            or resolved == temp_root
            or temp_root in resolved.parents
        )

    def is_blocked_repo_probe_file(self, path: Path) -> bool:
        """Return whether a mutating write targets a repo-level probe file.

        AutoDialect agents may use /tmp for probes.  The case worktree should
        only contain production SQLGlot edits, so top-level files such as
        test_alias.py, debug.py, repro_*.py, or check_*.py are blocked before
        they can pollute result diffs or later memory extraction.
        """
        if self._project_path is None:
            return False
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._project_path)
        except ValueError:
            temp_root = Path(tempfile.gettempdir()).resolve()
            if resolved == temp_root or temp_root in resolved.parents:
                return False
            return False
        if len(relative.parts) != 1 or resolved.suffix != ".py":
            return False
        name = resolved.name.lower()
        return bool(
            re.match(r"^(test|debug|repro|check)(?:_|\b|\.)", name)
            or name in {"test.py", "debug.py", "repro.py", "check.py"}
        )

    @cached_property
    def model_provider(self) -> str | None:
        return self.get_model_provider()

    @cached_property
    def name(self) -> str:
        return self.get_name()

    @cached_property
    def description(self) -> str:
        return self.get_description()

    @cached_property
    def parameters(self) -> list[ToolParameter]:
        return self.get_parameters()

    def get_model_provider(self) -> str | None:
        """Get the model provider."""
        return self._model_provider

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get the tool description."""
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Get the tool parameters."""
        pass

    @abstractmethod
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the tool with given parameters."""
        pass

    def get_input_schema(self) -> dict[str, object]:
        """Get the input schema for the tool."""
        schema: dict[str, object] = {
            "type": "object",
        }

        properties: dict[str, Property] = {}
        required: list[str] = []

        for param in self.parameters:
            param_schema: Property = {
                "type": param.type,
                "description": param.description,
            }

            # For OpenAI strict mode, all params must be in 'required'.
            # Optional params are made "nullable" to be compliant.
            if self.model_provider == "openai":
                required.append(param.name)
                if not param.required:
                    current_type = param_schema["type"]
                    if isinstance(current_type, str):
                        param_schema["type"] = [current_type, "null"]
                    elif isinstance(current_type, list) and "null" not in current_type:
                        param_schema["type"] = list(current_type) + ["null"]
            elif param.required:
                required.append(param.name)

            if param.enum:
                param_schema["enum"] = param.enum

            if param.items:
                param_schema["items"] = param.items

            # For OpenAI, nested objects also need additionalProperties: false
            if self.model_provider == "openai" and param.type == "object":
                param_schema["additionalProperties"] = False

            properties[param.name] = param_schema

        schema["properties"] = properties
        if len(required) > 0:
            schema["required"] = required

        # For OpenAI, the top-level schema needs additionalProperties: false
        if self.model_provider == "openai":
            schema["additionalProperties"] = False

        return schema

    async def close(self):
        """Ensure proper tool resource deallocation before task completion."""
        return None  # Using "pass" will trigger a Ruff check error: B027


class ToolExecutor:
    """Tool executor that manages tool execution."""

    def __init__(self, tools: list[Tool]):
        self._tools = tools
        self._tool_map: dict[str, Tool] | None = None

    async def close_tools(self):
        """Ensure all tool resources are properly released."""
        tasks = [tool.close() for tool in self._tools if hasattr(tool, "close")]
        res = await asyncio.gather(*tasks)
        return res

    def _normalize_name(self, name: str) -> str:
        """Normalize tool name by making it lowercase and removing underscores."""
        return name.lower().replace("_", "")

    @property
    def tools(self) -> dict[str, Tool]:
        if self._tool_map is None:
            self._tool_map = {self._normalize_name(tool.name): tool for tool in self._tools}
        return self._tool_map

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        normalized_name = self._normalize_name(tool_call.name)
        if normalized_name not in self.tools:
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Tool '{tool_call.name}' not found. Available tools: {[tool.name for tool in self._tools]}",
                call_id=tool_call.call_id,
                id=tool_call.id,
            )

        tool = self.tools[normalized_name]

        try:
            tool_exec_result = await tool.execute(tool_call.arguments)
            return ToolResult(
                name=tool_call.name,
                success=tool_exec_result.error_code == 0,
                result=tool_exec_result.output,
                error=tool_exec_result.error,
                call_id=tool_call.call_id,
                id=tool_call.id,
            )
        except Exception as e:
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Error executing tool '{tool_call.name}': {str(e)}",
                call_id=tool_call.call_id,
                id=tool_call.id,
            )

    async def sequential_tool_call(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls in sequential"""
        return [await self.execute_tool_call(call) for call in tool_calls]
