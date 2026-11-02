# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""Tools module for AutoDialect Agent."""

from autodialect_agent.tools.base import Tool, ToolCall, ToolExecutor, ToolResult
from autodialect_agent.tools.bash_tool import BashTool
from autodialect_agent.tools.edit_tool import TextEditorTool
from autodialect_agent.tools.memory_tool import MemoryTool
from autodialect_agent.tools.sequential_thinking_tool import SequentialThinkingTool
from autodialect_agent.tools.task_done_tool import TaskDoneTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolCall",
    "ToolExecutor",
    "BashTool",
    "TextEditorTool",
    "MemoryTool",
    "SequentialThinkingTool",
    "TaskDoneTool",
]

tools_registry: dict[str, type[Tool]] = {
    "bash": BashTool,
    "str_replace_based_edit_tool": TextEditorTool,
    "sequentialthinking": SequentialThinkingTool,
    "task_done": TaskDoneTool,
    "memory": MemoryTool,
}
