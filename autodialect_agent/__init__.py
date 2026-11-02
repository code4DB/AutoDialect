# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""AutoDialect Agent - LLM-based agent for general purpose software engineering tasks."""

__version__ = "0.1.0"

from autodialect_agent.agent.base_agent import BaseAgent
from autodialect_agent.agent.autodialect_agent import AutoDialectAgent
from autodialect_agent.tools.base import Tool, ToolExecutor
from autodialect_agent.utils.llm_clients.llm_client import LLMClient

__all__ = ["BaseAgent", "AutoDialectAgent", "LLMClient", "Tool", "ToolExecutor"]
