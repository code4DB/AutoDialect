# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""LLM client wrapper for the OpenAI-compatible runtime."""

from enum import Enum

from autodialect_agent.tools.base import Tool
from autodialect_agent.utils.config import ModelConfig
from autodialect_agent.utils.llm_clients.base_client import BaseLLMClient
from autodialect_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from autodialect_agent.utils.trajectory_recorder import TrajectoryRecorder


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI_COMPATIBLE = "openai_compatible"


class LLMClient:
    """Main LLM client for the configured provider."""

    def __init__(self, model_config: ModelConfig):
        self.provider: LLMProvider = LLMProvider(model_config.model_provider.provider)
        self.model_config: ModelConfig = model_config

        from .compatible_client import CompatibleClient

        self.client: BaseLLMClient = CompatibleClient(model_config)

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """Set the trajectory recorder for the underlying client."""
        self.client.set_trajectory_recorder(recorder)

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.client.set_chat_history(messages)

    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to the LLM."""
        return self.client.chat(messages, model_config, tools, reuse_history)
