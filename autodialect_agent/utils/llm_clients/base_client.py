# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT


from abc import ABC, abstractmethod

from autodialect_agent.tools.base import Tool
from autodialect_agent.utils.config import ModelConfig
from autodialect_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from autodialect_agent.utils.trajectory_recorder import TrajectoryRecorder


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    def __init__(self, model_config: ModelConfig):
        self.api_key: str = model_config.model_provider.api_key
        self.base_url: str | None = model_config.model_provider.base_url
        self.trajectory_recorder: TrajectoryRecorder | None = None  # TrajectoryRecorder instance

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """Set the trajectory recorder for this client."""
        self.trajectory_recorder = recorder

    @abstractmethod
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to the LLM."""
        pass
