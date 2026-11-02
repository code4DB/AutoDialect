# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""Generic OpenAI-compatible client used by AutoDialect."""

import openai

from autodialect_agent.utils.config import ModelConfig
from autodialect_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class CompatibleProvider(ProviderConfig):
    """Configuration for an OpenAI-compatible chat-completions endpoint."""

    def create_client(self, api_key: str, base_url: str | None) -> openai.OpenAI:
        """Create the OpenAI SDK client with the configured base URL."""
        return openai.OpenAI(base_url=base_url, api_key=api_key)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "OpenAI-compatible"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "openai_compatible"

    def get_extra_headers(self) -> dict[str, str]:
        """Return provider headers; none are required by default."""
        return {}

class CompatibleClient(OpenAICompatibleClient):
    """OpenAI-compatible client with AutoDialect tool integration."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config, CompatibleProvider())
