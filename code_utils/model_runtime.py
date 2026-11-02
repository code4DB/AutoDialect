#!/usr/bin/env python3
"""Resolve AUTODIALECT-compatible model and agent runtime settings.

1. ModelRuntime contract: expose concrete model/provider/base-url/api-key
   values and non-secret metadata for result records.
2. Config resolution: load AutoDialect config YAML, follow `os.environ/...`
   references, and validate required model/provider fields.
3. Agent feature flags: read config-backed agent switches such as
   `enable_memory` without importing the full AutoDialect runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

from code_utils.experiment_utils import ValidationError

ENV_PREFIX = "os.environ/"


# 1. ModelRuntime contract.
@dataclass(frozen=True)
class ModelRuntime:
    """Concrete model settings resolved from the configured environment."""

    config_file: str
    agent_model_key: str
    model_provider_key: str
    provider_type: str
    model: str
    base_url: str
    api_key: str

    def metadata(self) -> dict[str, str]:
        """Return serializable, non-secret metadata for result records."""
        return {
            "config_file": self.config_file,
            "agent_model_key": self.agent_model_key,
            "model_provider_key": self.model_provider_key,
            "provider_type": self.provider_type,
            "model": self.model,
            "base_url": self.base_url,
        }


def _resolve_env_reference(value: Any, *, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    if not value.startswith(ENV_PREFIX):
        return value

    env_name = value[len(ENV_PREFIX):]
    resolved = os.getenv(env_name, "")
    if not resolved:
        raise ValidationError(
            f"Environment variable `{env_name}` referenced by `{field_name}` is empty."
        )
    return resolved


# 2. Config resolution.
def resolve_model_runtime(config_file: str, *, agent_id: str = "autodialect_agent") -> ModelRuntime:
    """Resolve the selected AUTODIALECT agent model into concrete runtime settings."""
    if not config_file:
        raise ValidationError("A AUTODIALECT config file is required to resolve the model runtime.")
    if not os.path.isfile(config_file):
        raise ValidationError(f"AUTODIALECT config file does not exist: {config_file}")

    with open(config_file, "r", encoding="utf-8") as rf:
        config = yaml.safe_load(rf) or {}

    agents = config.get("agents") or {}
    agent_config = agents.get(agent_id) or {}
    agent_model_key = agent_config.get("model")
    if not agent_model_key:
        raise ValidationError(f"Missing agents.{agent_id}.model in {config_file}")

    models = config.get("models") or {}
    model_config = models.get(agent_model_key)
    if not isinstance(model_config, dict):
        raise ValidationError(f"Missing models.{agent_model_key} in {config_file}")

    provider_key = model_config.get("model_provider")
    if not provider_key:
        raise ValidationError(f"Missing models.{agent_model_key}.model_provider in {config_file}")

    providers = config.get("model_providers") or {}
    provider_config = providers.get(provider_key)
    if not isinstance(provider_config, dict):
        raise ValidationError(f"Missing model_providers.{provider_key} in {config_file}")

    model = _resolve_env_reference(model_config.get("model"), field_name=f"models.{agent_model_key}.model")
    base_url = _resolve_env_reference(
        provider_config.get("base_url"),
        field_name=f"model_providers.{provider_key}.base_url",
    )
    api_key = _resolve_env_reference(
        provider_config.get("api_key"),
        field_name=f"model_providers.{provider_key}.api_key",
    )
    provider_type = str(provider_config.get("provider") or provider_key)

    if not model:
        raise ValidationError(f"Resolved model is empty for {agent_model_key} in {config_file}")
    if not base_url:
        raise ValidationError(f"Resolved base_url is empty for {provider_key} in {config_file}")
    if not api_key:
        raise ValidationError(f"Resolved api_key is empty for {provider_key} in {config_file}")

    return ModelRuntime(
        config_file=config_file,
        agent_model_key=str(agent_model_key),
        model_provider_key=str(provider_key),
        provider_type=provider_type,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


# 3. Agent feature flags.
def resolve_agent_memory_enabled(config_file: str, *, agent_id: str = "autodialect_agent") -> bool:
    """Return whether the selected AUTODIALECT agent has opt-in memory enabled."""
    if not config_file:
        return False
    with open(config_file, "r", encoding="utf-8") as rf:
        config = yaml.safe_load(rf) or {}
    agents = config.get("agents") or {}
    agent_config = agents.get(agent_id) or {}
    value = agent_config.get("enable_memory")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
