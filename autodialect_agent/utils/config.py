# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

import os
from dataclasses import dataclass, field

import yaml


class ConfigError(Exception):
    pass


ENV_REF_PREFIX = "os.environ/"


@dataclass
class ModelProvider:
    """
    Model provider configuration. The runtime accepts an
    OpenAI-compatible endpoint; the base URL is optional.
    """

    api_key: str
    provider: str
    base_url: str | None = None


@dataclass
class ModelConfig:
    """
    Model configuration.
    """

    model: str
    model_provider: ModelProvider
    max_tokens: int
    temperature: float
    top_p: float
    parallel_tool_calls: bool
    max_retries: int

    def resolve_config_values(
        self,
        *,
        model_providers: dict[str, ModelProvider] | None = None,
        provider: str | None = None,
        model: str | None = None,
        model_base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        When some config values are provided through CLI or environment variables,
        they will override the values in the config file.
        """
        self.model = str(resolve_config_value(cli_value=model, config_value=self.model))

        # If the user wants to change the model provider, they should either:
        # * Make sure the provider name is available in the model_providers dict;
        # * If not, base url and api key should be provided to register a new model provider.
        if provider:
            if model_providers and provider in model_providers:
                self.model_provider = model_providers[provider]
            elif api_key is None:
                raise ConfigError("To register a new model provider, an api_key should be provided")
            else:
                self.model_provider = ModelProvider(
                    api_key=api_key,
                    provider=provider,
                    base_url=model_base_url,
                )

        # Map providers to their environment variable names
        env_var_api_key = str(self.model_provider.provider).upper() + "_API_KEY"
        env_var_api_base_url = str(self.model_provider.provider).upper() + "_BASE_URL"

        resolved_api_key = resolve_config_value(
            cli_value=api_key,
            config_value=self.model_provider.api_key,
            env_var=env_var_api_key,
        )

        resolved_api_base_url = resolve_config_value(
            cli_value=model_base_url,
            config_value=self.model_provider.base_url,
            env_var=env_var_api_base_url,
        )

        if resolved_api_key:
            self.model_provider.api_key = str(resolved_api_key)

        if resolved_api_base_url:
            self.model_provider.base_url = str(resolved_api_base_url)


@dataclass
class AgentConfig:
    """
    Base class for agent configurations.
    """

    max_steps: int
    model: ModelConfig
    tools: list[str]


@dataclass
class AutoDialectAgentConfig(AgentConfig):
    """
    AutoDialect agent configuration.
    """

    enable_memory: bool = False
    tools: list[str] = field(
        default_factory=lambda: [
            "bash",
            "str_replace_based_edit_tool",
            "sequentialthinking",
            "task_done",
        ]
    )

    def resolve_config_values(
        self,
        *,
        max_steps: int | None = None,
    ):
        resolved_value = resolve_config_value(cli_value=max_steps, config_value=self.max_steps)
        if resolved_value:
            self.max_steps = int(resolved_value)


@dataclass
class Config:
    """
    Configuration class for agents, models and model providers.
    """

    model_providers: dict[str, ModelProvider] | None = None
    models: dict[str, ModelConfig] | None = None

    autodialect_agent: AutoDialectAgentConfig | None = None

    @classmethod
    def create(
        cls,
        *,
        config_file: str | None = None,
        config_string: str | None = None,
    ) -> "Config":
        if config_file and config_string:
            raise ConfigError("Only one of config_file or config_string should be provided")

        # Parse YAML config from file or string
        try:
            if config_file is not None:
                with open(config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)
            elif config_string is not None:
                yaml_config = yaml.safe_load(config_string)
            else:
                raise ConfigError("No config file or config string provided")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML config: {e}") from e

        config = cls()

        # Parse model providers
        model_providers = yaml_config.get("model_providers", None)
        if model_providers is not None and len(model_providers.keys()) > 0:
            config_model_providers: dict[str, ModelProvider] = {}
            for model_provider_name, model_provider_config in model_providers.items():
                config_model_providers[model_provider_name] = ModelProvider(**model_provider_config)
            config.model_providers = config_model_providers
        else:
            raise ConfigError("No model providers provided")

        # Parse models and populate model_provider fields
        models = yaml_config.get("models", None)
        if models is not None and len(models.keys()) > 0:
            config_models: dict[str, ModelConfig] = {}
            for model_name, model_config in models.items():
                if model_config["model_provider"] not in config_model_providers:
                    raise ConfigError(f"Model provider {model_config['model_provider']} not found")
                config_models[model_name] = ModelConfig(**model_config)
                config_models[model_name].model_provider = config_model_providers[
                    model_config["model_provider"]
                ]
            config.models = config_models
        else:
            raise ConfigError("No models provided")

        # Parse agents
        agents = yaml_config.get("agents", None)
        if agents is not None and len(agents.keys()) > 0:
            for agent_name, agent_config in agents.items():
                agent_model_name = agent_config.get("model", None)
                if agent_model_name is None:
                    raise ConfigError(f"No model provided for {agent_name}")
                try:
                    agent_model = config_models[agent_model_name]
                except KeyError as e:
                    raise ConfigError(f"Model {agent_model_name} not found") from e
                match agent_name:
                    case "autodialect_agent":
                        autodialect_agent_config = AutoDialectAgentConfig(
                            **agent_config,
                        )
                        autodialect_agent_config.model = agent_model
                        config.autodialect_agent = autodialect_agent_config
                    case _:
                        raise ConfigError(f"Unknown agent: {agent_name}")
        else:
            raise ConfigError("No agent configs provided")
        return config

    def resolve_config_values(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        model_base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int | None = None,
    ):
        if self.autodialect_agent:
            self.autodialect_agent.resolve_config_values(
                max_steps=max_steps,
            )
            self.autodialect_agent.model.resolve_config_values(
                model_providers=self.model_providers,
                provider=provider,
                model=model,
                model_base_url=model_base_url,
                api_key=api_key,
            )
        return self

def resolve_config_value(
    *,
    cli_value: int | str | float | None,
    config_value: int | str | float | None,
    env_var: str | None = None,
) -> int | str | float | None:
    """Resolve configuration value with priority: CLI > ENV > Config > Default."""
    if cli_value is not None:
        return cli_value

    if isinstance(config_value, str) and config_value.startswith(ENV_REF_PREFIX):
        referenced_env_var = config_value[len(ENV_REF_PREFIX):]
        referenced_env_value = os.getenv(referenced_env_var)
        if referenced_env_value:
            return referenced_env_value

    if env_var and os.getenv(env_var):
        return os.getenv(env_var)

    if config_value is not None:
        return config_value

    return None
