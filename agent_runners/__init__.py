"""Runner bridge for the AutoDialect agent."""

from __future__ import annotations

from typing import Any

from .autodialect import ensure_agent_available as _ensure_autodialect_agent_available
from .autodialect import run_autodialect_agent


def ensure_agent_available(agent_id: str, model_runtime: Any | None = None) -> None:
    if agent_id != "autodialect_agent":
        raise RuntimeError(f"Unsupported agent id: {agent_id}")
    _ensure_autodialect_agent_available(agent_id)


def get_agent_cli_version(agent_id: str, model_runtime: Any | None = None) -> dict[str, Any]:
    ensure_agent_available(agent_id, model_runtime)
    return {"autodialect_agent": {"path": "in-process", "version": "standalone"}}


def prepare_agent_runtime(agent_id: str, model_runtime: Any | None = None) -> None:
    ensure_agent_available(agent_id, model_runtime)


def run_agent(
    agent_id: str,
    project_path: str,
    task_prompt: str,
    extra_args: dict[str, Any] | None,
    *,
    is_summary: bool = False,
) -> tuple[bool, str, float, dict[str, Any]]:
    if agent_id != "autodialect_agent":
        raise RuntimeError(f"Unsupported agent id: {agent_id}")
    if extra_args is None:
        raise RuntimeError("AutoDialect requires runner arguments.")
    return run_autodialect_agent(
        task_prompt,
        working_dir=project_path,
        config_file=extra_args["config_file"],
        must_patch=extra_args["must_patch"],
        patch_path=extra_args["patch_path"],
        trajectory_file=extra_args["trajectory_file"],
        console_type=extra_args["console_type"],
        agent_type=extra_args["agent_type"],
        completion_validation_path=extra_args.get("completion_validation_path"),
    )


__all__ = [
    "ensure_agent_available",
    "get_agent_cli_version",
    "prepare_agent_runtime",
    "run_agent",
]
