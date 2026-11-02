# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""Programmatic AutoDialect agent entrypoint used by the generation runner."""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

from rich.console import Console

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:] = [str(ROOT_DIR), *[path for path in sys.path if path != str(ROOT_DIR)]]

from code_utils.env_bootstrap import load_project_env

from autodialect_agent.agent import Agent
from autodialect_agent.utils.cli import SimpleCLIConsole
from autodialect_agent.utils.config import Config

_ = load_project_env()
console = Console()


def resolve_config_file(config_file: str) -> str:
    """Return an existing runtime config path."""
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"AutoDialect config file does not exist: {config_file}")
    return str(path)


def _failure(message: str) -> tuple[bool, str, float]:
    console.print(f"[red]{message}[/red]")
    return False, message, 0.0


def run(
    task: str | None,
    file_path: str | None,
    patch_path: str,
    provider: str | None = None,
    model: str | None = None,
    model_base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int | None = None,
    working_dir: str | None = None,
    must_patch: bool = False,
    config_file: str = "autodialect_config.yaml",
    trajectory_file: str | None = None,
    console_type: str | None = "simple",
    agent_type: str | None = "autodialect_agent",
    completion_validation_path: str | None = None,
):
    """Run one AutoDialect agent task and return execution metadata."""
    config_file = resolve_config_file(config_file)

    if file_path:
        if task:
            return _failure("Cannot use both a task string and --file.")
        try:
            task = Path(file_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return _failure(f"Task file does not exist: {file_path}")
    elif not task:
        return _failure("A task string or --file is required.")

    config = Config.create(config_file=config_file).resolve_config_values(
        provider=provider,
        model=model,
        model_base_url=model_base_url,
        api_key=api_key,
        max_steps=max_steps,
    )

    if not agent_type:
        return _failure("agent_type is required.")
    if console_type not in (None, "simple"):
        return _failure(f"Unsupported console type: {console_type}")

    cli_console = SimpleCLIConsole() if console_type == "simple" else None
    agent = Agent(agent_type, config, trajectory_file, cli_console)

    if working_dir:
        try:
            os.chdir(working_dir)
        except OSError as exc:
            return _failure(f"Could not enter working directory {working_dir}: {exc}")
    else:
        working_dir = os.getcwd()

    if not Path(working_dir).is_absolute():
        return _failure(f"Working directory must be absolute: {working_dir}")

    task_args = {
        "project_path": working_dir,
        "issue": task,
        "must_patch": "true" if must_patch else "false",
        "patch_path": patch_path,
    }
    if completion_validation_path:
        task_args["completion_validation_path"] = completion_validation_path

    try:
        execution = asyncio.run(agent.run(task, task_args))
        memory_audit = getattr(agent.agent, "memory_audit", {})
        return (
            execution.success,
            execution.final_result,
            execution.execution_time,
            memory_audit,
        )
    except KeyboardInterrupt:
        return False, "Task execution interrupted by user.", 0.0
    except Exception as exc:
        console.print(traceback.format_exc())
        return False, f"Unexpected AutoDialect runtime error: {exc}", 0.0
