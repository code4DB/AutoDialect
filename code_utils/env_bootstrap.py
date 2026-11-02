#!/usr/bin/env python3
"""Load project-local env files without overriding explicit shell variables."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local")


def _parse_env_assignment(line: str) -> tuple[str | None, str | None]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        return None, None

    key, raw_value = line.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        return None, None

    try:
        tokens = shlex.split(raw_value, comments=True, posix=True)
    except ValueError:
        tokens = [raw_value]

    value = tokens[0] if tokens else ""
    return key, value


def load_project_env(env_files: tuple[Path, ...] = DEFAULT_ENV_FILES) -> list[Path]:
    """Load .env files while keeping pre-existing process env vars authoritative."""

    loaded_files: list[Path] = []
    original_env_keys = set(os.environ)

    for env_path in env_files:
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, value = _parse_env_assignment(line)
            if key is None or key in original_env_keys:
                continue
            os.environ[key] = value or ""

        loaded_files.append(env_path)

    return loaded_files
