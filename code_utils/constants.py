"""Defaults for the AutoDialect runtime."""

from __future__ import annotations

import os
import sys

from code_utils.env_bootstrap import load_project_env

load_project_env()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CODE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "worktrees")

AUTODIALECT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "autodialect_config.yaml")
AUTODIALECT_AGENT = "autodialect_agent"
AUTODIALECT_PYTHON_BIN = os.getenv("AUTODIALECT_PYTHON_BIN", "").strip() or sys.executable

LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8284")
DEFAULT_AGENT_ID = os.getenv("LETTA_AGENT_ID", "")
DEFAULT_AGENT_NAME = os.getenv("LETTA_AGENT_NAME", "autodialect_memory_agent")

AUTODIALECT_MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT = int(
    os.getenv("AUTODIALECT_MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT", "2")
)
AUTODIALECT_GROUP_MERGE_ENABLED = os.getenv(
    "AUTODIALECT_GROUP_MERGE_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
AUTODIALECT_MAX_HARD_UNITS_PER_GROUP_BATCH = int(
    os.getenv("AUTODIALECT_MAX_HARD_UNITS_PER_GROUP_BATCH", "12")
)
AUTODIALECT_MAX_SYNTHETIC_VARIANT_SQL_CHARS = int(
    os.getenv("AUTODIALECT_MAX_SYNTHETIC_VARIANT_SQL_CHARS", "800")
)
AUTODIALECT_STAGE_HANDOFF_ENABLED = os.getenv(
    "AUTODIALECT_STAGE_HANDOFF_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
AUTODIALECT_COMPLETION_VALIDATION_ENABLED = os.getenv(
    "AUTODIALECT_COMPLETION_VALIDATION_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
AUTODIALECT_SPLIT_HELPER_TIMEOUT_S = int(
    os.getenv("AUTODIALECT_SPLIT_HELPER_TIMEOUT_S", "120")
)
AUTODIALECT_SQLGLOT_WORKER_TIMEOUT_S = float(
    os.getenv("AUTODIALECT_SQLGLOT_WORKER_TIMEOUT_S", "30")
)
AUTODIALECT_SQLGLOT_WORKER_INIT_TIMEOUT_S = float(
    os.getenv("AUTODIALECT_SQLGLOT_WORKER_INIT_TIMEOUT_S", "30")
)

SQLGLOT_WORKTREE_PREFIX = "sqlglot_worktree"
BACKUP_AUTODIALECT_TRAJECTORIES_DIR = os.path.join(RESULTS_DIR, "trajectories")
