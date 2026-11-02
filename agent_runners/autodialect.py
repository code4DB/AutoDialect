"""AutoDialect agent bridge utilities used by the main workflow.

1. Subprocess bridge: invoke autodialect_agent.cli.run() with the configured
   AutoDialect Python executable while keeping rule_main isolated.
2. Availability probe: fail early when the AutoDialect runtime cannot be imported or
   started from the selected Python environment.
3. Case contract: pass task, patch, trajectory, and completion-validation
   fields into AutoDialect and normalize its result tuple for experiment records.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess

from code_utils.constants import AUTODIALECT_AGENT, AUTODIALECT_PYTHON_BIN

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTODIALECT_RESULT_MARKER = "__AUTODIALECT_RESULT__"
AUTODIALECT_PROBE_CACHE = None

# 1. Subprocess bridge.
AUTODIALECT_SUBPROCESS_SCRIPT = r"""
import json
import os
import sys
import traceback

root_dir = sys.argv[1]
sys.path[:] = [root_dir, *[path for path in sys.path if path != root_dir]]

def emit(payload):
    sys.stdout.write("__AUTODIALECT_RESULT__" + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def exit_now(code):
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)

try:
    from autodialect_agent.cli import run as autodialect_run

    payload = json.loads(sys.stdin.read())
    run_result = autodialect_run(
        task=payload["task"],
        file_path=None,
        working_dir=payload["working_dir"],
        config_file=payload["config_file"],
        must_patch=payload["must_patch"],
        patch_path=payload["patch_path"],
        trajectory_file=payload["trajectory_file"],
        console_type=payload["console_type"],
        agent_type=payload["agent_type"],
        completion_validation_path=payload.get("completion_validation_path"),
    )
    if len(run_result) == 4:
        is_success, response, execution_time, memory_audit = run_result
    else:
        is_success, response, execution_time = run_result
        memory_audit = {}
    emit(
        {
            "ok": True,
            "is_success": is_success,
            "response": response,
            "execution_time": execution_time,
            "memory_audit": memory_audit,
        }
    )
    exit_now(0)
except Exception as exc:
    emit(
        {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    )
    exit_now(1)
"""


def _env_positive_float(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _run_autodialect_subprocess(payload, timeout=None):
    env_new = os.environ.copy()

    proc = subprocess.Popen(
        [AUTODIALECT_PYTHON_BIN, "-c", AUTODIALECT_SUBPROCESS_SCRIPT, ROOT_DIR],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=ROOT_DIR,
        env=env_new,
        start_new_session=timeout is not None,
    )
    try:
        stdout, stderr = proc.communicate(json.dumps(payload, ensure_ascii=False), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return False, f"Error occurs when executing: {str(exc)}", 0.0, {}

    result = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(AUTODIALECT_RESULT_MARKER):
            try:
                candidate = json.loads(line[len(AUTODIALECT_RESULT_MARKER):])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(candidate, dict):
                result = candidate
                break

    if result is None:
        details = (stderr or stdout or "").strip()
        if not details:
            details = f"Fail to execute, errorcode: {proc.returncode}"
        return False, details, 0.0, {}

    if not result.get("ok"):
        details = result.get("error") or "Unknown autodialect subprocess error"
        traceback_text = result.get("traceback")
        if traceback_text:
            details = f"{details}\n{traceback_text}"
        return False, details, 0.0, {}

    if proc.returncode != 0 and not result.get("is_success"):
        details = result.get("response") or stderr or f"Fail to execute, errorcode: {proc.returncode}"
        return False, details, float(result.get("execution_time") or 0.0), result.get("memory_audit") or {}

    return (
        bool(result.get("is_success")),
        result.get("response") or "",
        float(result.get("execution_time") or 0.0),
        result.get("memory_audit") or {},
    )


# 2. Availability probe.
def _probe_autodialect_subprocess():
    global AUTODIALECT_PROBE_CACHE
    if AUTODIALECT_PROBE_CACHE is not None:
        return AUTODIALECT_PROBE_CACHE

    try:
        completed = subprocess.run(
            [
                AUTODIALECT_PYTHON_BIN,
                "-c",
                (
                    "import os, sys; "
                    "root=sys.argv[1]; "
                    "sys.path[:]=[root]+[p for p in sys.path if p != root]; "
                    "import autodialect_agent.cli"
                ),
                ROOT_DIR,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=ROOT_DIR,
            check=False,
        )
    except FileNotFoundError:
        AUTODIALECT_PROBE_CACHE = f"AutoDialect Python executable does not exist: {AUTODIALECT_PYTHON_BIN}"
        return AUTODIALECT_PROBE_CACHE
    except Exception as exc:
        AUTODIALECT_PROBE_CACHE = str(exc)
        return AUTODIALECT_PROBE_CACHE

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        AUTODIALECT_PROBE_CACHE = (
            f"AutoDialect agent is unavailable from Python `{AUTODIALECT_PYTHON_BIN}`"
            + (f": {details}" if details else "")
        )
        return AUTODIALECT_PROBE_CACHE

    AUTODIALECT_PROBE_CACHE = ""
    return AUTODIALECT_PROBE_CACHE


def ensure_agent_available(agent_id):
    if agent_id != AUTODIALECT_AGENT:
        return
    probe_error = _probe_autodialect_subprocess()
    if probe_error:
        raise RuntimeError(
            "AutoDialect agent is unavailable through the configured Python. "
            f"{probe_error}"
        )


# 3. Case contract.
def run_autodialect_agent(
    task,
    working_dir,
    config_file,
    must_patch,
    patch_path,
    trajectory_file,
    console_type,
    agent_type,
    completion_validation_path=None,
):
    ensure_agent_available(agent_type)
    payload = {
        "task": task,
        "working_dir": working_dir,
        "config_file": config_file,
        "must_patch": must_patch,
        "patch_path": patch_path,
        "trajectory_file": trajectory_file,
        "console_type": console_type,
        "agent_type": agent_type,
        "completion_validation_path": completion_validation_path,
    }
    timeout = _env_positive_float("AUTODIALECT_STAGE_TIMEOUT_S")

    return _run_autodialect_subprocess(payload, timeout=timeout)
