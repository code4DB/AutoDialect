"""Runtime helpers for the AutoDialect AUTODIALECT pipeline.

This module owns helpers that depend on live pipeline/runtime state:
1. Runtime safety/config and normalization helpers: parse conservative env
   defaults, validate repo-relative paths, and normalize shared runtime ids.
2. Stage memory audit helpers: preserve per-stage retrieval audit data with a
   compact aggregate summary.
3. Handoff context and payload helpers: summarize git/worktree, filesystem,
   validation, group execution, and patch evidence into compact stage handoffs.
4. JSON subprocess and parser-gate helpers: run Python JSON helpers, execute
   parser gates, and render prompt-friendly parser-gate summaries.

General SQL comparison, SQLGlot transpile workers, and git/file-copy utilities
belong in code_utils.common_tool instead.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import time
from typing import Any

from code_utils.common_tool import compact_output_fields, model_facing_diagnostics
from code_utils.constants import AUTODIALECT_STAGE_HANDOFF_ENABLED, AUTODIALECT_PYTHON_BIN
from code_utils.patch_evidence import (
    extract_changed_files,
    extract_patch_evidence,
    infer_patch_tags,
)
from prompt.agent_prompt import STAGE_HANDOFF_INSTRUCTIONS

JSON_SENTINEL = "__AUTODIALECT_JSON__:"
_PARSER_GATE_SCRIPT = r"""
import json
import sys

payload = json.loads(sys.stdin.read() or "{}")
package_root = payload["package_root"]
sql = payload["sql"]
dialect = payload["dialect"]
sys.path.insert(0, package_root)

def emit(obj):
    print("__AUTODIALECT_JSON__:" + json.dumps(obj, ensure_ascii=False))

try:
    import sqlglot

    ast = sqlglot.parse_one(sql, read=dialect)
    nodes = []
    for node in ast.walk():
        name = node.__class__.__name__
        if len(nodes) < 40:
            nodes.append(name)
    emit({
        "ok": True,
        "status": "parse_ok",
        "root": ast.__class__.__name__,
        "node_types": nodes,
        "ast_sql": ast.sql(dialect=dialect),
    })
except Exception as exc:
    emit({
        "ok": False,
        "status": "parse_failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    })
"""


# 1. Runtime safety/config and normalization helpers.
def safe_relative_path(path: str) -> str | None:
    """Return a normalized repo-relative path, or None if it is unsafe."""
    if not isinstance(path, str):
        return None
    normalized = os.path.normpath(path)
    if normalized in ("", ".") or os.path.isabs(normalized):
        return None
    if normalized == ".." or normalized.startswith(f"..{os.sep}"):
        return None
    return normalized


STAGE_HANDOFF_ENABLED = AUTODIALECT_STAGE_HANDOFF_ENABLED


def _normalize_group_ids(value: Any) -> list[str]:
    if value in (None, "", [], ()):
        return []
    if isinstance(value, set):
        raw_values = sorted(value, key=str)
    elif isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = [value]
    group_ids = []
    seen = set()
    for item in raw_values:
        text = str(item or "").strip()
        if text and text not in seen:
            group_ids.append(text)
            seen.add(text)
    return group_ids


# 2. Stage memory audit helpers.
def _truthy_audit_flag(section: dict[str, Any], key: str) -> bool:
    value = section.get(key)
    if isinstance(value, bool):
        return value
    return bool(value)


def _build_memory_audit_summary(stages: list[dict[str, Any]], latest_stage: str) -> dict[str, Any]:
    """Aggregate per-stage retrieval outcomes for the final case audit.

    Malformed stage entries are ignored. Error stages preserve execution order;
    the counters describe stage outcomes rather than individual memory calls.
    """
    summary: dict[str, Any] = {
        "stage_count": len(stages),
        "latest_stage": latest_stage,
        "retrieval_attempted_count": 0,
        "retrieval_success_count": 0,
        "retrieval_accepted_count": 0,
        "retrieval_injected_count": 0,
        "retrieval_partial_failure_count": 0,
        "stages_with_errors": [],
    }
    stages_with_errors: list[str] = []

    for item in stages:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "")
        audit = item.get("audit")
        if not isinstance(audit, dict):
            continue
        retrieval = audit.get("retrieval") if isinstance(audit.get("retrieval"), dict) else {}

        if _truthy_audit_flag(retrieval, "attempted"):
            summary["retrieval_attempted_count"] += 1
        if _truthy_audit_flag(retrieval, "success"):
            summary["retrieval_success_count"] += 1
        if _truthy_audit_flag(retrieval, "accepted"):
            summary["retrieval_accepted_count"] += 1
        if _truthy_audit_flag(retrieval, "injected"):
            summary["retrieval_injected_count"] += 1
        if _truthy_audit_flag(retrieval, "partial_failure"):
            summary["retrieval_partial_failure_count"] += 1

        has_error = (
            bool(audit.get("init_error"))
            or bool(retrieval.get("error"))
            or bool(retrieval.get("errors"))
        )
        if has_error and stage:
            stages_with_errors.append(stage)

    summary["stages_with_errors"] = stages_with_errors
    return summary


def _record_stage_memory_audit(
    memory_audit: dict[str, Any],
    *,
    stage: str,
    attempt_memory_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Append one stage audit and rebuild its compact aggregate summary."""
    stages = (
        list(memory_audit.get("stages", []))
        if isinstance(memory_audit, dict) and isinstance(memory_audit.get("stages"), list)
        else []
    )
    attempt = dict(attempt_memory_audit or {})
    stages.append({"stage": stage, "audit": attempt})
    return {
        "stages": stages,
        "summary": _build_memory_audit_summary(stages, stage),
    }


# 3. Handoff context and payload helpers.
def _compact_worktree_context(
    project_path: str,
    base_ref: str | None = None,
    max_diff_chars: int = 6000,
    max_untracked_files: int = 5,
    max_untracked_chars: int = 4000,
) -> dict[str, Any]:
    """Collect a compact, read-only summary of current worktree edits."""
    context: dict[str, Any] = {
        "changed_files": [],
        "diff": "",
        "diff_truncated": False,
        "untracked_file_snippets": [],
    }
    if not os.path.isdir(project_path):
        context["error"] = f"project path does not exist: {project_path}"
        return context

    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_path,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if status.returncode == 0:
            context["changed_files"] = [
                line[3:].strip()
                for line in status.stdout.splitlines()
                if len(line) >= 4 and line[3:].strip()
            ][:30]
    except Exception as exc:
        context["status_error"] = f"{type(exc).__name__}: {exc}"

    diff_cmd = ["git", "--no-pager", "diff"]
    if base_ref:
        diff_cmd.append(base_ref)
    try:
        diff = subprocess.run(
            diff_cmd,
            cwd=project_path,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if diff.returncode == 0:
            context["diff"] = diff.stdout[:max_diff_chars]
            context["diff_truncated"] = len(diff.stdout) > max_diff_chars
        else:
            context["diff_error"] = diff.stderr.strip()[:1000]
    except Exception as exc:
        context["diff_error"] = f"{type(exc).__name__}: {exc}"

    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=project_path,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if untracked.returncode == 0:
            snippets = []
            remaining_chars = max_untracked_chars
            for rel_path in untracked.stdout.splitlines()[:max_untracked_files]:
                normalized = safe_relative_path(rel_path)
                if normalized is None:
                    continue
                abs_path = os.path.join(project_path, normalized)
                if not os.path.isfile(abs_path):
                    continue
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as handle:
                        content = handle.read(max(0, remaining_chars))
                except OSError as exc:
                    snippets.append({"path": normalized, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                if not content:
                    continue
                remaining_chars -= len(content)
                snippets.append(
                    {
                        "path": normalized,
                        "content": content,
                        "truncated": remaining_chars <= 0,
                    }
                )
                if remaining_chars <= 0:
                    break
            context["untracked_file_snippets"] = snippets
        else:
            context["untracked_error"] = untracked.stderr.strip()[:1000]
    except Exception as exc:
        context["untracked_error"] = f"{type(exc).__name__}: {exc}"

    return context


def _compact_filesystem_diff_context(
    project_path: str,
    source_root: str,
    *,
    max_diff_chars: int = 8000,
    max_files: int = 30,
) -> dict[str, Any]:
    """Collect a diff by comparing the copied worktree against its source tree."""
    context: dict[str, Any] = {
        "changed_files": [],
        "diff": "",
        "diff_truncated": False,
        "untracked_file_snippets": [],
    }
    if not os.path.isdir(project_path) or not os.path.isdir(source_root):
        context["error"] = f"missing project/source root: {project_path} / {source_root}"
        return context

    def iter_files(root: str) -> set[str]:
        output = set()
        for current_root, dir_names, file_names in os.walk(root):
            dir_names[:] = [
                name for name in dir_names
                if name not in {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
            ]
            for file_name in file_names:
                if not file_name.endswith(".py"):
                    continue
                abs_path = os.path.join(current_root, file_name)
                rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
                output.add(rel_path)
        return output

    def read_lines(path: str) -> list[str]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read().splitlines()
        except OSError:
            return []

    diff_parts = []
    remaining = max_diff_chars
    try:
        rel_paths = sorted(iter_files(project_path) | iter_files(source_root))
        for rel_path in rel_paths:
            project_file = os.path.join(project_path, rel_path)
            source_file = os.path.join(source_root, rel_path)
            project_exists = os.path.isfile(project_file)
            source_exists = os.path.isfile(source_file)
            if not project_exists and not source_exists:
                continue
            project_lines = read_lines(project_file) if project_exists else []
            source_lines = read_lines(source_file) if source_exists else []
            if project_lines == source_lines:
                continue
            context["changed_files"].append(rel_path)
            if remaining > 0:
                diff_text = "\n".join(
                    difflib.unified_diff(
                        source_lines,
                        project_lines,
                        fromfile=f"a/{rel_path}",
                        tofile=f"b/{rel_path}",
                        lineterm="",
                    )
                )
                if diff_text:
                    chunk = diff_text[:remaining]
                    diff_parts.append(chunk)
                    remaining -= len(chunk)
                    if len(diff_text) > len(chunk):
                        context["diff_truncated"] = True
            if len(context["changed_files"]) >= max_files:
                if len(context["changed_files"]) < len(rel_paths):
                    context["diff_truncated"] = True
                break
    except Exception as exc:
        context["diff_error"] = f"{type(exc).__name__}: {exc}"

    context["diff"] = "\n".join(diff_parts)
    return context


def _truncate_handoff_value(value: Any, *, string_limit: int = 240, list_limit: int = 4) -> Any:
    if isinstance(value, str):
        return value if len(value) <= string_limit else value[:string_limit] + "...[truncated]"
    if isinstance(value, list):
        return [
            _truncate_handoff_value(item, string_limit=string_limit, list_limit=list_limit)
            for item in value[:list_limit]
        ]
    if isinstance(value, dict):
        return {
            key: _truncate_handoff_value(item, string_limit=string_limit, list_limit=list_limit)
            for key, item in value.items()
            if item not in (None, [], {})
        }
    return value


def _compact_validation_for_handoff(validation: dict[str, Any] | None) -> dict[str, Any]:
    """Reduce validation output to bounded repair evidence for the next agent.

    The compact form keeps the primary failure, repair target, and a small
    sample of hard split/group failures. It is prompt context only and must not
    be used as a replacement for the complete validation result.
    """
    if not validation:
        return {}
    validation = model_facing_diagnostics(validation)
    compact: dict[str, Any] = {
        key: _truncate_handoff_value(validation.get(key))
        for key in (
            "passed",
            "failed_stage",
            "rule_group_id",
            "rule_group_ids",
            "error_type",
            "error",
            "actual_sql",
            "expected_sql",
            "diff_summary",
        )
        if validation.get(key) not in (None, [], {})
    }

    repair_target = validation.get("repair_target")
    if isinstance(repair_target, dict):
        target = {
            key: _truncate_handoff_value(repair_target.get(key))
            for key in (
                "stage",
                "rule_group_id",
                "rule_group_ids",
                "reason",
                "instruction",
                "error_type",
                "error",
            )
            if repair_target.get(key) not in (None, [], {})
        }
        failed_splits = []
        for split in repair_target.get("failed_splits") or []:
            if not isinstance(split, dict):
                continue
            failed_splits.append(
                {
                    key: _truncate_handoff_value(split.get(key))
                    for key in (
                        "split_id",
                        "rule_group_id",
                        "actual_sql",
                        "expected_sql",
                        "diff_summary",
                        "error_type",
                        "error",
                    )
                    if split.get(key) not in (None, [], {})
                }
            )
            if len(failed_splits) >= 2:
                break
        if failed_splits:
            target["failed_splits"] = failed_splits
        compact["repair_target"] = target

    failed_groups = []
    for group in validation.get("group_results") or []:
        if not isinstance(group, dict) or group.get("passed"):
            continue
        failed_groups.append(
            {
                key: _truncate_handoff_value(group.get(key))
                for key in ("rule_group_id", "failed_split_count", "passed")
                if group.get(key) not in (None, [], {})
            }
        )
        if len(failed_groups) >= 3:
            break
    if failed_groups:
        compact["failed_groups"] = failed_groups
    failed_group_evidence = []
    seen_group_ids: set[str] = set()
    for split in validation.get("split_results") or []:
        if not isinstance(split, dict):
            continue
        if split.get("passed") or not split.get("hard_assertion", True):
            continue
        group_id = str(split.get("rule_group_id") or "").strip()
        if not group_id or group_id in seen_group_ids:
            continue
        selected_diff = split.get("selected_target_diff")
        if not isinstance(selected_diff, dict):
            selected_diff = {}
        evidence = {
            key: _truncate_handoff_value(value)
            for key, value in (
                ("rule_group_id", group_id),
                ("split_id", split.get("split_id")),
                ("actual_sql", split.get("actual_sql")),
                ("expected_sql", split.get("expected_sql")),
                ("diff_summary", split.get("diff_summary", [])),
                ("source_node_sql", selected_diff.get("source_node_sql")),
                ("actual_node_sql", selected_diff.get("actual_node_sql")),
                ("expected_node_sql", selected_diff.get("expected_node_sql")),
                ("actual_only", selected_diff.get("actual_only")),
                ("expected_only", selected_diff.get("expected_only")),
                ("transition_summary", split.get("transition_summary")),
                ("error_type", split.get("error_type")),
                ("error", split.get("error")),
            )
            if value not in (None, [], {})
        }
        failed_group_evidence.append(evidence)
        seen_group_ids.add(group_id)
        if len(failed_group_evidence) >= 5:
            break
    if failed_group_evidence:
        compact["failed_group_evidence"] = failed_group_evidence
    return compact


def _compact_group_execution_for_handoff(group_execution: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize attempted, passed, repair, and failed group state for handoff.

    Pending groups without attempts are omitted and all lists are capped so a
    long execution queue cannot dominate the next stage prompt.
    """
    from code_utils.sql_ut.group_execution import (
        attempted_group_ids,
        failed_group_records,
        group_ids_by_status,
        ordered_group_statuses,
    )

    if not group_execution:
        return {}
    statuses = []
    for status in ordered_group_statuses(group_execution):
        if not isinstance(status, dict):
            continue
        group_status = status.get("status")
        attempts = int(status.get("attempts") or 0)
        if not attempts and group_status == "pending":
            continue
        statuses.append(
            {
                key: _truncate_handoff_value(status.get(key))
                for key in (
                    "rule_group_id",
                    "status",
                    "attempts",
                    "last_stage",
                    "last_failed_stage",
                )
                if status.get(key) not in (None, [], {})
            }
        )
        if len(statuses) >= 8:
            break
    failed_groups = []
    for failed in failed_group_records(group_execution):
        if not isinstance(failed, dict):
            continue
        repair_target = failed.get("repair_target") if isinstance(failed.get("repair_target"), dict) else {}
        item = {
            key: _truncate_handoff_value(failed.get(key))
            for key in ("rule_group_id", "attempts", "reason")
            if failed.get(key) not in (None, [], {})
        }
        if repair_target:
            item["repair_target"] = {
                key: _truncate_handoff_value(repair_target.get(key))
                for key in ("stage", "rule_group_id", "reason")
                if repair_target.get(key) not in (None, [], {})
            }
        failed_groups.append(item)
        if len(failed_groups) >= 3:
            break
    compact = {
        "attempted_groups": attempted_group_ids(group_execution)[-8:],
        "passed_groups": group_ids_by_status(group_execution, "passed")[-8:],
        "repair_groups": group_ids_by_status(group_execution, "repair")[-8:],
        "failed_groups": failed_groups,
        "group_statuses": statuses,
    }
    return {key: value for key, value in compact.items() if value}


def _compact_patch_snippets(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for snippet in snippets[:2]:
        if not isinstance(snippet, dict):
            continue
        compact.append(
            {
                "file": snippet.get("file"),
                "hunk_header": _truncate_handoff_value(snippet.get("hunk_header"), string_limit=140),
                "added_lines": [
                    _truncate_handoff_value(line, string_limit=120)
                    for line in (snippet.get("added_lines") or [])[:3]
                ],
                "removed_lines": [
                    _truncate_handoff_value(line, string_limit=120)
                    for line in (snippet.get("removed_lines") or [])[:1]
                ],
                "symbols": list(snippet.get("symbols") or [])[:5],
            }
        )
    return compact


def _build_stage_handoff_payload(
    *,
    worktree_context: dict[str, Any],
    group_execution: dict[str, Any] | None,
    stage: str,
    group: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    agent_success: bool | None = None,
    agent_calls: int | None = None,
) -> dict[str, Any]:
    """Combine stage outcome, group progress, and patch evidence into a handoff.

    ``worktree_context`` is already read-only evidence collected by the caller.
    This function only normalizes and bounds it into the handoff schema.
    """
    record = {
        "gen_code": worktree_context.get("diff") or "",
        "file_changes": worktree_context.get("changed_files") or [],
    }
    changed_files = extract_changed_files(record) or list(worktree_context.get("changed_files") or [])
    patch_evidence = extract_patch_evidence(record, changed_files)
    return compact_output_fields(
        {
            "schema": "autodialect_stage_handoff_v1",
            "previous_stage": {
                "stage": stage,
                "rule_group_id": (group or {}).get("rule_group_id"),
                "rule_group_ids": (group or {}).get("rule_group_ids")
                or _normalize_group_ids((group or {}).get("rule_group_id")),
                "agent_success": agent_success,
                "agent_calls_so_far": agent_calls,
            },
            "instructions": list(STAGE_HANDOFF_INSTRUCTIONS),
            "group_progress": _compact_group_execution_for_handoff(group_execution),
            "validation": _compact_validation_for_handoff(validation),
            "worktree": {
                "changed_files": changed_files[:12],
                "diff_chars": len(worktree_context.get("diff") or ""),
                "diff_truncated": bool(worktree_context.get("diff_truncated")),
                "status_error": worktree_context.get("status_error"),
                "diff_error": worktree_context.get("diff_error"),
                "patch_tags": infer_patch_tags(changed_files, record)[:8],
                "sqlglot_changed_files": patch_evidence.get("sqlglot_changed_files", [])[:8],
                "auxiliary_changed_files": patch_evidence.get("auxiliary_changed_files", [])[:5],
                "patch_symbols": patch_evidence.get("patch_symbols", [])[:12],
                "patch_snippets": _compact_patch_snippets(
                    patch_evidence.get("patch_snippets") or []
                ),
            },
        }
    )


def _build_stage_handoff(
    *,
    project_path: str,
    base_ref: str | None,
    source_root: str | None = None,
    group_execution: dict[str, Any] | None,
    stage: str,
    group: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    agent_success: bool | None = None,
    agent_calls: int | None = None,
) -> dict[str, Any] | None:
    """Build the optional compact handoff passed between AUTODIALECT pipeline stages.

    When handoff is disabled, return ``None`` without inspecting the worktree.
    A copied source tree uses filesystem comparison; normal runs use the git
    base reference. Collection errors remain visible in the returned context.
    """
    if not STAGE_HANDOFF_ENABLED:
        return None
    if source_root:
        worktree_context = _compact_filesystem_diff_context(
            project_path,
            source_root,
            max_diff_chars=8000,
        )
    else:
        worktree_context = _compact_worktree_context(
            project_path,
            base_ref,
            max_diff_chars=8000,
            max_untracked_files=0,
        )
    return _build_stage_handoff_payload(
        worktree_context=worktree_context,
        group_execution=group_execution,
        stage=stage,
        group=group,
        validation=validation,
        agent_success=agent_success,
        agent_calls=agent_calls,
    )


# 4. JSON subprocess and parser-gate helpers.
def _normalize_helper_result(
    result: dict[str, Any],
    *,
    duration_s: float,
    timeout: int | float,
) -> dict[str, Any]:
    normalized = dict(result)
    if "ok" not in normalized:
        status = str(normalized.get("status") or "")
        normalized["ok"] = status.endswith("_ok") or status == "ok"
    if "status" not in normalized:
        normalized["status"] = "helper_ok" if normalized.get("ok") else "helper_failed"
    normalized.setdefault("helper_protocol", "autodialect_python_json_v1")
    normalized.setdefault("helper_duration_s", round(duration_s, 4))
    normalized.setdefault("helper_timeout_s", timeout)
    return normalized


def _run_python_json_command(
    command: list[str],
    args: list[str] | None = None,
    timeout: int = 60,
    stdin_json: dict[str, Any] | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    """Run a Python helper and return its structured protocol result.

    The last sentinel-prefixed line is authoritative; plain final-line JSON is
    accepted for legacy helpers. Timeout, process, empty-output, and JSON errors
    are converted to failure dictionaries instead of being raised. ``normalize``
    adds the shared protocol, duration, and timeout fields.
    """
    import json

    started = time.monotonic()

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        if not normalize:
            return result
        return _normalize_helper_result(
            result,
            duration_s=time.monotonic() - started,
            timeout=timeout,
        )

    command = list(command)
    if args:
        command.extend(args)
    stdin_text = json.dumps(stdin_json, ensure_ascii=False) if stdin_json is not None else None

    try:
        completed = subprocess.run(
            command,
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return finish(
            {
                "ok": False,
                "status": "helper_timeout",
                "error_type": "TimeoutExpired",
                "error": f"helper timed out after {exc.timeout} seconds",
            }
        )
    except OSError as exc:
        return finish(
            {
                "ok": False,
                "status": "helper_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    stdout = (completed.stdout or "").strip()
    if completed.returncode != 0:
        return finish(
            {
                "ok": False,
                "status": "helper_error",
                "error_type": "SubprocessError",
                "error": (completed.stderr or stdout or f"exit code {completed.returncode}")[:4000],
            }
        )
    if not stdout:
        return finish(
            {
                "ok": False,
                "status": "helper_error",
                "error_type": "ProtocolError",
                "error": "empty stdout",
            }
        )
    for line in reversed(stdout.splitlines()):
        if line.startswith(JSON_SENTINEL):
            try:
                return finish(json.loads(line[len(JSON_SENTINEL):]))
            except json.JSONDecodeError as exc:
                return finish(
                    {
                        "ok": False,
                        "status": "helper_error",
                        "error_type": "JSONDecodeError",
                        "error": f"{exc}: {line[-1000:]}",
                    }
                )
    try:
        return finish(json.loads(stdout.splitlines()[-1]))
    except json.JSONDecodeError as exc:
        return finish(
            {
                "ok": False,
                "status": "helper_error",
                "error_type": "JSONDecodeError",
                "error": f"{exc}: {stdout[-1000:]}",
            }
        )


def run_python_json(
    script: str,
    args: list[str] | None = None,
    timeout: int = 60,
    stdin_json: dict[str, Any] | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    """Run a Python helper script string and parse its sentinel-prefixed JSON output."""
    return _run_python_json_command(
        [AUTODIALECT_PYTHON_BIN, "-c", script],
        args=args,
        timeout=timeout,
        stdin_json=stdin_json,
        normalize=normalize,
    )


def run_parser_gate(*, project_path: str, sql: str, dialect: str, role: str = "source") -> dict[str, Any]:
    """Parse one SQL against the specified SQLGlot worktree and dialect.

    The isolated helper returns ``parse_ok`` with compact AST evidence or a
    structured parse/runtime failure. ``role`` labels the result for source vs.
    target repair routing but does not alter parsing behavior.
    """
    result = run_python_json(
        _PARSER_GATE_SCRIPT,
        stdin_json={"package_root": project_path, "sql": sql, "dialect": dialect},
        normalize=True,
    )
    if "status" not in result:
        result["status"] = "parse_ok" if result.get("ok") else "parse_failed"
    result["role"] = role
    result["dialect"] = dialect
    return result


def run_python_json_file(
    script_path: str,
    args: list[str] | None = None,
    timeout: int = 60,
    stdin_json: dict[str, Any] | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    """Run a Python helper file and parse its sentinel-prefixed JSON output."""
    return _run_python_json_command(
        [AUTODIALECT_PYTHON_BIN, script_path],
        args=args,
        timeout=timeout,
        stdin_json=stdin_json,
        normalize=normalize,
    )
