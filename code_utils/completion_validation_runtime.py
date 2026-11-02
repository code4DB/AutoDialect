"""Completion validation protocol helpers for AutoDialect.

This module owns the contract between rule_main and the AUTODIALECT agent:
1. Shared protocol helpers: normalize group ids, build validation stage
   records, and decide whether an in-agent audit can be reused by rule_main.
2. rule_main-side parser-gate failure records: convert unrepaired source or
   target parser-gate failures into hard validation results.
3. rule_main-side validation-spec writing: write rule and parser-repair
   completion-validation specs for the AUTODIALECT agent.
4. AUTODIALECT-agent-side validation execution: load specs and run parser-gate or
   split-manifest validation after task_done.
5. AUTODIALECT-agent-side failure messaging: render focused repair instructions for
   the next agent attempt.
6. AUTODIALECT-agent-side audit recording: persist completion-validation state into
   memory_audit.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

from code_utils.common_tool import (
    compact_output_fields,
    model_facing_diagnostics,
    strip_legacy_family_metadata as _strip_legacy_family_metadata,
)
from code_utils.constants import AUTODIALECT_COMPLETION_VALIDATION_ENABLED
from code_utils.experiment_utils import (
    ExperimentSpec,
    case_expected_sql,
    case_source_sql,
    case_src_dialect,
    case_tgt_dialect,
    save_json,
)
from code_utils.sql_ut.group_execution import focus_validation_on_group
from code_utils.sql_ut.sql_ut_splitter import split_is_reliable_rule_evidence
from prompt.agent_prompt import (
    COMPLETION_VALIDATION_GROUP_REPAIR_INSTRUCTION,
    COMPLETION_VALIDATION_PASSED_INSTRUCTION,
    COMPLETION_VALIDATION_RETRY_PROMPT,
    COMPLETION_VALIDATION_SINGLE_REPAIR_INSTRUCTION,
)

COMPLETION_VALIDATION_ENABLED = AUTODIALECT_COMPLETION_VALIDATION_ENABLED
RULE_COMPLETION_VALIDATION_SCHEMA = "autodialect_completion_validation_v1"
PARSER_COMPLETION_VALIDATION_SCHEMA = "autodialect_parser_completion_validation_v1"


# 1. Shared protocol helpers used by rule_main and AUTODIALECT-agent audit handling.
def _normalize_group_ids(value: Any) -> list[str]:
    """Normalize one or more group ids into a stable deduplicated list."""
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


def _validation_stage_record(
    *,
    stage_name: str,
    validation: dict[str, Any],
    scope: str,
    focus_rule_group_id: Any | None,
    validation_source: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Build the validation stage record persisted by rule_main."""
    focus_rule_group_ids = _normalize_group_ids(focus_rule_group_id)
    single_focus_rule_group_id = focus_rule_group_ids[0] if len(focus_rule_group_ids) == 1 else None
    record = {
        "stage": stage_name,
        "validation_scope": scope,
        "focus_rule_group_id": single_focus_rule_group_id,
        "focus_rule_group_ids": focus_rule_group_ids,
        "validation_source": validation_source,
        "passed": validation.get("passed"),
        "failed_stage": validation.get("failed_stage"),
        "warning_stage": validation.get("warning_stage"),
        "warning_reason": validation.get("warning_reason"),
        "validation_warning": validation.get("validation_warning"),
        "repair_target": validation.get("repair_target"),
        "focused_repair_target": validation.get("focused_repair_target"),
        "duration_s": validation.get("duration_s"),
    }
    if fallback_reason:
        record["fallback_reason"] = fallback_reason
    return compact_output_fields(record)


def _completion_validation_result_from_audit(
    audit: dict[str, Any] | None,
    *,
    expected_scope: str,
    expected_spec_path: str | None = None,
    expected_focus_rule_group_id: str | None = None,
    expected_focus_rule_group_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Reuse an in-agent result only when it proves the requested validation.

    The audit must be enabled, final for the current completion attempt,
    contain a result dictionary, and match the expected scope and focused group ids.
    Returns ``(result, None)`` on reuse or ``(None, reason)`` so rule_main can
    record why it must perform outer validation instead.
    """
    if not isinstance(audit, dict):
        return None, "missing_completion_validation_audit"
    if not audit.get("enabled"):
        return None, "completion_validation_disabled"
    if not (
        audit.get("completed_current_task_done")
        or audit.get("completed_current_attempt")
    ):
        return None, "completion_validation_not_final"
    if expected_spec_path:
        actual_spec_path = audit.get("spec_path")
        if actual_spec_path != expected_spec_path:
            return None, f"completion_validation_spec_path_mismatch:{actual_spec_path or 'unknown'}"
    result = audit.get("final_result")
    if not isinstance(result, dict):
        return None, "missing_final_validation_result"
    if result.get("failed_stage") == "completion_validation_spec":
        error_type = result.get("error_type") or "unknown"
        return None, f"completion_validation_spec_error:{error_type}"

    actual_scope = result.get("validation_scope") or audit.get("validation_scope")
    if actual_scope != expected_scope:
        return None, f"validation_scope_mismatch:{actual_scope or 'unknown'}"

    if expected_scope == "focused_group":
        expected_group_ids = _normalize_group_ids(
            expected_focus_rule_group_ids or expected_focus_rule_group_id
        )
        actual_group_ids = _normalize_group_ids(
            result.get("focus_rule_group_ids")
            or audit.get("focus_rule_group_ids")
            or result.get("focus_rule_group_id")
            or audit.get("focus_rule_group_id")
        )
        if expected_group_ids and set(actual_group_ids) != set(expected_group_ids):
            if len(expected_group_ids) == 1:
                actual = actual_group_ids[0] if len(actual_group_ids) == 1 else ",".join(actual_group_ids)
                return None, f"focus_rule_group_id_mismatch:{actual or 'unknown'}"
            return None, f"focus_rule_group_ids_mismatch:{','.join(actual_group_ids) or 'unknown'}"

    return result, None


# 2. rule_main-side parser-gate failure records.
def _target_parser_failure_validation(target_parser_gate_result: dict[str, Any]) -> dict[str, Any]:
    """Convert an unrepaired target parser-gate failure into validation output."""
    repair_target = {
        "stage": "target_parser",
        "reason": "target_parser_gate_failed_after_parser_repair",
        "status": target_parser_gate_result.get("status"),
        "error_type": target_parser_gate_result.get("error_type"),
        "error": target_parser_gate_result.get("error"),
        "instruction": (
            "The expected target SQL still cannot be parsed, so UT split and variant validation "
            "cannot be trusted. Fix target dialect parser support before continuing rule generation."
        ),
    }
    return compact_output_fields({
        "passed": False,
        "failed_stage": "target_parser_gate",
        "error_type": target_parser_gate_result.get("error_type"),
        "error": target_parser_gate_result.get("error"),
        "repair_hint": "Target parser gate failed after parser repair; finish target parser support before UT split validation.",
        "repair_target": repair_target,
    })


def _source_parser_failure_validation(parser_gate_result: dict[str, Any]) -> dict[str, Any]:
    """Convert an unrepaired source parser-gate failure into validation output."""
    repair_target = {
        "stage": "source_parser",
        "reason": "source_parser_gate_failed_after_parser_repair",
        "status": parser_gate_result.get("status"),
        "error_type": parser_gate_result.get("error_type"),
        "error": parser_gate_result.get("error"),
        "instruction": (
            "The source SQL still cannot be parsed after parser repair. "
            "Finish source dialect parser/tokenizer support before rule generation can be validated."
        ),
    }
    return compact_output_fields({
        "passed": False,
        "failed_stage": "parser_gate",
        "error_type": parser_gate_result.get("error_type"),
        "error": parser_gate_result.get("error"),
        "repair_hint": "Source parser gate failed after parser repair; finish source parser support first.",
        "repair_target": repair_target,
    })


# 3. rule_main-side validation-spec writing.
def _validation_spec_location(spec: ExperimentSpec, rule_name: str, stage: str) -> tuple[str, dict[str, Any]]:
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "stage"
    safe_rule = re.sub(r"[^A-Za-z0-9_.-]+", "_", rule_name).strip("_") or "case"
    unique_token = f"pid{os.getpid()}_{time.time_ns()}"
    spec_id = f"{safe_rule}:{safe_stage}:{unique_token}"
    path = os.path.abspath(
        os.path.join(
            spec.result_folder,
            safe_rule,
            "validation_specs",
            f"{safe_stage}__{unique_token}.json",
        )
    )
    return path, {
        "spec_id": spec_id,
        "spec_path": path,
        "spec_created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "spec_writer_pid": os.getpid(),
        "rule_name": rule_name,
        "stage": stage,
    }


def _write_completion_validation_spec(
    *,
    spec: ExperimentSpec,
    rule_name: str,
    stage: str,
    split_manifest: dict[str, Any] | None,
    case: dict[str, Any],
    focus_rule_group_id: Any | None = None,
    focus_rule_group_ids: list[str] | tuple[str, ...] | None = None,
    validation_scope: str = "full_manifest",
) -> str | None:
    """Write the rule-validation contract consumed by the AUTODIALECT agent."""
    if not COMPLETION_VALIDATION_ENABLED or not split_manifest:
        return None
    normalized_focus_group_ids = _normalize_group_ids(focus_rule_group_ids or focus_rule_group_id)
    single_focus_rule_group_id = (
        normalized_focus_group_ids[0] if len(normalized_focus_group_ids) == 1 else None
    )
    path, metadata = _validation_spec_location(spec, rule_name, stage)
    reliable_splits = [
        split
        for split in split_manifest.get("splits", [])
        if isinstance(split, dict) and split_is_reliable_rule_evidence(split)
    ]
    reliable_split_ids = {
        str(split.get("split_id"))
        for split in reliable_splits
        if split.get("split_id") not in (None, "")
    }
    reliable_splits_by_group: dict[str, list[dict[str, Any]]] = {}
    for split in reliable_splits:
        group_id = str(split.get("rule_group_id") or "")
        if group_id:
            reliable_splits_by_group.setdefault(group_id, []).append(split)
    reliable_groups = []
    for group in split_manifest.get("groups", []):
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("rule_group_id") or "")
        group_splits = reliable_splits_by_group.get(group_id, [])
        if not group_splits:
            continue
        reliable_group = dict(group)
        reliable_group["split_ids"] = [
            split_id
            for split_id in group.get("split_ids", [])
            if str(split_id) in reliable_split_ids
        ]
        reliable_group["hard_assertion_count"] = len(group_splits)
        reliable_group["skipped_split_count"] = 0
        hard_signature = next(
            (split.get("rule_signature") for split in group_splits if split.get("rule_signature")),
            None,
        )
        if hard_signature:
            reliable_group["rule_signature"] = hard_signature
        reliable_groups.append(reliable_group)
    validation_manifest = {
        **split_manifest,
        "splits": reliable_splits,
        "groups": reliable_groups,
        "completion_validation_projection": {
            "original_split_count": len(split_manifest.get("splits", [])),
            "reliable_split_count": len(reliable_splits),
            "excluded_split_count": len(split_manifest.get("splits", [])) - len(reliable_splits),
        },
    }
    save_json(
        path,
        {
            **metadata,
            "schema": RULE_COMPLETION_VALIDATION_SCHEMA,
            "split_manifest": validation_manifest,
            "parent_source_sql": case_source_sql(spec, case),
            "parent_expected_sql": case_expected_sql(spec, case),
            "src_dialect": case_src_dialect(spec, case),
            "tgt_dialect": case_tgt_dialect(spec, case),
            "focus_rule_group_id": single_focus_rule_group_id,
            "focus_rule_group_ids": normalized_focus_group_ids,
            "validation_scope": validation_scope,
            "validate_variants": True,
        },
    )
    return path


def _write_parser_completion_validation_spec(
    *,
    spec: ExperimentSpec,
    rule_name: str,
    stage: str,
    sql: str,
    dialect: str,
    parser_role: str,
) -> str | None:
    """Write the parser-gate validation contract consumed by the AUTODIALECT agent."""
    if not COMPLETION_VALIDATION_ENABLED:
        return None
    path, metadata = _validation_spec_location(spec, rule_name, stage)
    save_json(
        path,
        {
            **metadata,
            "schema": PARSER_COMPLETION_VALIDATION_SCHEMA,
            "validation_kind": "parser_gate",
            "parser_role": parser_role,
            "sql": sql,
            "dialect": dialect,
            "validation_scope": "parser_gate",
        },
    )
    return path


# 4. AUTODIALECT-agent-side spec loading and validation execution.
def _invalid_completion_validation_spec(
    completion_validation_path: str,
    message: str,
) -> dict[str, Any]:
    return {
        "schema": RULE_COMPLETION_VALIDATION_SCHEMA,
        "spec_path": completion_validation_path,
        "load_error_type": "InvalidCompletionValidationSpec",
        "load_error": message,
    }


def _validate_completion_validation_spec(
    payload: dict[str, Any],
    completion_validation_path: str,
) -> dict[str, Any]:
    """Validate parser/rule spec shape without executing project code.

    Valid payloads are returned with ``spec_source='file'``. Invalid payloads
    become structured load-error specs so the agent reports a runtime contract
    failure rather than treating malformed input as a SQL rule failure.
    """
    schema = payload.get("schema")
    if schema == PARSER_COMPLETION_VALIDATION_SCHEMA:
        if payload.get("validation_kind") != "parser_gate":
            return _invalid_completion_validation_spec(
                completion_validation_path,
                "Parser completion validation spec must set validation_kind='parser_gate'.",
            )
        for key in ("sql", "dialect", "parser_role"):
            if not isinstance(payload.get(key), str) or not payload[key].strip():
                return _invalid_completion_validation_spec(
                    completion_validation_path,
                    f"Parser completion validation spec requires a non-empty string field {key!r}.",
                )
    elif schema == RULE_COMPLETION_VALIDATION_SCHEMA:
        manifest = payload.get("split_manifest")
        if not isinstance(manifest, dict):
            return _invalid_completion_validation_spec(
                completion_validation_path,
                "Rule completion validation spec requires split_manifest to be a JSON object.",
            )
        if not isinstance(manifest.get("splits"), list) or not isinstance(manifest.get("groups"), list):
            return _invalid_completion_validation_spec(
                completion_validation_path,
                "Rule completion validation split_manifest requires list fields 'splits' and 'groups'.",
            )
        for key in ("parent_source_sql", "parent_expected_sql", "src_dialect", "tgt_dialect"):
            if not isinstance(payload.get(key), str) or not payload[key].strip():
                return _invalid_completion_validation_spec(
                    completion_validation_path,
                    f"Rule completion validation spec requires a non-empty string field {key!r}.",
                )
        validation_scope = payload.get("validation_scope") or "full_manifest"
        if validation_scope not in {"full_manifest", "focused_group"}:
            return _invalid_completion_validation_spec(
                completion_validation_path,
                f"Unsupported completion validation scope {validation_scope!r}.",
            )
        if (
            validation_scope == "focused_group"
            and not _normalize_group_ids(
                payload.get("focus_rule_group_ids") or payload.get("focus_rule_group_id")
            )
        ):
            return _invalid_completion_validation_spec(
                completion_validation_path,
                "Focused completion validation requires at least one rule group id.",
            )
    else:
        return _invalid_completion_validation_spec(
            completion_validation_path,
            f"Unsupported completion validation schema {schema!r}.",
        )
    payload.setdefault("spec_source", "file")
    return payload


def load_completion_validation_spec(completion_validation_path: str | None) -> dict[str, Any] | None:
    """Load a JSON validation spec, returning a structured load error on failure."""
    if not completion_validation_path:
        return None
    payload: dict[str, Any] | Any
    try:
        with open(completion_validation_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        return {
            "schema": RULE_COMPLETION_VALIDATION_SCHEMA,
            "spec_path": completion_validation_path,
            "load_error_type": type(exc).__name__,
            "load_error": str(exc),
        }
    if not isinstance(payload, dict):
        return _invalid_completion_validation_spec(
            completion_validation_path,
            "Completion validation spec must be a JSON object.",
        )
    return _validate_completion_validation_spec(payload, completion_validation_path)


def _focus_completion_validation_on_group(
    validation: dict[str, Any],
    group_id: str | None,
) -> dict[str, Any]:
    """Prefer the active group as completion-validation repair target."""
    return _strip_legacy_family_metadata(
        focus_validation_on_group(
            validation,
            group_id,
            instruction="Repair only the focused group from the current agent call.",
            include_confidence=True,
        )
    )


def run_completion_validation(
    *,
    project_path: str,
    completion_validation_path: str | None,
    spec: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Execute the loaded completion-validation contract after ``task_done``.

    Parser specs run one parser gate; rule specs run split/variant validation in
    either full-manifest or focused-group scope. Spec and runtime exceptions are
    returned as repair-shaped failures. Full validation may retarget a failure
    to the active group, while focused validation preserves its scoped result.
    """
    if not spec:
        return None
    if spec.get("load_error"):
        return {
            "passed": False,
            "failed_stage": "completion_validation_spec",
            "runtime_error": True,
            "spec_path": spec.get("spec_path") or completion_validation_path,
            "error_type": spec.get("load_error_type"),
            "error": spec.get("load_error"),
            "repair_target": {
                "stage": "completion_validation_spec",
                "reason": "failed_to_load_completion_validation_spec",
                "instruction": (
                    "This is an AutoDialect validation-spec runtime error, not a SQL rule error. "
                    "Do not rewrite unrelated code."
                ),
            },
        }

    if spec.get("validation_kind") == "parser_gate":
        try:
            from code_utils.execution_runtime import run_parser_gate

            gate = run_parser_gate(
                project_path=project_path,
                sql=str(spec.get("sql") or ""),
                dialect=str(spec.get("dialect") or ""),
                role=str(spec.get("parser_role") or "source"),
            )
            passed = gate.get("status") == "parse_ok"
            parser_role = str(spec.get("parser_role") or gate.get("role") or "source")
            is_target = parser_role.startswith("target")
            failed_stage = None if passed else ("target_parser_gate" if is_target else "parser_gate")
            return compact_output_fields({
                "passed": passed,
                "failed_stage": failed_stage,
                "validation_scope": "parser_gate",
                "parser_role": parser_role,
                "dialect": spec.get("dialect"),
                "parser_gate": gate,
                "error_type": gate.get("error_type"),
                "error": gate.get("error"),
                "repair_target": None if passed else {
                    "stage": "target_parser" if is_target else "source_parser",
                    "reason": "parser_gate_failed_after_task_done",
                    "status": gate.get("status"),
                    "error_type": gate.get("error_type"),
                    "error": gate.get("error"),
                    "instruction": (
                        "Continue repairing parser/tokenizer support in the current worktree. "
                        "Do not implement translation rules in this parser repair stage; call task_done "
                        "again only after sqlglot.parse_one succeeds for the configured SQL and dialect."
                    ),
                },
            })
        except Exception as exc:
            return compact_output_fields({
                "passed": False,
                "failed_stage": "parser_gate_validation_error",
                "validation_scope": "parser_gate",
                "parser_role": spec.get("parser_role"),
                "dialect": spec.get("dialect"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "repair_target": {
                    "stage": "parser_gate_validation_error",
                    "reason": "parser_completion_validation_raised",
                    "instruction": "Inspect the parser validation runtime error and repair the relevant dialect/parser code.",
                },
            })

    try:
        from code_utils.sql_ut.split_validator import validate_manifest

        validation = validate_manifest(
            package_root=project_path,
            sys_path_backup=list(sys.path),
            manifest=spec.get("split_manifest"),
            parent_source_sql=str(spec.get("parent_source_sql") or ""),
            parent_expected_sql=str(spec.get("parent_expected_sql") or ""),
            src_dialect=str(spec.get("src_dialect") or ""),
            tgt_dialect=str(spec.get("tgt_dialect") or ""),
            validate_variants=bool(spec.get("validate_variants", True)),
            validation_scope=str(spec.get("validation_scope") or "full_manifest"),
            focus_rule_group_id=spec.get("focus_rule_group_id"),
            focus_rule_group_ids=spec.get("focus_rule_group_ids"),
        )
    except Exception as exc:
        validation = {
            "passed": False,
            "failed_stage": "validation_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "repair_target": {
                "stage": "validation_error",
                "reason": "completion_validation_raised",
                "instruction": "Inspect the validation error and repair the relevant dialect/parser code.",
            },
        }
    focus_group_id = spec.get("focus_rule_group_id")
    if str(spec.get("validation_scope") or "full_manifest") == "full_manifest":
        return _focus_completion_validation_on_group(validation, focus_group_id)
    return validation


# 5. AUTODIALECT-agent-side failure message rendering.
def _compact_failed_group_evidence(
    validation: dict[str, Any],
    *,
    group_limit: int = 5,
) -> list[dict[str, Any]]:
    """Select bounded hard-failure evidence while maximizing group coverage.

    The first pass keeps one failed split per group; remaining slots may include
    additional unique splits. Legacy metadata is removed before prompt use.
    """
    candidates: list[dict[str, Any]] = []
    for split in validation.get("split_results") or []:
        if not isinstance(split, dict):
            continue
        if split.get("passed") or not split.get("hard_assertion", True):
            continue
        group_id = str(split.get("rule_group_id") or "").strip()
        if not group_id:
            continue
        selected_diff = split.get("selected_target_diff")
        if not isinstance(selected_diff, dict):
            selected_diff = {}
        item = {
            "rule_group_id": group_id,
            "split_id": split.get("split_id"),
            "actual_sql": split.get("actual_sql"),
            "expected_sql": split.get("expected_sql"),
            "diff_summary": split.get("diff_summary", []),
            "source_node_sql": selected_diff.get("source_node_sql"),
            "actual_node_sql": selected_diff.get("actual_node_sql"),
            "expected_node_sql": selected_diff.get("expected_node_sql"),
            "actual_only": selected_diff.get("actual_only"),
            "expected_only": selected_diff.get("expected_only"),
            "transition_summary": split.get("transition_summary"),
            "error_type": split.get("error_type"),
            "error": split.get("error"),
        }
        candidates.append(_strip_legacy_family_metadata(item))

    evidence: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_split_ids: set[str] = set()
    for item in candidates:
        group_id = str(item.get("rule_group_id") or "").strip()
        if group_id in seen_group_ids:
            continue
        evidence.append(item)
        seen_group_ids.add(group_id)
        split_id = str(item.get("split_id") or "").strip()
        identity = split_id or json.dumps(item, ensure_ascii=False, sort_keys=True)
        seen_split_ids.add(identity)
        if len(evidence) >= group_limit:
            return evidence

    for item in candidates:
        split_id = str(item.get("split_id") or "").strip()
        identity = split_id or json.dumps(item, ensure_ascii=False, sort_keys=True)
        if identity in seen_split_ids:
            continue
        evidence.append(item)
        seen_split_ids.add(identity)
        if len(evidence) >= group_limit:
            return evidence
    return evidence


def _repair_prompt_text(value: Any, *, limit: int = 900) -> str:
    if value in (None, "", [], {}):
        return ""
    cleaned = model_facing_diagnostics(_strip_legacy_family_metadata(value))
    if cleaned in (None, "", [], {}):
        return ""
    if isinstance(cleaned, str):
        text = cleaned.strip()
    else:
        text = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _append_repair_line(lines: list[str], label: str, value: Any, *, limit: int = 900) -> None:
    text = _repair_prompt_text(value, limit=limit)
    if text:
        lines.append(f"- {label}: {text}")


def _append_repair_block(lines: list[str], label: str, value: Any, *, limit: int = 1200) -> None:
    text = _repair_prompt_text(value, limit=limit)
    if not text:
        return
    lines.append(f"{label}:")
    lines.extend(f"  {line}" for line in text.splitlines())


def _completion_validation_group_summary(validation: dict[str, Any]) -> list[str]:
    groups = _strip_legacy_family_metadata(validation.get("failed_groups") or [])
    if not isinstance(groups, list):
        return []
    lines = ["Failed groups:"]
    shown = 0
    for group in groups[:5]:
        if not isinstance(group, dict):
            continue
        group_id = _repair_prompt_text(group.get("rule_group_id"), limit=120)
        if not group_id:
            continue
        parts = [group_id]
        for key in ("failed_split_count", "hard_split_count", "validated_split_count", "passed"):
            value = _repair_prompt_text(group.get(key), limit=120)
            if value:
                parts.append(f"{key}={value}")
        lines.append("- " + "; ".join(parts))
        shown += 1
    return lines if shown else []


def _completion_validation_failure_card(item: dict[str, Any], index: int, *, title: str) -> str:
    clean_item = _strip_legacy_family_metadata(item)
    if not isinstance(clean_item, dict):
        return ""
    label_bits = []
    for key in ("rule_group_id", "split_id", "variant_id", "mutation_type"):
        value = _repair_prompt_text(clean_item.get(key), limit=100)
        if value:
            label_bits.append(f"{key}={value}")
    heading = f"{title} {index}"
    if label_bits:
        heading += " [" + "; ".join(label_bits) + "]"
    lines = [heading]
    for key, label in (
        ("repair_hint", "Repair hint"),
        ("source_node_sql", "Source node SQL"),
        ("actual_node_sql", "Current node SQL"),
        ("expected_node_sql", "Expected node SQL"),
        ("actual_sql", "Current output"),
        ("expected_sql", "Expected output"),
        ("actual_only", "Actual only"),
        ("expected_only", "Expected only"),
        ("transition_summary", "Transition summary"),
        ("diff_summary", "Diff summary"),
        ("error_type", "Error type"),
        ("error", "Error"),
    ):
        value = clean_item.get(key)
        if key in {"actual_sql", "expected_sql", "source_node_sql", "actual_node_sql", "expected_node_sql"}:
            _append_repair_block(lines, label, value)
        else:
            _append_repair_line(lines, label, value)
    return "\n".join(lines)


def _completion_validation_failure_cards(validation: dict[str, Any]) -> list[str]:
    """Render deduplicated split/variant evidence cards for a repair prompt.

    Representative group evidence is shown first, followed by explicit repair
    target failures. Each source is bounded so malformed or large validation
    output cannot produce an unbounded retry message.
    """
    cards: list[str] = []
    evidence = _compact_failed_group_evidence(validation)
    for index, item in enumerate(evidence, start=1):
        card = _completion_validation_failure_card(item, index, title="Failed split evidence")
        if card:
            cards.append(card)

    repair_target = validation.get("repair_target") if isinstance(validation.get("repair_target"), dict) else {}
    failed_items = []
    for key, title in (
        ("failed_splits", "Failed split"),
        ("failed_variants", "Failed variant"),
    ):
        for item in repair_target.get(key) or []:
            if isinstance(item, dict):
                failed_items.append((title, item))
    for item in validation.get("variant_hard_failures") or []:
        if isinstance(item, dict):
            failed_items.append(("Failed variant", item))
    if not failed_items and any(validation.get(key) for key in ("actual_sql", "expected_sql", "diff_summary", "error")):
        failed_items.append(("Validation failure", validation))

    deduped_items = []
    seen_items: set[tuple[str, str, str, str, str]] = set()
    for title, item in failed_items:
        identity = (
            title,
            str(item.get("rule_group_id") or ""),
            str(item.get("split_id") or ""),
            str(item.get("variant_id") or ""),
            str(item.get("actual_sql") or item.get("expected_sql") or item.get("error") or ""),
        )
        if identity in seen_items:
            continue
        seen_items.add(identity)
        deduped_items.append((title, item))

    for index, (title, item) in enumerate(deduped_items[:5], start=1):
        card = _completion_validation_failure_card(item, index, title=title)
        if card:
            cards.append(card)
    return cards


def _format_completion_validation_failure_message(validation: dict[str, Any]) -> str:
    """Render the structured validation failure passed to the next agent turn.

    The message preserves stage, scope, repair target, failed-group summary, and
    bounded SQL/diff evidence while removing deprecated family metadata.
    """
    clean_validation = model_facing_diagnostics(_strip_legacy_family_metadata(validation))
    if not isinstance(clean_validation, dict):
        clean_validation = {}
    repair_target = clean_validation.get("repair_target")
    if not isinstance(repair_target, dict):
        repair_target = {}

    lines = ["[AutoDialect Completion Validation]", "Validation result: failed"]
    for key, label in (
        ("failed_stage", "Failed stage"),
        ("validation_scope", "Validation scope"),
        ("rule_group_id", "Primary rule group"),
        ("rule_group_ids", "Focused rule groups"),
        ("parser_role", "Parser role"),
        ("dialect", "Dialect"),
        ("repair_hint", "Repair hint"),
        ("error_type", "Error type"),
        ("error", "Error"),
    ):
        _append_repair_line(lines, label, clean_validation.get(key))

    target_lines: list[str] = []
    for key, label in (
        ("stage", "Stage"),
        ("rule_group_id", "Rule group"),
        ("rule_group_ids", "Rule groups"),
        ("reason", "Reason"),
        ("status", "Status"),
        ("instruction", "Instruction"),
    ):
        _append_repair_line(target_lines, label, repair_target.get(key))
    if target_lines:
        lines.extend(["", "Repair target:"])
        lines.extend(target_lines)

    group_lines = _completion_validation_group_summary(clean_validation)
    if group_lines:
        lines.extend(["", *group_lines])

    cards = _completion_validation_failure_cards(clean_validation)
    if cards:
        lines.extend(["", "Failed group evidence:"])
        for card in cards:
            lines.extend(["", card])
    elif clean_validation.get("diff_summary"):
        lines.extend(["", "Failure diff:"])
        _append_repair_line(lines, "Diff summary", clean_validation.get("diff_summary"))

    duration = _repair_prompt_text(clean_validation.get("duration_s"), limit=80)
    if duration:
        lines.extend(["", f"Validation duration: {duration}s"])
    return "\n".join(lines).strip()


def _completion_validation_repair_instruction(validation: dict[str, Any]) -> str:
    if validation.get("passed"):
        return COMPLETION_VALIDATION_PASSED_INSTRUCTION
    focus_group_ids = _normalize_group_ids(validation.get("focus_rule_group_ids"))
    failed_group_count = len(validation.get("failed_groups") or [])
    grouped_failure = len(focus_group_ids) > 1 or failed_group_count > 1
    if grouped_failure:
        return COMPLETION_VALIDATION_GROUP_REPAIR_INSTRUCTION
    return COMPLETION_VALIDATION_SINGLE_REPAIR_INSTRUCTION


def format_completion_validation_message(validation: dict[str, Any]) -> str:
    """Render the next repair turn without duplicating the initial task policy.

    Provider clients retain the initial task in message history, so the
    comparison policy injected by ``build_prompt`` remains visible here.
    """
    repair_instruction = _completion_validation_repair_instruction(validation)
    validation_message = _format_completion_validation_failure_message(validation)
    return COMPLETION_VALIDATION_RETRY_PROMPT.format(
        repair_instruction=repair_instruction,
        validation_message=validation_message,
    )


# 6. AUTODIALECT-agent-side audit recording.
def build_completion_validation_audit_section(
    validation: dict[str, Any] | None,
    *,
    previous_section: dict[str, Any] | None,
    spec: dict[str, Any] | None,
    completion_validation_path: str | None,
    failed_task_done_count: int,
    rejected: bool = False,
    completed: bool = False,
) -> dict[str, Any]:
    """Build the completion-validation section persisted in ``memory_audit``.

    Attempts accumulate from the previous section. A rejected ``task_done``
    stores only the latest failure metadata; a completed one stores the final
    result that rule_main may later reuse after checking scope and group ids.
    """
    previous_section = previous_section if isinstance(previous_section, dict) else {}
    spec = spec if isinstance(spec, dict) else {}
    validation_result = _strip_legacy_family_metadata(validation) if isinstance(validation, dict) else None
    validation_scope = (
        (validation_result or {}).get("validation_scope")
        or spec.get("validation_scope")
    )
    focus_rule_group_id = (
        (validation_result or {}).get("focus_rule_group_id")
        or spec.get("focus_rule_group_id")
    )
    focus_rule_group_ids = _normalize_group_ids(
        (validation_result or {}).get("focus_rule_group_ids")
        or spec.get("focus_rule_group_ids")
        or ([focus_rule_group_id] if focus_rule_group_id else [])
    )
    if not focus_rule_group_id and len(focus_rule_group_ids) == 1:
        focus_rule_group_id = focus_rule_group_ids[0]
    section = {
        "enabled": bool(completion_validation_path),
        "spec_path": completion_validation_path,
        "spec_id": spec.get("spec_id"),
        "attempts": int(previous_section.get("attempts") or 0) + 1,
        "failed_task_done_count": failed_task_done_count,
    }
    if rejected:
        section["rejected_current_task_done"] = True
        section["validation_scope"] = validation_scope
        section["focus_rule_group_id"] = focus_rule_group_id
        section["focus_rule_group_ids"] = focus_rule_group_ids
    if completed:
        section["completed_current_task_done"] = True
    for key in ("load_error_type", "load_error"):
        if spec.get(key) not in (None, [], {}):
            section[key] = spec.get(key)
    if completed:
        section["final_result"] = validation_result
    if rejected and validation_result:
        section["last_rejected_stage"] = validation_result.get("failed_stage")
        repair_target = validation_result.get("repair_target")
        if isinstance(repair_target, dict):
            section["last_rejected_reason"] = repair_target.get("reason")
    return compact_output_fields(section)
