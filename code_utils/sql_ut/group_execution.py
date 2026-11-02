"""Focused rule-group scheduling and batching helpers for AutoDialect.

1. Execution state: keep serializable per-group status, attempts, failures,
   and repair targets for rule_main.
2. Queue planning: derive initial and repair group queues from split manifests.
3. Batch shaping and metadata: merge compatible groups, cap hard units, and
   build focus arguments, handoff metadata, and compact stage records.
4. Public plan assembly and validation focus: package initial/repair execution
   plans and prefer the actively edited group as a repair target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from code_utils.common_tool import compact_output_fields
from code_utils.constants import (
    AUTODIALECT_GROUP_MERGE_ENABLED,
    AUTODIALECT_MAX_HARD_UNITS_PER_GROUP_BATCH,
)
from code_utils.sql_ut.sql_ut_splitter import split_is_reliable_rule_evidence

GROUP_MERGE_ENABLED = AUTODIALECT_GROUP_MERGE_ENABLED
MAX_HARD_UNITS_PER_GROUP_BATCH = AUTODIALECT_MAX_HARD_UNITS_PER_GROUP_BATCH

_FUNC_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CAST_TYPE_RE = re.compile(r"\bCAST\s*\(.*?\bAS\s+([A-Za-z][A-Za-z0-9_ ]*)\)", re.IGNORECASE)
_SINGLE_QUOTED_LITERAL_RE = re.compile(r"'((?:''|\\'|[^'])*)'")
_LITERAL_TOKEN_RE = re.compile(r"%[A-Za-z]|[A-Za-z]+[0-9]*|[0-9]+")
_GENERIC_TARGET_FUNCTIONS = {
    "ASSUMENOTNULL",
    "CAST",
    "NULLABLE",
}
_SQLGLOT_CACHE: dict[str, Any] = {"loaded": False, "sqlglot": None, "exp": None}


# 1. Execution state helpers.
@dataclass(frozen=True)
class GroupExecutionPlan:
    """Initial focused-group queue, state, batches, and audit stage record."""

    group_queue: list[dict[str, Any]]
    group_execution: dict[str, Any]
    group_batches: list[list[dict[str, Any]]]
    stage_record: dict[str, Any]


@dataclass(frozen=True)
class GroupRepairPlan:
    """Repair-only queue, batches, and optional audit stage record."""

    repair_queue: list[dict[str, Any]]
    repair_group_batches: list[list[dict[str, Any]]]
    stage_record: dict[str, Any] | None


def split_is_hard_assertion(split: dict[str, Any]) -> bool:
    """Return whether a split must pass for its rule group to pass."""
    return split_is_reliable_rule_evidence(split)


def new_group_execution(group_queue: list[dict[str, Any]]) -> dict[str, Any]:
    """Create serializable bookkeeping for group-level AUTODIALECT scheduling."""
    queued_groups = []
    for group in group_queue:
        group_id = group.get("rule_group_id")
        if not group_id:
            continue
        queued_groups.append(group_id)
    return {
        "queued_groups": queued_groups,
        "group_statuses": {},
    }


def _ensure_group_status(
    group_execution: dict[str, Any],
    group: dict[str, Any],
) -> dict[str, Any] | None:
    group_id = group.get("rule_group_id")
    if not group_id:
        return None
    statuses = group_execution.setdefault("group_statuses", {})
    return statuses.setdefault(
        group_id,
        {
            "rule_group_id": group_id,
            "status": "pending",
            "attempts": 0,
        },
    )


def ordered_group_statuses(group_execution: dict[str, Any]) -> list[dict[str, Any]]:
    """Return group status records in queue order, followed by extra records."""
    statuses = group_execution.get("group_statuses") or {}
    if not isinstance(statuses, dict):
        return []
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group_id in group_execution.get("queued_groups") or []:
        status = statuses.get(group_id)
        if isinstance(status, dict):
            if not status.get("rule_group_id"):
                status = {**status, "rule_group_id": group_id}
            ordered.append(status)
            seen.add(group_id)
    for group_id, status in statuses.items():
        if group_id in seen or not isinstance(status, dict):
            continue
        if not status.get("rule_group_id"):
            status = {**status, "rule_group_id": group_id}
        ordered.append(status)
    return ordered


def group_ids_by_status(group_execution: dict[str, Any], *target_statuses: str) -> list[str]:
    """Return queued group ids whose current status matches a target status."""
    targets = set(target_statuses)
    return [
        status.get("rule_group_id")
        for status in ordered_group_statuses(group_execution)
        if status.get("rule_group_id") and status.get("status") in targets
    ]


def attempted_group_ids(group_execution: dict[str, Any]) -> list[str]:
    """Return group ids that have at least one recorded execution attempt."""
    return [
        status.get("rule_group_id")
        for status in ordered_group_statuses(group_execution)
        if status.get("rule_group_id") and int(status.get("attempts") or 0) > 0
    ]


def failed_group_records(group_execution: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered status records for groups currently marked failed."""
    failed_groups = []
    for status in ordered_group_statuses(group_execution):
        if status.get("status") != "failed":
            continue
        failed_groups.append(
            {
                "rule_group_id": status.get("rule_group_id"),
                "attempts": int(status.get("attempts") or 0),
                "reason": status.get("failure_reason") or "focused_group_failed_after_agent_run",
                "repair_target": status.get("last_repair_target") or {},
            }
        )
    return failed_groups


def record_group_attempt(
    group_execution: dict[str, Any],
    group: dict[str, Any],
    *,
    stage: str,
    validation: dict[str, Any] | None = None,
    terminal_failure: bool = True,
) -> None:
    """Update one group's state after a focused agent call and validation.

    Passing group evidence marks the group passed. Hard split/variant failures
    either queue repair or become terminal based on ``terminal_failure``.
    Missing or unusable validation is fail-closed and records a repair target.
    The supplied serializable execution dictionary is mutated in place.
    """
    group_id = group.get("rule_group_id")
    if not group_id:
        return
    status = _ensure_group_status(group_execution, group)
    if status is None:
        return
    status["attempts"] = int(status.get("attempts") or 0) + 1
    status["last_stage"] = stage

    def clear_repair_state() -> None:
        status.pop("repair_reason", None)
        status.pop("retry_reason", None)

    def record_failed_group(repair_target: dict[str, Any]) -> None:
        reason = (
            "focused_group_failed_after_repair_run"
            if terminal_failure and str(stage).startswith("rule_group_repair")
            else "focused_group_failed_after_agent_run"
        )
        status["last_repair_target"] = repair_target
        if terminal_failure:
            clear_repair_state()
            status["status"] = "failed"
            status["failure_reason"] = reason
            return
        status["status"] = "repair"
        status["repair_reason"] = reason
        status.pop("failure_reason", None)

    if validation:
        repair_target = validation.get("repair_target") or {}
        variant_repair_target = _validation_variant_repair_target(validation, group_id)
        group_result = next(
            (
                item for item in validation.get("group_results", [])
                if item.get("rule_group_id") == group_id
            ),
            None,
        )
        status["last_failed_stage"] = validation.get("failed_stage")
        status["last_repair_target"] = repair_target
        if validation.get("passed") or (
            group_result and group_result.get("passed") and not variant_repair_target
        ):
            clear_repair_state()
            status["status"] = "passed"
            status.pop("last_repair_target", None)
            status.pop("failure_reason", None)
            return
        if variant_repair_target:
            record_failed_group(variant_repair_target)
            return
        if group_result and not group_result.get("passed"):
            group_repair_target = repair_target
            if repair_target.get("rule_group_id") != group_id:
                group_repair_target = {
                    "stage": validation.get("failed_stage") or "split_group",
                    "rule_group_id": group_id,
                    "reason": "focused_group_failed_after_agent_run",
                    "instruction": "Repair this failed group from the current grouped agent call.",
                }
            record_failed_group(group_repair_target)
            return
        if (
            repair_target.get("stage") in {"split_group", "variant"}
            and repair_target.get("rule_group_id") == group_id
        ):
            record_failed_group(repair_target)
            return
        if not validation.get("passed"):
            record_failed_group(
                {
                    "stage": validation.get("failed_stage") or "focused_group",
                    "rule_group_id": group_id,
                    "reason": "focused_group_validation_failed_without_group_detail",
                    "instruction": "Repair this focused group from the failed grouped validation.",
                }
            )
            return
    record_failed_group(
        {
            "stage": "focused_group",
            "rule_group_id": group_id,
            "reason": "focused_group_validation_missing",
            "instruction": "Focused validation did not return a usable result for this group.",
        }
    )


def _validation_variant_repair_target(
    validation: dict[str, Any],
    group_id: str | None,
) -> dict[str, Any] | None:
    if not group_id:
        return None
    repair_target = validation.get("repair_target") or {}
    if repair_target.get("stage") == "variant" and repair_target.get("rule_group_id") == group_id:
        return repair_target
    for variant in validation.get("variant_hard_failures") or []:
        if not isinstance(variant, dict) or variant.get("rule_group_id") != group_id:
            continue
        return {
            "stage": "variant",
            "rule_group_id": group_id,
            "reason": variant.get("reason") or "non_quote_only_variant_failed",
            "failed_variants": [variant],
            "instruction": "Repair this group's failed variant before considering the group passed.",
        }
    return None


def record_group_prevalidation(
    group_execution: dict[str, Any],
    group: dict[str, Any],
    *,
    stage: str,
    validation: dict[str, Any],
) -> bool:
    """Record focused prevalidation without incrementing agent attempts.

    A clean group with no hard variant failure is marked passed and returns
    ``True``. Otherwise it remains pending with the latest repair evidence and
    returns ``False`` so rule_main may launch the focused agent stage.
    """
    group_id = group.get("rule_group_id")
    if not group_id:
        return False
    status = _ensure_group_status(group_execution, group)
    if status is None:
        return False
    group_result = next(
        (
            item for item in validation.get("group_results", [])
            if item.get("rule_group_id") == group_id
        ),
        None,
    )
    variant_repair_target = _validation_variant_repair_target(validation, group_id)
    group_passed = bool(
        validation.get("passed")
        or (group_result and group_result.get("passed") and not variant_repair_target)
    )
    status["last_stage"] = stage
    status["last_failed_stage"] = validation.get("failed_stage")
    status["last_repair_target"] = variant_repair_target or validation.get("repair_target")
    if group_passed:
        status["status"] = "passed"
        status.pop("last_repair_target", None)
        status.pop("repair_reason", None)
        status.pop("retry_reason", None)
        status.pop("failure_reason", None)
        return True
    status["status"] = "pending"
    status.pop("repair_reason", None)
    status.pop("retry_reason", None)
    status.pop("failure_reason", None)
    return False


def _queued_rule_group_ids(group_queue: list[dict[str, Any]]) -> list[str]:
    return [
        group_id
        for group_id in (group.get("rule_group_id") for group in group_queue)
        if group_id
    ]


def build_repair_group_queue(
    group_queue: list[dict[str, Any]],
    group_execution: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return queued groups that failed initial validation and need one repair run."""
    group_by_id = {
        group.get("rule_group_id"): group
        for group in group_queue
        if group.get("rule_group_id")
    }
    repair_groups = []
    for group_id in group_ids_by_status(group_execution, "repair"):
        group = group_by_id.get(group_id)
        if group:
            repair_groups.append(group)
    return repair_groups


def all_queued_groups_passed(
    group_execution: dict[str, Any],
    group_queue: list[dict[str, Any]],
) -> bool:
    """Return whether every queued group has reached the passed state."""
    group_ids = _queued_rule_group_ids(group_queue)
    if not group_ids:
        return True
    passed = set(group_ids_by_status(group_execution, "passed"))
    return all(group_id in passed for group_id in group_ids)


def group_execution_pending_validation(
    group_execution: dict[str, Any],
    group_queue: list[dict[str, Any]],
) -> dict[str, Any]:
    """Block parent validation until every queued hard group has passed.

    The returned validation-shaped failure lists pending and failed group ids
    and reuses the first available repair target, allowing the normal repair
    reporting path to explain why full parent validation was not run.
    """
    group_ids = _queued_rule_group_ids(group_queue)
    statuses = group_execution.get("group_statuses") or {}
    passed = set(group_ids_by_status(group_execution, "passed"))
    failed = failed_group_records(group_execution)
    pending_ids = [group_id for group_id in group_ids if group_id not in passed]
    first_id = pending_ids[0] if pending_ids else None
    first_status = statuses.get(first_id, {}) if first_id else {}
    repair_target = first_status.get("last_repair_target") or {}
    if not repair_target and first_id:
        repair_target = {
            "stage": "split_group",
            "rule_group_id": first_id,
            "reason": "queued_group_not_passed",
            "instruction": "Finish focused validation for this rule group before running parent SQL validation.",
        }
    return {
        "passed": False,
        "failed_stage": "group_execution",
        "validation_scope": "group_queue",
        "pending_group_ids": pending_ids,
        "failed_groups": failed,
        "repair_target": repair_target,
        "repair_hint": "Parent SQL validation is delayed until every queued rule group passes focused split and variant validation.",
    }


# 2. Queue planning helpers.
def _manifest_group_for_queue(
    split_manifest: dict[str, Any],
    group: dict[str, Any],
) -> dict[str, Any] | None:
    """Materialize a queue item for a manifest group with hard assertion splits."""
    rule_group_id = group.get("rule_group_id")
    if not rule_group_id:
        return None
    split_ids = set(group.get("split_ids") or [])
    group_splits = [
        split for split in split_manifest.get("splits", [])
        if split.get("split_id") in split_ids and split_is_hard_assertion(split)
    ]
    if not group_splits:
        return None
    return {
        "rule_group_id": rule_group_id,
        "hard_unit_count": len(group_splits),
        "total_units": len(group_splits),
    }


def build_initial_group_queue(
    split_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Queue every manifest group that owns at least one hard assertion split."""
    indexed_groups = []
    for index, group in enumerate(split_manifest.get("groups", [])):
        materialized = _manifest_group_for_queue(
            split_manifest,
            group,
        )
        if materialized:
            indexed_groups.append((index, materialized))
    indexed_groups.sort(
        key=lambda item: (
            item[1].get("total_units") or 0,
            -item[0],
        ),
        reverse=True,
    )
    return [group for _index, group in indexed_groups]


def split_manifest_stage_record(
    split_manifest: dict[str, Any],
    *,
    duration_s: float,
) -> dict[str, Any]:
    """Return the pipeline stage record for one UT split manifest."""
    return {
        "stage": "ut_split",
        "status": split_manifest.get("status"),
        "duration_s": duration_s,
        "groups": len(split_manifest.get("groups", [])),
        "hard_validation_uts": sum(
            1 for split in split_manifest.get("splits", [])
            if split_is_hard_assertion(split)
        ),
    }


# 3. Batch shaping, merge-key, and metadata helpers.
def _manifest_group_by_id(split_manifest: dict[str, Any], group_id: str | None) -> dict[str, Any]:
    if not group_id:
        return {}
    return next(
        (
            item for item in split_manifest.get("groups", [])
            if item.get("rule_group_id") == group_id
        ),
        {},
    )


def _normalized_merge_token(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text or None


def _sqlglot_modules() -> tuple[Any | None, Any | None]:
    if not _SQLGLOT_CACHE["loaded"]:
        try:
            import sqlglot  # type: ignore
            from sqlglot import exp  # type: ignore

            _SQLGLOT_CACHE.update({"loaded": True, "sqlglot": sqlglot, "exp": exp})
        except Exception:
            _SQLGLOT_CACHE.update({"loaded": True, "sqlglot": None, "exp": None})
    return _SQLGLOT_CACHE.get("sqlglot"), _SQLGLOT_CACHE.get("exp")


def _signature_values(signature: str | None, field: str) -> list[str]:
    values: list[str] = []
    prefix = f"{field}="
    for line in str(signature or "").splitlines():
        text = line.strip()
        if text.startswith("identity=hint="):
            text = text.removeprefix("identity=hint=")
        elif text.startswith("hint="):
            text = text.removeprefix("hint=")
        if not text.startswith(prefix):
            continue
        for raw_value in text.removeprefix(prefix).split("|"):
            value = _normalized_merge_token(raw_value)
            if value and value not in values:
                values.append(value)
    return values


def _signature_text_field(signature: str | None, field: str) -> str | None:
    prefix = f"{field}="
    for line in str(signature or "").splitlines():
        text = line.strip()
        if text.startswith(prefix):
            return text.removeprefix(prefix).strip() or None
    return None


def _group_rule_signature(group: dict[str, Any], split_manifest: dict[str, Any]) -> str | None:
    group_id = group.get("rule_group_id")
    manifest_group = _manifest_group_by_id(split_manifest, group_id)
    hard_splits = _group_hard_splits(group, split_manifest)
    for split in hard_splits:
        if split.get("rule_signature"):
            return split.get("rule_signature")
    split_ids = set(manifest_group.get("split_ids") or group.get("split_ids") or [])
    present_splits = [
        split
        for split in split_manifest.get("splits", [])
        if split.get("split_id") in split_ids
    ]
    # Preserve group-only legacy inputs, but never fall back to a manifest
    # signature when present split records are missing or unreliable.
    if not present_splits:
        return group.get("rule_signature") or manifest_group.get("rule_signature")
    present_split_ids = {split.get("split_id") for split in present_splits}
    if present_split_ids == split_ids and len(hard_splits) == len(present_splits):
        return group.get("rule_signature") or manifest_group.get("rule_signature")
    return None


def _group_hard_splits(group: dict[str, Any], split_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    group_id = group.get("rule_group_id")
    manifest_group = _manifest_group_by_id(split_manifest, group_id)
    split_ids = set(manifest_group.get("split_ids") or group.get("split_ids") or [])
    if not split_ids:
        return []
    return [
        split
        for split in split_manifest.get("splits", [])
        if split.get("split_id") in split_ids and split_is_hard_assertion(split)
    ]


def group_hard_unit_count(group: dict[str, Any], split_manifest: dict[str, Any]) -> int:
    manifest_group = _manifest_group_by_id(split_manifest, group.get("rule_group_id"))
    split_ids = set(manifest_group.get("split_ids") or group.get("split_ids") or [])
    present_splits = [
        split
        for split in split_manifest.get("splits", [])
        if split.get("split_id") in split_ids
    ]
    if present_splits:
        return len(_group_hard_splits(group, split_manifest))

    # Preserve group-only legacy callers when no split records are available.
    for key in ("hard_unit_count", "total_units", "shown_units"):
        try:
            count = int(group.get(key) or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count

    hard_splits = _group_hard_splits(group, split_manifest)
    if hard_splits:
        return len(hard_splits)

    return max(1, len(split_ids))


def batch_hard_unit_count(batch: list[dict[str, Any]], split_manifest: dict[str, Any]) -> int:
    return sum(group_hard_unit_count(group, split_manifest) for group in batch)


def _cap_batch_by_hard_units(
    batch: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    *,
    max_hard_units_per_batch: int,
) -> list[list[dict[str, Any]]]:
    max_units = max(0, int(max_hard_units_per_batch or 0))
    if not max_units:
        return [batch]

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_units = 0
    for group in batch:
        group_units = group_hard_unit_count(group, split_manifest)
        if current and current_units + group_units > max_units:
            batches.append(current)
            current = []
            current_units = 0
        current.append(group)
        current_units += group_units
    if current:
        batches.append(current)
    return batches


def _group_expression_pair(
    group: dict[str, Any],
    split_manifest: dict[str, Any],
) -> tuple[str | None, str | None]:
    for split in _group_hard_splits(group, split_manifest):
        source_sql = split.get("source_expr_sql") or split.get("source_sql")
        expected_sql = split.get("expected_expr_sql") or split.get("expected_sql")
        if source_sql and expected_sql:
            return str(source_sql), str(expected_sql)
    signature = _group_rule_signature(group, split_manifest)
    return _signature_text_field(signature, "source"), _signature_text_field(signature, "expected")


def _parse_expression(expression_sql: str | None) -> Any | None:
    text = str(expression_sql or "").strip()
    if not text:
        return None
    sqlglot, _sql_exp = _sqlglot_modules()
    if sqlglot is None:
        return None
    for candidate in (text, f"SELECT {text}"):
        try:
            return sqlglot.parse_one(candidate)
        except Exception:
            continue
    return None


def _root_expression(node: Any | None) -> Any | None:
    if node is None:
        return None
    node_name = node.__class__.__name__
    if node_name == "Select":
        expressions = list(node.args.get("expressions") or [])
        return _root_expression(expressions[0]) if expressions else node
    if node_name == "Alias":
        return _root_expression(node.args.get("this"))
    return node


def _function_node_name(node: Any | None) -> str | None:
    if node is None:
        return None
    _sqlglot, sql_exp = _sqlglot_modules()
    if sql_exp is None:
        return None
    try:
        if isinstance(node, sql_exp.Anonymous):
            return _normalized_merge_token(node.args.get("this"))
        if isinstance(node, sql_exp.Func):
            sql_name = node.sql_name()
            name = None if sql_name == "ANONYMOUS" else sql_name
            return _normalized_merge_token(name or node.args.get("this"))
    except Exception:
        return None
    return None


def _expression_root_function_name(expression_sql: str | None) -> str | None:
    node = _root_expression(_parse_expression(expression_sql))
    function_name = _function_node_name(node)
    if function_name:
        return function_name
    match = _FUNC_CALL_RE.search(str(expression_sql or ""))
    return _normalized_merge_token(match.group(1)) if match else None


def _expression_function_names(expression_sql: str | None) -> set[str]:
    names: set[str] = set()
    node = _parse_expression(expression_sql)
    if node is not None:
        try:
            for child in node.walk():
                function_name = _function_node_name(child)
                if function_name:
                    names.add(function_name)
        except Exception:
            pass
    if names:
        return names
    return {
        value
        for value in (_normalized_merge_token(match.group(1)) for match in _FUNC_CALL_RE.finditer(str(expression_sql or "")))
        if value
    }


def _datatype_node_name(node: Any | None) -> str | None:
    if node is None:
        return None
    _sqlglot, sql_exp = _sqlglot_modules()
    if sql_exp is None:
        return None
    try:
        if not isinstance(node, sql_exp.DataType):
            return None
        kind = node.args.get("kind")
        if kind:
            return _normalized_merge_token(kind)
        value = node.args.get("this")
        if hasattr(value, "value"):
            return _normalized_merge_token(value.value)
        if hasattr(value, "name"):
            return _normalized_merge_token(value.name)
        return _normalized_merge_token(node.sql())
    except Exception:
        return None


def _expression_type_names(expression_sql: str | None) -> list[str]:
    names: list[str] = []
    node = _parse_expression(expression_sql)
    if node is not None:
        try:
            for child in node.walk():
                type_name = _datatype_node_name(child)
                if type_name and type_name not in names:
                    names.append(type_name)
        except Exception:
            pass
    if names:
        return names
    for match in _CAST_TYPE_RE.finditer(str(expression_sql or "")):
        type_name = _normalized_merge_token(match.group(1))
        if type_name and type_name not in names:
            names.append(type_name)
    return names


def _split_expression_triplet(split: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    selected_diff = split.get("selected_target_diff")
    if not isinstance(selected_diff, dict):
        selected_diff = {}
    source_expr = (
        selected_diff.get("source_node_sql")
        or split.get("source_expr_sql")
        or split.get("source_sql")
    )
    actual_expr = (
        selected_diff.get("actual_node_sql")
        or split.get("actual_expr_sql")
        or split.get("actual_sql")
    )
    expected_expr = (
        selected_diff.get("expected_node_sql")
        or split.get("expected_expr_sql")
        or split.get("expected_sql")
    )
    return (
        str(source_expr) if source_expr else None,
        str(actual_expr) if actual_expr else None,
        str(expected_expr) if expected_expr else None,
    )


def _literal_tokens(text: str | None) -> list[str]:
    tokens: list[str] = []
    for match in _LITERAL_TOKEN_RE.finditer(str(text or "")):
        raw = match.group(0)
        if raw.isdigit():
            token = "<NUM>"
        elif raw.startswith("%"):
            token = raw
        else:
            token = _normalized_merge_token(raw)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _quoted_literal_token_sets(expression_sql: str | None) -> list[list[str]]:
    token_sets: list[list[str]] = []
    for match in _SINGLE_QUOTED_LITERAL_RE.finditer(str(expression_sql or "")):
        literal = match.group(1).replace("''", "'").replace("\\'", "'")
        tokens = _literal_tokens(literal)
        if tokens:
            token_sets.append(tokens)
    return token_sets


def _ordered_token_delta(left: list[str], right: list[str]) -> list[str]:
    right_set = set(right)
    return [token for token in left if token not in right_set]


def _literal_delta_signature(actual_expr: str | None, expected_expr: str | None) -> str | None:
    actual_literals = _quoted_literal_token_sets(actual_expr)
    expected_literals = _quoted_literal_token_sets(expected_expr)
    deltas: list[str] = []
    for index in range(max(len(actual_literals), len(expected_literals))):
        actual_tokens = actual_literals[index] if index < len(actual_literals) else []
        expected_tokens = expected_literals[index] if index < len(expected_literals) else []
        actual_only = _ordered_token_delta(actual_tokens, expected_tokens)
        expected_only = _ordered_token_delta(expected_tokens, actual_tokens)
        if not actual_only and not expected_only:
            continue
        deltas.append(
            "literal_delta:"
            f"a={','.join(actual_only) or '<none>'}"
            f"=>e={','.join(expected_only) or '<none>'}"
        )
    return "|".join(deltas[:3]) or None


def _nested_function_signature(
    source_expr: str | None,
    actual_expr: str | None,
    expected_expr: str | None,
) -> str | None:
    source_root = _expression_root_function_name(source_expr)
    actual_root = _expression_root_function_name(actual_expr)
    expected_root = _expression_root_function_name(expected_expr)
    ignored = set(_GENERIC_TARGET_FUNCTIONS)
    nested = {
        "s": sorted(_expression_function_names(source_expr) - {source_root} - ignored),
        "a": sorted(_expression_function_names(actual_expr) - {actual_root} - ignored),
        "e": sorted(_expression_function_names(expected_expr) - {expected_root} - ignored),
    }
    if not any(nested.values()):
        return None
    parts = [
        f"{side}={','.join(values)}"
        for side, values in nested.items()
        if values
    ]
    return "nested_functions:" + "|".join(parts) if parts else None


def _group_refinement_key(group: dict[str, Any], split_manifest: dict[str, Any]) -> str | None:
    """Return an intra-merge-bucket key based on generic expression diff shape."""
    signatures: list[str] = []
    for split in _group_hard_splits(group, split_manifest):
        source_expr, actual_expr, expected_expr = _split_expression_triplet(split)
        for signature in (
            _literal_delta_signature(actual_expr, expected_expr),
            _nested_function_signature(source_expr, actual_expr, expected_expr),
        ):
            if signature and signature not in signatures:
                signatures.append(signature)
        if len(signatures) >= 4:
            break
    return ";".join(signatures) or None


def _merge_key_allows_refinement(merge_key: str | None) -> bool:
    return bool(
        merge_key
        and (
            merge_key.startswith("function_transform:")
            or merge_key.startswith("target_functions:")
        )
    )


def group_merge_key(group: dict[str, Any], split_manifest: dict[str, Any]) -> str | None:
    """Return the semantic key used to co-schedule related rule groups."""
    signature = _group_rule_signature(group, split_manifest)
    operators = _signature_values(signature, "operator")
    non_quote_operators = sorted({operator for operator in operators if operator != "IDENTIFIER_QUOTE"})
    if non_quote_operators:
        return f"operator:{'|'.join(non_quote_operators)}"

    source_expr, expected_expr = _group_expression_pair(group, split_manifest)
    source_root = _expression_root_function_name(source_expr)
    expected_root = _expression_root_function_name(expected_expr)
    if source_root and expected_root and source_root != expected_root:
        return f"function_transform:{source_root}->{expected_root}"

    source_types = _expression_type_names(source_expr)
    expected_types = _expression_type_names(expected_expr)
    if source_types and expected_types and source_types[0] != expected_types[0]:
        return f"type_transform:{source_types[0]}->{expected_types[0]}"

    source_functions = _expression_function_names(source_expr)
    expected_functions = _expression_function_names(expected_expr)
    target_functions = sorted(
        function
        for function in (expected_functions - source_functions)
        if function not in _GENERIC_TARGET_FUNCTIONS
    )
    if target_functions:
        return f"target_functions:{'|'.join(target_functions)}"

    if "IDENTIFIER_QUOTE" in operators:
        return "operator:IDENTIFIER_QUOTE"
    return None


def build_group_batches(
    group_queue: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    *,
    merge_enabled: bool = GROUP_MERGE_ENABLED,
    max_hard_units_per_batch: int = MAX_HARD_UNITS_PER_GROUP_BATCH,
) -> list[list[dict[str, Any]]]:
    """Batch compatible queued groups without weakening validation boundaries.

    Groups first merge by semantic rule key, then selected merge families are
    refined by expression details. Every resulting batch is capped by hard-unit
    count; groups lacking a safe merge key remain isolated and order is stable.
    """
    group_queue = [
        group
        for group in group_queue
        if group_hard_unit_count(group, split_manifest) > 0
    ]
    if not group_queue:
        return []

    def capped(batch: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return _cap_batch_by_hard_units(
            batch,
            split_manifest,
            max_hard_units_per_batch=max_hard_units_per_batch,
        )

    if not merge_enabled:
        return [batch for group in group_queue for batch in capped([group])]

    merged_batches: list[list[dict[str, Any]]] = []
    keyed_batches: dict[str, list[dict[str, Any]]] = {}
    for group in group_queue:
        merge_key = group_merge_key(group, split_manifest)
        if not merge_key:
            merged_batches.append([group])
            continue
        if merge_key not in keyed_batches:
            keyed_batches[merge_key] = []
            merged_batches.append(keyed_batches[merge_key])
        keyed_batches[merge_key].append(group)

    def refined(batch: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        merge_key = batch_merge_key(batch, split_manifest)
        if not _merge_key_allows_refinement(merge_key):
            return [batch]
        refined_batches: list[list[dict[str, Any]]] = []
        keyed_refined_batches: dict[str, list[dict[str, Any]]] = {}
        for group in batch:
            refinement_key = _group_refinement_key(group, split_manifest) or "refine:none"
            if refinement_key not in keyed_refined_batches:
                keyed_refined_batches[refinement_key] = []
                refined_batches.append(keyed_refined_batches[refinement_key])
            keyed_refined_batches[refinement_key].append(group)
        return refined_batches if len(refined_batches) > 1 else [batch]

    batches: list[list[dict[str, Any]]] = []
    for batch in merged_batches:
        for refined_batch in refined(batch):
            batches.extend(capped(refined_batch))
    return batches


def batch_merge_key(batch: list[dict[str, Any]], split_manifest: dict[str, Any]) -> str | None:
    if not batch:
        return None
    return group_merge_key(batch[0], split_manifest)


def batch_refinement_key(batch: list[dict[str, Any]], split_manifest: dict[str, Any]) -> str | None:
    if not batch:
        return None
    if not _merge_key_allows_refinement(batch_merge_key(batch, split_manifest)):
        return None
    return _group_refinement_key(batch[0], split_manifest)


def batch_group_ids(batch: list[dict[str, Any]]) -> list[str]:
    return [
        group_id
        for group_id in (group.get("rule_group_id") for group in batch)
        if group_id
    ]


def focus_group_arg(group_ids: list[str]) -> str | list[str] | None:
    if not group_ids:
        return None
    if len(group_ids) == 1:
        return group_ids[0]
    return group_ids


def handoff_group_for_batch(batch: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not batch:
        return None
    group_ids = batch_group_ids(batch)
    if len(batch) == 1:
        return batch[0]
    return {
        "rule_group_id": ",".join(group_ids),
        "rule_group_ids": group_ids,
        "batch_size": len(batch),
    }


def describe_group_batch(
    batch: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    *,
    batch_index: int | None = None,
) -> dict[str, Any]:
    """Return the batch context needed by prompts, handoff, and stage records."""
    group_ids = batch_group_ids(batch)
    description = {
        "rule_group_id": group_ids[0] if len(group_ids) == 1 else None,
        "rule_group_ids": group_ids,
        "focus_rule_group_arg": focus_group_arg(group_ids),
        "handoff_group": handoff_group_for_batch(batch),
        "label": ",".join(group_ids),
        "batch_size": len(batch),
        "hard_unit_count": batch_hard_unit_count(batch, split_manifest),
        "merge_key": batch_merge_key(batch, split_manifest),
        "refinement_key": batch_refinement_key(batch, split_manifest),
    }
    if batch_index is not None:
        description["batch_index"] = batch_index
    return description


def _batch_stage_records(
    group_batches: list[list[dict[str, Any]]],
    split_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for batch_index, batch in enumerate(group_batches):
        batch_context = describe_group_batch(batch, split_manifest, batch_index=batch_index)
        records.append(
            {
                "batch_index": batch_index,
                "rule_group_ids": batch_context["rule_group_ids"],
                "batch_size": batch_context["batch_size"],
                "hard_unit_count": batch_context["hard_unit_count"],
                "merge_key": batch_context["merge_key"],
                "refinement_key": batch_context["refinement_key"],
            }
        )
    return records


def _queue_stage_record(
    *,
    stage: str,
    group_queue: list[dict[str, Any]],
    group_batches: list[list[dict[str, Any]]],
    split_manifest: dict[str, Any],
    include_groups: bool,
) -> dict[str, Any]:
    record = {
        "stage": stage,
        "queued_group_count": len(group_queue),
        "group_batch_count": len(group_batches),
        "max_hard_units_per_batch": (
            MAX_HARD_UNITS_PER_GROUP_BATCH
            if MAX_HARD_UNITS_PER_GROUP_BATCH > 0 else None
        ),
        "group_batches": _batch_stage_records(group_batches, split_manifest),
    }
    if include_groups:
        record.update(
            {
                "groups": [
                    {
                        "rule_group_id": group.get("rule_group_id"),
                        "hard_unit_count": group_hard_unit_count(group, split_manifest),
                        "total_units": group.get("total_units"),
                    }
                    for group in group_queue
                ],
            }
        )
    return compact_output_fields(record)


# 4. Public plan assembly and validation-focus entrypoints.
def prepare_group_execution_plan(split_manifest: dict[str, Any]) -> GroupExecutionPlan:
    """Build the complete initial focused-group execution plan.

    Only manifest groups with hard assertions enter the queue. The returned
    immutable container bundles serializable status state, compatible batches,
    and the audit stage record consumed by rule_main.
    """
    group_queue = build_initial_group_queue(split_manifest)
    group_execution = new_group_execution(group_queue)
    group_batches = build_group_batches(group_queue, split_manifest)
    return GroupExecutionPlan(
        group_queue=group_queue,
        group_execution=group_execution,
        group_batches=group_batches,
        stage_record=_queue_stage_record(
            stage="group_queue",
            group_queue=group_queue,
            group_batches=group_batches,
            split_manifest=split_manifest,
            include_groups=True,
        ),
    )


def prepare_group_repair_plan(
    group_queue: list[dict[str, Any]],
    group_execution: dict[str, Any],
    split_manifest: dict[str, Any],
) -> GroupRepairPlan:
    """Build a repair-only plan from groups currently marked for repair.

    Repair groups retain initial queue order and use the same compatibility and
    hard-unit batching rules. The stage record is ``None`` when no repair work
    remains.
    """
    repair_queue = build_repair_group_queue(group_queue, group_execution)
    repair_group_batches = build_group_batches(repair_queue, split_manifest)
    stage_record = None
    if repair_group_batches:
        stage_record = _queue_stage_record(
            stage="group_repair_queue",
            group_queue=repair_queue,
            group_batches=repair_group_batches,
            split_manifest=split_manifest,
            include_groups=False,
        )
    return GroupRepairPlan(
        repair_queue=repair_queue,
        repair_group_batches=repair_group_batches,
        stage_record=stage_record,
    )


def focus_validation_on_group(
    validation: dict[str, Any],
    group_id: str | None,
    *,
    instruction: str = "Repair only the focused group from the current group agent call.",
    include_confidence: bool = False,
) -> dict[str, Any]:
    """Retarget full-validation failure evidence to the actively edited group.

    The result is unchanged unless that group has a failed hard split. When it
    does, a focused split-group repair target is attached while retaining the
    complete validation evidence for audit and outer fallback handling.
    """
    if not validation or not group_id or validation.get("passed"):
        return validation
    failed_items = [
        item for item in validation.get("split_results", [])
        if item.get("rule_group_id") == group_id
        and item.get("hard_assertion")
        and not item.get("passed")
    ]
    if not failed_items:
        return validation
    repair_target = {
        "stage": "split_group",
        "rule_group_id": group_id,
        "reason": "focused_group_hard_split_failed",
        "failed_splits": [
            dict(
                {
                    "split_id": item.get("split_id"),
                    "rule_group_id": item.get("rule_group_id"),
                    "actual_sql": item.get("actual_sql"),
                    "expected_sql": item.get("expected_sql"),
                    "diff_summary": item.get("diff_summary", []),
                    "quote_insensitive_passed": item.get("quote_insensitive_passed"),
                    "error_type": item.get("error_type"),
                    "error": item.get("error"),
                },
                **({"confidence": item.get("confidence")} if include_confidence else {}),
            )
            for item in failed_items[:5]
        ],
        "instruction": instruction,
    }
    validation.update(
        {
            "failed_stage": "split_group",
            "rule_group_id": group_id,
            "actual_sql": failed_items[0].get("actual_sql"),
            "expected_sql": failed_items[0].get("expected_sql"),
            "diff_summary": failed_items[0].get("diff_summary", []),
            "repair_target": repair_target,
            "focused_repair_target": True,
        }
    )
    return validation


__all__ = [
    "GROUP_MERGE_ENABLED",
    "GroupExecutionPlan",
    "GroupRepairPlan",
    "MAX_HARD_UNITS_PER_GROUP_BATCH",
    "all_queued_groups_passed",
    "attempted_group_ids",
    "batch_group_ids",
    "batch_hard_unit_count",
    "batch_merge_key",
    "batch_refinement_key",
    "build_group_batches",
    "build_initial_group_queue",
    "build_repair_group_queue",
    "describe_group_batch",
    "failed_group_records",
    "focus_group_arg",
    "focus_validation_on_group",
    "group_execution_pending_validation",
    "group_hard_unit_count",
    "group_ids_by_status",
    "group_merge_key",
    "handoff_group_for_batch",
    "new_group_execution",
    "ordered_group_statuses",
    "prepare_group_execution_plan",
    "prepare_group_repair_plan",
    "record_group_attempt",
    "record_group_prevalidation",
    "split_is_hard_assertion",
    "split_manifest_stage_record",
]
