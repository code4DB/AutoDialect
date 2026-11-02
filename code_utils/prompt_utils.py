"""Prompt assembly helpers for AutoDialect case execution.

This module owns the prompt contract between rule_main and the active agent.
Functions are ordered by prompt priority and then by responsibility:
1. Public prompt entrypoints and final template assembly.
2. Case context and parser-gate rendering.
3. Split diagnostics and overview prompt rendering.
4. Focused unit and variant prompt rendering.
5. Memory query construction and payload compaction.
6. Prompt-safe text and excerpt helpers.
"""

from __future__ import annotations

import copy
import json
import re
from textwrap import dedent

from code_utils.common_tool import (
    is_legacy_family_key as _is_legacy_family_key,
)
from code_utils.common_tool import model_facing_diagnostics
from code_utils.common_tool import (
    scrub_legacy_family_text as _scrub_legacy_family_text,
)
from code_utils.common_tool import (
    strip_legacy_family_metadata as _strip_legacy_family_metadata,
)
from code_utils.memory.sql_identity import sql_pair_hash
from code_utils.sql_ut.sql_ut_workset import (
    MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT,
    MAX_PROMPT_RULE_GROUPS,
    build_rule_workset,
)
from code_utils.sql_ut.sql_ut_splitter import split_is_reliable_rule_evidence
from prompt.agent_prompt import (
    INITIAL_SIMPLE_PROMPT,
    FOCUSED_SPLIT_EMPTY_SCOPE_NOTE,
    FOCUSED_SPLIT_OVERSIZED_NOTE,
    FOCUSED_SPLIT_WORK_ORDER_HEADER,
    GROUP_SIMPLE_PROMPT,
    MEMORY_QUERY_SECTION_PROMPT,
    PARSER_STAGE_REPAIR_PROMPT,
    RULE_EXAMPLES,
    RULE_TASK_CONTRACT,
    SIMPLE_PROMPT,
    SQL_COMPARISON_POLICY,
    STAGE_HANDOFF_PROMPT,
    SUMMARY_PROMPT,
    TSQL_DIALECT_CONTEXT,
)

PROMPT_VARIANT_EXAMPLE_LIMIT_PER_UNIT = 1


# Public prompt entrypoints and final template assembly.
def build_prompt(
    info,
    is_summary,
    *,
    parser_gate=None,
    target_parser_gate=None,
    split_manifest=None,
    prompt_kind="rule",
    focus_rule_group_id=None,
    max_groups=None,
    max_diagnostic_hints_per_category=None,
    stage_handoff=None,
    include_memory_query=False,
):
    """Build the final prompt string for either summary mode or patch mode."""
    if is_summary:
        return format_summary_prompt(info)
    if prompt_kind == "parser_repair":
        task = format_parser_repair_prompt(info, parser_gate)
        task = (
            _append_memory_query(task, info, stage_handoff=stage_handoff)
            if include_memory_query
            else task
        )
        return _with_sql_comparison_policy(task)
    if prompt_kind == "target_parser_repair":
        task = format_target_parser_repair_prompt(info, target_parser_gate)
        task = (
            _append_memory_query(task, info, stage_handoff=stage_handoff)
            if include_memory_query
            else task
        )
        return _with_sql_comparison_policy(task)
    if parser_gate is not None or target_parser_gate is not None or split_manifest is not None:
        task = format_task_prompt(
            info,
            parser_gate=parser_gate,
            target_parser_gate=target_parser_gate,
            split_manifest=split_manifest,
            focus_rule_group_id=focus_rule_group_id,
            max_groups=max_groups,
            max_diagnostic_hints_per_category=max_diagnostic_hints_per_category,
            stage_handoff=stage_handoff,
            include_memory_query=include_memory_query,
        )
        return _with_sql_comparison_policy(task)
    task = format_initial_task_prompt(info)
    if include_memory_query:
        task = _append_memory_query(task, info, stage_handoff=stage_handoff)
    return _with_sql_comparison_policy(task)


def _with_sql_comparison_policy(task):
    """Prepend the benchmark comparison contract exactly once per prompt."""
    policy = SQL_COMPARISON_POLICY.strip()
    task = str(task or "").lstrip()
    if policy in task:
        return task
    return f"{policy}\n\n{task}"


def format_task_prompt(
    info,
    *,
    parser_gate=None,
    target_parser_gate=None,
    split_manifest=None,
    focus_rule_group_id=None,
    max_groups=None,
    max_diagnostic_hints_per_category=None,
    stage_handoff=None,
    include_memory_query=False,
):
    """Format the patch-generation prompt from structured case information."""
    stage_handoff = _sanitize_stage_handoff_for_split_manifest(stage_handoff, split_manifest)
    focus_group_ids = _normalize_focus_group_ids(focus_rule_group_id)
    split_summary = _render_split_manifest(
        split_manifest,
        focus_rule_group_id=focus_rule_group_id,
        max_groups=max_groups,
        max_diagnostic_hints_per_category=max_diagnostic_hints_per_category,
    )
    template = GROUP_SIMPLE_PROMPT if focus_group_ids else SIMPLE_PROMPT
    compact_parser_context = bool(focus_group_ids)
    task = template.format(
        src_db=info["src_db"], tgt_db=info["tgt_db"],
        src_sql=info["src_sql"], tgt_sql=info["tgt_sql"],
        path=_target_path(info),
        dialect_note=_dialect_note(info),
        task_contract=_rule_task_contract(),
        parser_gate=_render_parser_gate(parser_gate, compact_sql=compact_parser_context),
        target_parser_gate=_render_parser_gate(target_parser_gate, compact_sql=compact_parser_context),
        split_summary=split_summary,
        rule_examples=RULE_EXAMPLES.strip(),
    )
    task = dedent(task)
    if include_memory_query:
        task = _append_memory_query(
            task,
            info,
            split_manifest=split_manifest,
            focus_rule_group_id=focus_rule_group_id,
            max_groups=max_groups,
            max_diagnostic_hints_per_category=max_diagnostic_hints_per_category,
            stage_handoff=stage_handoff,
        )

    return _prepend_stage_handoff(task, stage_handoff)


def format_initial_task_prompt(info):
    """Format the initial compact prompt before staged validation begins."""
    task = INITIAL_SIMPLE_PROMPT.format(
        src_db=info["src_db"],
        tgt_db=info["tgt_db"],
        src_sql_json=json.dumps(info["src_sql"], ensure_ascii=False),
        tgt_sql_json=json.dumps(info["tgt_sql"], ensure_ascii=False),
        path=_target_path(info),
        dialect_note=_dialect_note(info),
        task_contract=_rule_task_contract(),
    )
    return dedent(task)


def _format_parser_stage_repair_prompt(info, *, role, gate=None):
    """Format source/target parser repair with one role-specific template."""
    if role == "source":
        values = {
            "stage_subject": "source SQL",
            "repair_scope": "source dialect",
            "primary_path_label": "source parser path",
            "primary_path": _source_path(info),
            "secondary_path_note": (
                f"The later target rule path hint is `{_target_path(info)}`; do not edit it in this "
                "parser-only stage unless it is required for parser correctness."
            ),
            "gate_label": "Parser gate failure",
            "gate": _render_parser_gate(gate),
            "parse_dialect": info["src_db"],
            "parse_subject": "source SQL",
            "preferred_change_scope": "source dialect parser/tokenizer changes",
            "avoid_change_scope": (
                "Do not implement target generator rules in this parser repair stage unless absolutely "
                "necessary for parser correctness."
            ),
        }
    else:
        values = {
            "stage_subject": "expected target SQL",
            "repair_scope": "target dialect",
            "primary_path_label": "target parser path",
            "primary_path": _target_path(info),
            "secondary_path_note": "",
            "gate_label": "Target parser gate failure",
            "gate": _render_parser_gate(gate),
            "parse_dialect": info["tgt_db"],
            "parse_subject": "expected target SQL",
            "preferred_change_scope": "target dialect parser/tokenizer changes",
            "avoid_change_scope": (
                "Do not implement source parser rules in this target parser repair stage unless absolutely "
                "necessary for parser correctness."
            ),
        }
    task = PARSER_STAGE_REPAIR_PROMPT.format(
        src_db=info["src_db"],
        tgt_db=info["tgt_db"],
        src_sql=info["src_sql"],
        tgt_sql=info["tgt_sql"],
        dialect_note=_dialect_note(info),
        **values,
    )
    return dedent(task)


def format_parser_repair_prompt(info, parser_gate=None):
    """Format the source parser-only repair prompt."""
    return _format_parser_stage_repair_prompt(info, role="source", gate=parser_gate)


def format_target_parser_repair_prompt(info, target_parser_gate=None):
    """Format the target parser-only repair prompt."""
    return _format_parser_stage_repair_prompt(info, role="target", gate=target_parser_gate)


def format_summary_prompt(info):
    """Format the summary-only prompt from structured case information."""
    task = SUMMARY_PROMPT.format(
        src_db=info["src_db"], tgt_db=info["tgt_db"],
        src_sql=info["src_sql"], tgt_sql=info["tgt_sql"],
    )
    task = dedent(task)

    return task


def _sanitize_stage_handoff_for_split_manifest(stage_handoff, split_manifest):
    """Remove stale split-derived handoff evidence outside the reliable scope."""
    if not isinstance(stage_handoff, dict):
        return stage_handoff
    # Without a manifest no split/group reference can be authenticated. Treat
    # that as an empty reliable scope while still preserving parent-stage and
    # parser-stage handoff fields below.
    if not isinstance(split_manifest, dict):
        split_manifest = {}
    reliable_splits = [
        split
        for split in split_manifest.get("splits", [])
        if isinstance(split, dict) and _is_hard_split(split)
    ]
    reliable_split_ids = {
        str(split.get("split_id"))
        for split in reliable_splits
        if split.get("split_id") not in (None, "")
    }
    reliable_group_ids = {
        str(split.get("rule_group_id"))
        for split in reliable_splits
        if split.get("rule_group_id") not in (None, "")
    }
    reliable_group_by_split_id = {
        str(split.get("split_id")): str(split.get("rule_group_id") or "")
        for split in reliable_splits
        if split.get("split_id") not in (None, "")
    }
    reliable_variant_ids_by_split_id = {
        str(split.get("split_id")): {
            str(variant.get("variant_id"))
            for variant in (split.get("variants") or [])
            if isinstance(variant, dict)
            and variant.get("hard_assertion") is True
            and variant.get("variant_id") not in (None, "")
        }
        for split in reliable_splits
        if split.get("split_id") not in (None, "")
    }
    sanitized = copy.deepcopy(stage_handoff)

    def split_item_is_reliable(item):
        if not isinstance(item, dict):
            return False
        split_id = item.get("split_id")
        group_id = item.get("rule_group_id")
        # A group id alone cannot authenticate arbitrary split SQL. Split-level
        # records must name a current reliable split, and an optional group id
        # must agree with that split's manifest ownership.
        if split_id in (None, "") or str(split_id) not in reliable_split_ids:
            return False
        if group_id not in (None, "") and reliable_group_by_split_id.get(str(split_id)) != str(group_id):
            return False
        return True

    def group_item_is_reliable(item):
        if not isinstance(item, dict):
            return False
        group_id = item.get("rule_group_id")
        if group_id in (None, "") or str(group_id) not in reliable_group_ids:
            return False
        split_id = item.get("split_id")
        return split_id in (None, "") or split_item_is_reliable(item)

    def variant_item_is_reliable(item):
        if not split_item_is_reliable(item):
            return False
        variant_id = item.get("variant_id")
        if variant_id in (None, ""):
            return False
        split_id = str(item.get("split_id"))
        return str(variant_id) in reliable_variant_ids_by_split_id.get(split_id, set())

    def sanitize_group_item(item):
        """Project nested group evidence onto its current reliable splits."""
        if not group_item_is_reliable(item):
            return None
        group_id = str(item.get("rule_group_id"))
        sanitized_item = copy.deepcopy(item)
        if "split_ids" in sanitized_item:
            sanitized_item["split_ids"] = [
                split_id
                for split_id in (sanitized_item.get("split_ids") or [])
                if str(split_id) in reliable_split_ids
                and reliable_group_by_split_id.get(str(split_id)) == group_id
            ]
        for key in ("failures", "failed_splits"):
            if key not in sanitized_item:
                continue
            filtered = []
            for failure in sanitized_item.get(key) or []:
                if not isinstance(failure, dict):
                    continue
                reference = dict(failure)
                reference.setdefault("rule_group_id", group_id)
                if split_item_is_reliable(reference):
                    filtered.append(failure)
            sanitized_item[key] = filtered
        return sanitized_item

    def filter_group_values(values):
        filtered = []
        for value in values or []:
            if isinstance(value, dict):
                if group_item_is_reliable(value):
                    filtered.append(value)
            elif str(value) in reliable_group_ids:
                filtered.append(value)
        return filtered

    previous_stage = sanitized.get("previous_stage")
    if isinstance(previous_stage, dict):
        group_ids = _normalize_focus_group_ids(
            previous_stage.get("rule_group_ids") or previous_stage.get("rule_group_id")
        )
        valid_group_ids = [group_id for group_id in group_ids if group_id in reliable_group_ids]
        previous_stage.pop("rule_group_id", None)
        previous_stage.pop("rule_group_ids", None)
        if valid_group_ids:
            previous_stage["rule_group_ids"] = valid_group_ids
            if len(valid_group_ids) == 1:
                previous_stage["rule_group_id"] = valid_group_ids[0]

    group_progress = sanitized.get("group_progress")
    if isinstance(group_progress, dict):
        for key in (
            "passed_groups",
            "repair_groups",
            "attempted_groups",
            "failed_groups",
            "group_statuses",
        ):
            if key in group_progress:
                group_progress[key] = filter_group_values(group_progress.get(key))

    validation = sanitized.get("validation")
    if isinstance(validation, dict):
        validation_group_ids = _normalize_focus_group_ids(
            validation.get("rule_group_ids") or validation.get("rule_group_id")
        )
        valid_validation_group_ids = [
            group_id for group_id in validation_group_ids if group_id in reliable_group_ids
        ]
        for key in ("split_results", "failed_group_evidence"):
            if key in validation:
                validation[key] = [
                    item for item in (validation.get(key) or []) if split_item_is_reliable(item)
                ]
        for key in ("group_results", "failed_groups"):
            if key in validation:
                validation[key] = [
                    sanitized_item
                    for item in (validation.get(key) or [])
                    if isinstance(item, dict)
                    for sanitized_item in [sanitize_group_item(item)]
                    if sanitized_item is not None
                ]
        if "variant_hard_failures" in validation:
            validation["variant_hard_failures"] = [
                item
                for item in (validation.get("variant_hard_failures") or [])
                if variant_item_is_reliable(item)
            ]
        repair_target = validation.get("repair_target")
        valid_target_group_ids = []
        if isinstance(repair_target, dict):
            target_group_ids = _normalize_focus_group_ids(
                repair_target.get("rule_group_ids") or repair_target.get("rule_group_id")
            )
            valid_target_group_ids = [
                group_id for group_id in target_group_ids if group_id in reliable_group_ids
            ]
            if target_group_ids and not valid_target_group_ids:
                validation.pop("repair_target", None)
            else:
                repair_target.pop("rule_group_id", None)
                repair_target.pop("rule_group_ids", None)
                if valid_target_group_ids:
                    repair_target["rule_group_ids"] = valid_target_group_ids
                    if len(valid_target_group_ids) == 1:
                        repair_target["rule_group_id"] = valid_target_group_ids[0]
                if "failed_splits" in repair_target:
                    repair_target["failed_splits"] = [
                        item
                        for item in (repair_target.get("failed_splits") or [])
                        if split_item_is_reliable(item)
                    ]
                if "failed_variants" in repair_target:
                    repair_target["failed_variants"] = [
                        item
                        for item in (repair_target.get("failed_variants") or [])
                        if variant_item_is_reliable(item)
                    ]
        repair_target = validation.get("repair_target")
        repair_target_stage = str(
            repair_target.get("stage") if isinstance(repair_target, dict) else ""
        ).lower()
        failed_stage = str(validation.get("failed_stage") or "").lower()
        split_scoped_validation = bool(validation_group_ids or valid_target_group_ids) or any(
            marker in failed_stage or marker in repair_target_stage
            for marker in ("split", "group", "variant")
        )
        if split_scoped_validation:
            # These top-level fields duplicate a focused split failure but have
            # no split id, so they cannot be authenticated against the manifest.
            # Keep only nested failed_splits/failed_variants that passed the
            # split-level reference check above. Parent-stage SQL remains intact.
            for key in (
                "actual_sql",
                "expected_sql",
                "diff_summary",
                "raw_diff_summary",
                "repair_hint",
                "selected_target_diff",
                "transition_summary",
                "quote_insensitive_passed",
            ):
                validation.pop(key, None)
            validation.pop("rule_group_id", None)
            validation.pop("rule_group_ids", None)
            scoped_group_ids = list(dict.fromkeys(valid_validation_group_ids + valid_target_group_ids))
            if scoped_group_ids:
                validation["rule_group_ids"] = scoped_group_ids
                if len(scoped_group_ids) == 1:
                    validation["rule_group_id"] = scoped_group_ids[0]
    return sanitized


def _prepend_stage_handoff(task, stage_handoff):
    if not stage_handoff:
        return task
    handoff_payload = model_facing_diagnostics(_strip_legacy_family_metadata(stage_handoff))
    handoff = _json_dumps_limited(handoff_payload, limit=5000)
    return STAGE_HANDOFF_PROMPT.format(
        handoff=handoff,
        task=task,
    )


# Case context and parser-gate rendering.
def _target_path(info):
    return info.get("target_path") or info.get("target_primary_path") or info.get("path") or ""


def _source_path(info):
    if info.get("source_path") or info.get("source_primary_path"):
        return info.get("source_path") or info.get("source_primary_path")
    path = info.get("path") or info.get("target_path") or ""
    src_db = info.get("src_db")
    if path and src_db:
        if "/" in path:
            return f"{path.rsplit('/', 1)[0]}/{src_db}.py"
        return f"{src_db}.py"
    return path


def _dialect_note(info):
    dialects = {str(info.get("src_db") or ""), str(info.get("tgt_db") or "")}
    if "tsql" not in dialects:
        return ""
    return "\n" + TSQL_DIALECT_CONTEXT


def _rule_task_contract():
    return RULE_TASK_CONTRACT


def _render_parser_gate(parser_gate, *, compact_sql=False):
    if not parser_gate:
        return "not available"
    if compact_sql:
        compact = {
            key: parser_gate.get(key)
            for key in ("ok", "status", "root", "role", "dialect", "error_type", "error")
            if parser_gate.get(key) not in (None, [], {})
        }
        node_types = list(parser_gate.get("node_types") or [])
        if node_types:
            compact["node_types"] = node_types[:12]
            compact["node_type_count"] = len(node_types)
        return _json_dumps_limited(compact, limit=1600)
    return _json_dumps_limited(parser_gate, limit=4000)


def _json_dumps_limited(value, limit=10000):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= limit:
        return text
    compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(compact) <= limit:
        return compact
    return json.dumps(
        {
            "prompt_truncated": True,
            "reason": "json_payload_exceeded_prompt_limit",
            "original_json_chars": len(text),
        },
        ensure_ascii=False,
        indent=2,
    )


# Split diagnostics and overview prompt rendering.
def _render_split_manifest(
    split_manifest,
    focus_rule_group_id=None,
    *,
    max_groups=None,
    max_diagnostic_hints_per_category=None,
):
    if not split_manifest:
        return "not available"
    split_manifest = model_facing_diagnostics(split_manifest)
    focus_group_ids = _normalize_focus_group_ids(focus_rule_group_id)
    if focus_group_ids:
        return _render_focused_split_work_order(
            split_manifest,
            focus_group_ids,
        )

    hint_limit = _diagnostic_hint_limit(max_diagnostic_hints_per_category)
    workset = build_rule_workset(
        split_manifest,
        max_groups=_overview_group_limit(max_groups),
        max_diagnostic_hints_per_category=hint_limit,
    )
    summary = _full_sql_diagnostic_summary(split_manifest, workset)
    text = _render_full_sql_diagnostic_cards(summary)
    if len(text) <= 10000:
        return text
    compressed = _compress_diagnostic_summary_for_prompt(summary)
    text = _render_full_sql_diagnostic_cards(compressed, compact=True)
    if len(text) <= 10000:
        return text
    policy = summary.get("diagnostic_hint_policy") or {}
    hint_ids = [
        _prompt_text(hint.get("hint_id"), limit=80)
        for hint in summary.get("diagnostic_hints", [])
        if isinstance(hint, dict) and hint.get("hint_id")
    ]
    fallback_lines = [
        "Full SQL contract: Source SQL and Expected target SQL above are the task contract.",
        "Diagnostic note: split-derived hints were too large to render fully; they remain evidence only.",
        "Overview coverage:",
        f"- showing {len(summary.get('diagnostic_hints') or [])} representative hint(s)",
        f"- {policy.get('total_hard_assertion_count', 0)} hard split assertion(s) may be checked",
    ]
    if hint_ids:
        fallback_lines.append("- hint ids: " + ", ".join(hint_ids[:20]))
    return "\n".join(fallback_lines)


def _overview_group_limit(max_groups):
    return MAX_PROMPT_RULE_GROUPS if max_groups is None else max(0, int(max_groups))


def _diagnostic_hint_limit(max_diagnostic_hints_per_category):
    if max_diagnostic_hints_per_category is None:
        return MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT
    return max(0, int(max_diagnostic_hints_per_category))


def _is_hard_split(split):
    return split_is_reliable_rule_evidence(split)


def _group_signature_from_hard_units(group, hard_units, *, limit=900):
    signature = next(
        (
            _prompt_text(unit.get("rule_signature"), limit=limit)
            for unit in hard_units
            if unit.get("rule_signature")
        ),
        "",
    )
    if signature:
        return signature
    hard_split_ids = {
        str(unit.get("split_id"))
        for unit in hard_units
        if unit.get("split_id") not in (None, "")
    }
    group_split_ids = {
        str(split_id)
        for split_id in (group.get("split_ids") or [])
        if split_id not in (None, "")
    }
    # A manifest-level signature is trustworthy only when every split that
    # contributed to it is still reliable. This prevents stale focused group
    # ids from reviving a signature derived from an AST fallback/misalignment.
    if hard_units and group_split_ids and group_split_ids.issubset(hard_split_ids):
        return _prompt_text(group.get("rule_signature"), limit=limit)
    return ""


def _normalize_focus_group_ids(value):
    if value in (None, "", [], ()):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = [value]
    group_ids = []
    for item in raw_values:
        text = str(item or "").strip()
        if text:
            group_ids.append(text)
    return group_ids


def _full_sql_diagnostic_summary(split_manifest, workset):
    shown_hints = _diagnostic_hints_from_workset(workset)
    total_hard_assertions = sum(
        1 for split in split_manifest.get("splits", []) if _is_hard_split(split)
    )
    summary = {
        "split_quality": split_manifest.get("split_quality"),
        "diagnostic_hint_policy": {
            "total_hard_assertion_count": total_hard_assertions,
            "hidden_category_count": max(
                0,
                int(workset.get("total_group_count") or 0)
                - len(workset.get("shown_groups") or []),
            ),
            "max_diagnostic_hints_per_category": workset.get("max_diagnostic_hints_per_category"),
        },
        "diagnostic_hints": shown_hints,
        "error_type": split_manifest.get("error_type"),
        "error": split_manifest.get("error"),
    }
    return _strip_legacy_family_metadata(summary)


def _compress_diagnostic_summary_for_prompt(summary):
    compressed = json.loads(json.dumps(summary, ensure_ascii=False))
    for hint in compressed.get("diagnostic_hints", []):
        sql_specs = (
            ("source_sql", "source_text_span", "source_expr", ("source_only", "actual_only", "expected_only")),
            ("current_output", "current_text_span", "current_expr", ("actual_only", "target_actual_only", "target_actual_changed_tokens")),
            ("expected_output", "expected_text_span", "expected_expr", ("expected_only", "target_expected_only", "target_expected_changed_tokens")),
        )
        for key, span_key, expr_key, token_keys in sql_specs:
            value = hint.get(key)
            if isinstance(value, str) and len(value) > 500:
                hint[key] = _hint_sql_excerpt(
                    hint,
                    key,
                    span_key=span_key,
                    expr_key=expr_key,
                    token_keys=token_keys,
                    limit=500,
                )
        expr_specs = (
            ("source_expr", ("source_only", "actual_only", "expected_only")),
            ("current_expr", ("actual_only", "target_actual_only", "target_actual_changed_tokens")),
            ("expected_expr", ("expected_only", "target_expected_only", "target_expected_changed_tokens")),
        )
        for key, token_keys in expr_specs:
            value = hint.get(key)
            if isinstance(value, str) and len(value) > 240:
                hint[key] = _hint_expr_excerpt(hint, key, token_keys=token_keys, limit=240)
        mismatch = hint.get("mismatch")
        if isinstance(mismatch, dict) and isinstance(mismatch.get("diff_summary"), list):
            mismatch["diff_summary"] = mismatch["diff_summary"][:4]
    return compressed


def _diagnostic_hints_from_workset(workset):
    hints = []
    for group in workset.get("shown_groups", []):
        category_hint = group.get("rule_group_id")
        for split in group.get("hint_source_splits", []):
            hint = _diagnostic_hint_from_split(split, category_hint=category_hint)
            if hint:
                hints.append(hint)
    return hints


def _diagnostic_hint_from_split(split, *, category_hint=None):
    diff_anchor = split.get("diff_anchor") if isinstance(split.get("diff_anchor"), dict) else {}
    hint = {
        "hint_id": split.get("split_id"),
        "category_hint": category_hint or split.get("rule_group_id"),
        "slice": split.get("slice_type"),
        "hard_assertion": split.get("hard_assertion"),
        "diagnostic_only_reason": split.get("diagnostic_only_reason") or split.get("skip_reason"),
        "source_sql": split.get("source_sql"),
        "current_output": split.get("actual_sql"),
        "expected_output": split.get("expected_sql"),
        "source_expr": split.get("source_rule_fragment_sql") or split.get("source_expr_sql"),
        "current_expr": split.get("actual_rule_fragment_sql") or split.get("actual_expr_sql"),
        "expected_expr": split.get("expected_rule_fragment_sql") or split.get("expected_expr_sql"),
        "source_text_span": split.get("source_text_span") or diff_anchor.get("source_text_span"),
        "current_text_span": split.get("actual_text_span") or diff_anchor.get("actual_text_span"),
        "expected_text_span": split.get("expected_text_span") or diff_anchor.get("expected_text_span"),
        "source_ast_path": split.get("source_ast_path"),
        "selected_ancestor_path": split.get("selected_ancestor_path"),
        "mismatch": _diagnostic_mismatch(split),
    }
    return {key: value for key, value in hint.items() if value not in (None, [], {})}


def _diagnostic_mismatch(split):
    selected = split.get("selected_target_diff") if isinstance(split.get("selected_target_diff"), dict) else {}
    mismatch = {}
    for key in ("actual_only", "expected_only", "actual_changed_tokens", "expected_changed_tokens"):
        value = selected.get(key)
        if value:
            mismatch[f"target_{key}"] = value
    transition = split.get("transition_summary") if isinstance(split.get("transition_summary"), dict) else {}
    for key in ("source_only", "actual_only", "expected_only"):
        value = transition.get(key)
        if value:
            mismatch[key] = value
    diff_summary = split.get("diff_summary") or []
    if diff_summary:
        mismatch["diff_summary"] = diff_summary[:4]
    return mismatch


def _render_full_sql_diagnostic_cards(summary, *, compact=False):
    policy = summary.get("diagnostic_hint_policy") if isinstance(summary.get("diagnostic_hint_policy"), dict) else {}
    lines = [
        "Full SQL contract: Source SQL and Expected target SQL above are the task contract.",
        "Diagnostic note: split-derived hints below are evidence only; they are not task scope.",
        (
            "Validation note: full validation uses the parent SQL as the primary contract and may also "
            "check hard split assertions."
        ),
    ]
    if compact:
        lines.append("Long diagnostic hints are excerpted around the relevant difference.")

    split_quality = summary.get("split_quality") if isinstance(summary.get("split_quality"), dict) else {}
    quality_bits = []
    for key in ("degraded", "split_truncated", "skipped_split_count", "parent_fallback_count"):
        value = split_quality.get(key)
        if value not in (None, [], {}, False, 0):
            quality_bits.append(f"{key}={_prompt_text(value, limit=120)}")

    coverage = [
        f"showing {len(summary.get('diagnostic_hints') or [])} representative hint(s)",
        f"{policy.get('total_hard_assertion_count', 0)} hard split assertion(s) may be checked",
    ]
    hidden_categories = policy.get("hidden_category_count")
    if hidden_categories not in (None, 0):
        coverage.append(f"{hidden_categories} diagnostic category/categories hidden")
    max_per_category = policy.get("max_diagnostic_hints_per_category")
    if max_per_category not in (None, 0):
        coverage.append(f"max {max_per_category} hint(s) per category shown")
    if quality_bits:
        coverage.append("split quality: " + "; ".join(quality_bits))
    lines.extend(["", "Overview coverage:"])
    lines.extend(f"- {item}" for item in coverage)

    hints = summary.get("diagnostic_hints") or []
    if not hints:
        lines.extend(["", "No split-derived diagnostic hints are available. Use the full SQL contract above."])
    else:
        for index, hint in enumerate(hints, start=1):
            if not isinstance(hint, dict):
                continue
            lines.extend(["", _render_diagnostic_hint_card(hint, index, compact=compact)])

    if summary.get("error_type") or summary.get("error"):
        lines.extend(["", "Split diagnostic error:"])
        if summary.get("error_type"):
            lines.append(f"- error_type: {_prompt_text(summary.get('error_type'), limit=160)}")
        if summary.get("error"):
            lines.append(f"- error: {_prompt_text(summary.get('error'), limit=500)}")
    return "\n".join(lines).strip()


def _render_diagnostic_hint_card(hint, index, *, compact=False):
    hint_id = _prompt_text(hint.get("hint_id"), limit=80) or f"hint_{index}"
    category = _prompt_text(hint.get("category_hint"), limit=100)
    slice_type = _prompt_text(hint.get("slice"), limit=80)
    bracket_bits = [bit for bit in (category, slice_type) if bit]
    heading = f"Hint {hint_id}"
    if bracket_bits:
        heading += f" [{' / '.join(bracket_bits)}]"
    lines = [heading]
    if hint.get("hard_assertion") is False:
        reason = _prompt_text(hint.get("diagnostic_only_reason"), limit=160)
        note = "Diagnostic only; this hint is not a hard validation assertion."
        if reason:
            note += f" Reason: {reason}."
        lines.append(note)

    if compact:
        source = _hint_expr_excerpt(
            hint,
            "source_expr",
            token_keys=("source_only", "actual_only", "expected_only"),
            limit=280,
        ) or _hint_sql_excerpt(
            hint,
            "source_sql",
            span_key="source_text_span",
            expr_key="source_expr",
            token_keys=("source_only", "actual_only", "expected_only"),
            limit=280,
        )
        current = _hint_expr_excerpt(
            hint,
            "current_expr",
            token_keys=("actual_only", "target_actual_only", "target_actual_changed_tokens"),
            limit=280,
        ) or _hint_sql_excerpt(
            hint,
            "current_output",
            span_key="current_text_span",
            expr_key="current_expr",
            token_keys=("actual_only", "target_actual_only", "target_actual_changed_tokens"),
            limit=280,
        )
        expected = _hint_expr_excerpt(
            hint,
            "expected_expr",
            token_keys=("expected_only", "target_expected_only", "target_expected_changed_tokens"),
            limit=280,
        ) or _hint_sql_excerpt(
            hint,
            "expected_output",
            span_key="expected_text_span",
            expr_key="expected_expr",
            token_keys=("expected_only", "target_expected_only", "target_expected_changed_tokens"),
            limit=280,
        )
        if source:
            lines.append(f"Source: {source}")
        if current:
            lines.append(f"Current: {current}")
        if expected:
            lines.append(f"Expected: {expected}")
        mismatch = _render_hint_mismatch(hint.get("mismatch"), limit=2)
        if mismatch:
            lines.append("Mismatch: " + " | ".join(line[2:] if line.startswith("- ") else line for line in mismatch))
        return "\n".join(lines)

    source_sql = _indent_text(
        _hint_sql_excerpt(
            hint,
            "source_sql",
            span_key="source_text_span",
            expr_key="source_expr",
            token_keys=("source_only", "actual_only", "expected_only"),
            limit=900,
        )
    )
    if source_sql:
        lines.extend(["Source snippet:", source_sql])

    current_sql = _indent_text(
        _hint_sql_excerpt(
            hint,
            "current_output",
            span_key="current_text_span",
            expr_key="current_expr",
            token_keys=("actual_only", "target_actual_only", "target_actual_changed_tokens"),
            limit=900,
        )
    )
    if current_sql:
        lines.extend(["Current output:", current_sql])

    expected_sql = _indent_text(
        _hint_sql_excerpt(
            hint,
            "expected_output",
            span_key="expected_text_span",
            expr_key="expected_expr",
            token_keys=("expected_only", "target_expected_only", "target_expected_changed_tokens"),
            limit=900,
        )
    )
    if expected_sql:
        lines.extend(["Expected output:", expected_sql])

    expr_parts = []
    source_expr = _hint_expr_excerpt(
        hint,
        "source_expr",
        token_keys=("source_only", "actual_only", "expected_only"),
        limit=260,
    )
    current_expr = _hint_expr_excerpt(
        hint,
        "current_expr",
        token_keys=("actual_only", "target_actual_only", "target_actual_changed_tokens"),
        limit=260,
    )
    expected_expr = _hint_expr_excerpt(
        hint,
        "expected_expr",
        token_keys=("expected_only", "target_expected_only", "target_expected_changed_tokens"),
        limit=260,
    )
    if source_expr:
        expr_parts.append(f"source_expr={source_expr}")
    if current_expr and current_expr != source_expr:
        expr_parts.append(f"current_expr={current_expr}")
    if expected_expr:
        expr_parts.append(f"expected_expr={expected_expr}")
    if expr_parts:
        lines.extend(["Expression focus:", "  " + "; ".join(expr_parts)])

    mismatch = _render_hint_mismatch(hint.get("mismatch"))
    if mismatch:
        lines.append("Mismatch:")
        lines.extend(mismatch)
    return "\n".join(lines)


def _render_hint_mismatch(mismatch, *, limit=4):
    if not isinstance(mismatch, dict):
        return []
    lines = []
    for key in (
        "source_only",
        "actual_only",
        "expected_only",
        "target_actual_only",
        "target_expected_only",
        "target_actual_changed_tokens",
        "target_expected_changed_tokens",
        "diff_summary",
    ):
        value = mismatch.get(key)
        if value:
            lines.append(f"- {key}: {_prompt_text(value, limit=240)}")
        if len(lines) >= limit:
            break
    return lines


# Focused unit and variant prompt rendering.
def _render_focused_split_work_order(
    split_manifest,
    focus_group_ids,
):
    focus_group_id_set = set(focus_group_ids)
    selected_groups = [
        group for group in split_manifest.get("groups", [])
        if group.get("rule_group_id") in focus_group_id_set
    ]
    split_ids_by_group = {
        group.get("rule_group_id"): set(group.get("split_ids") or [])
        for group in selected_groups
    }
    hard_units_by_group = {group_id: [] for group_id in focus_group_ids}
    for split in split_manifest.get("splits", []):
        if not _is_hard_split(split):
            continue
        group_id = split.get("rule_group_id")
        if group_id not in focus_group_id_set:
            continue
        if split_ids_by_group.get(group_id) and split.get("split_id") not in split_ids_by_group[group_id]:
            continue
        hard_units_by_group.setdefault(group_id, []).append(_strip_legacy_family_metadata(split))

    group_signals = []
    for group in selected_groups:
        cleaned_group = _strip_legacy_family_metadata(group)
        group_id = cleaned_group.get("rule_group_id")
        hard_units = hard_units_by_group.get(group_id, [])
        signature = _group_signature_from_hard_units(cleaned_group, hard_units)
        if signature:
            group_signals.append(
                (
                    _prompt_text(group_id, limit=80),
                    signature,
                )
            )

    focused_units = []
    for group_id in focus_group_ids:
        units = hard_units_by_group.get(group_id, [])
        focused_units.extend(units)

    total_hard_units = sum(len(units) for units in hard_units_by_group.values())
    split_quality = split_manifest.get("split_quality") or {}
    quality_bits = []
    for key in ("degraded", "split_truncated", "skipped_split_count", "parent_fallback_count"):
        value = split_quality.get(key)
        if value not in (None, [], {}, False, 0):
            quality_bits.append(f"{key}={_prompt_text(value, limit=120)}")

    def render_lines(*, sql_limit, variant_sql_limit, oversized_note=False):
        lines = FOCUSED_SPLIT_WORK_ORDER_HEADER.format(
            total_hard_units=total_hard_units,
        ).splitlines()
        if group_signals:
            lines.extend(["", "Focused group signals:"])
            for group_id, signature in group_signals:
                lines.append(f"- Group {group_id or 'unknown'}:")
                lines.append(_indent_text(signature, indent="    "))
        if oversized_note:
            lines.append(FOCUSED_SPLIT_OVERSIZED_NOTE)

        if not focused_units:
            lines.append(FOCUSED_SPLIT_EMPTY_SCOPE_NOTE)
        else:
            for index, unit in enumerate(focused_units, start=1):
                lines.extend(
                    [
                        "",
                        _render_expanded_unit(
                            unit,
                            index,
                            sql_limit=sql_limit,
                            variant_sql_limit=variant_sql_limit,
                        ),
                    ]
                )

        if quality_bits:
            lines.extend(["", "Split quality note: " + "; ".join(quality_bits)])
        return lines

    for sql_limit, variant_sql_limit in (
        (1200, 700),
        (120, 90),
    ):
        text = "\n".join(
            render_lines(
                sql_limit=sql_limit,
                variant_sql_limit=variant_sql_limit,
            )
        ).strip()
        if len(text) <= 10000:
            return text

    return "\n".join(
        render_lines(
            sql_limit=120,
            variant_sql_limit=90,
            oversized_note=True,
        )
    ).strip()


def _render_expanded_unit(
    unit,
    index,
    *,
    sql_limit=1200,
    variant_sql_limit=700,
):
    split_id = _prompt_text(unit.get("split_id"), limit=80) or f"unit_{index}"
    slice_type = _prompt_text(unit.get("slice_type"), limit=80) or "sql_unit"
    lines = [f"Unit {split_id} [{slice_type}] required"]

    source_sql = _indent_text(_unit_sql_excerpt(unit, "source_sql", limit=sql_limit))
    if source_sql:
        lines.extend(["Source SQL:", source_sql])

    actual_sql = _indent_text(_unit_sql_excerpt(unit, "actual_sql", limit=sql_limit))
    if actual_sql:
        lines.extend(["Current output:", actual_sql])

    expected_sql = _indent_text(_unit_sql_excerpt(unit, "expected_sql", limit=sql_limit))
    if expected_sql:
        lines.extend(["Expected output:", expected_sql])

    focus_line = _unit_focus_line(unit)
    if focus_line:
        lines.extend(["Expression focus:", f"  {focus_line}"])

    mismatch = _transition_lines(unit)
    if mismatch:
        lines.append("Key mismatch:")
        lines.extend(mismatch)

    variants = unit.get("variants") or []
    lines.extend(
        _render_variant_examples(
            variants,
            sql_limit=variant_sql_limit,
        )
    )

    return "\n".join(lines)


def _render_variant_examples(
    variants,
    *,
    max_examples=PROMPT_VARIANT_EXAMPLE_LIMIT_PER_UNIT,
    sql_limit=700,
):
    if not variants:
        return []

    shown = []
    prompt_variants = [
        variant
        for variant in variants
        if isinstance(variant, dict) and variant.get("hard_assertion", True)
    ]
    prompt_variants.sort(key=lambda variant: variant.get("mutation_type") == "raw_boundary_contract")
    for index, variant in enumerate(prompt_variants[:max_examples], start=1):
        lines = [_variant_header(variant, index)]
        source_sql = _indent_text(_variant_sql_excerpt(variant, "source_sql", limit=sql_limit), indent="    ")
        if source_sql:
            lines.extend(["  Source:", source_sql])
        actual_sql = _indent_text(_variant_sql_excerpt(variant, "actual_sql", limit=sql_limit), indent="    ")
        if actual_sql:
            lines.extend(["  Current:", actual_sql])
        expected_sql = _indent_text(_variant_sql_excerpt(variant, "expected_sql", limit=sql_limit), indent="    ")
        if expected_sql:
            lines.extend(["  Expected:", expected_sql])
        shown.append("\n".join(lines))

    if not shown:
        return []

    lines = ["Variant examples for generalization:"]
    lines.extend(shown)
    return lines


def _variant_header(variant, index):
    variant_id = _prompt_text(variant.get("variant_id"), limit=80) or f"variant_{index}"
    mutation_type = _prompt_text(variant.get("mutation_type"), limit=100)
    return f"- {variant_id}" + (f" [{mutation_type}]" if mutation_type else "")


def _variant_sql_excerpt(variant, key, *, limit):
    raw = variant.get(key)
    if raw in (None, "", [], {}):
        return ""
    text = _prompt_text_full(raw)
    if len(text) <= limit:
        return text

    markers = []
    for other_key in ("source_sql", "actual_sql", "expected_sql"):
        if other_key == key:
            continue
        other_raw = variant.get(other_key)
        if other_raw in (None, "", [], {}):
            continue
        markers.extend(_changed_segment_markers(text, _prompt_text_full(other_raw)))
    return _marker_excerpt(text, markers, limit=limit)


def _changed_segment_markers(text, other_text):
    if not text or not other_text or text == other_text:
        return []
    start = 0
    max_start = min(len(text), len(other_text))
    while start < max_start and text[start] == other_text[start]:
        start += 1

    text_end = len(text)
    other_end = len(other_text)
    while text_end > start and other_end > start and text[text_end - 1] == other_text[other_end - 1]:
        text_end -= 1
        other_end -= 1

    changed = text[start:text_end].strip(" \n\t,;()")
    if not changed:
        return []
    markers = [changed]
    tokens = re.findall(
        r"`[^`]+`|\"[^\"]+\"|'[^']*'|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[/+*%=<>-]",
        changed,
    )
    markers.extend(token for token in tokens if token)
    deduped = []
    seen = set()
    for marker in markers:
        if marker in seen:
            continue
        deduped.append(marker)
        seen.add(marker)
    return deduped


def _unit_focus_line(unit):
    source_expr = _unit_expr_excerpt(unit, "source_expr_sql", limit=260)
    actual_expr = _unit_expr_excerpt(unit, "actual_expr_sql", limit=260)
    expected_expr = _unit_expr_excerpt(unit, "expected_expr_sql", limit=260)
    if not source_expr and not expected_expr:
        return ""
    parts = []
    if source_expr:
        parts.append(f"source_expr={source_expr}")
    if actual_expr and actual_expr != source_expr:
        parts.append(f"actual_expr={actual_expr}")
    if expected_expr:
        parts.append(f"expected_expr={expected_expr}")
    return "; ".join(parts)


def _unit_sql_excerpt(unit, sql_key, *, limit):
    specs = {
        "source_sql": (
            "source_text_span",
            "source_expr_sql",
            ("source_only", "actual_only", "expected_only"),
        ),
        "actual_sql": (
            "actual_text_span",
            "actual_expr_sql",
            ("actual_only", "target_actual_only", "target_actual_changed_tokens"),
        ),
        "expected_sql": (
            "expected_text_span",
            "expected_expr_sql",
            ("expected_only", "target_expected_only", "target_expected_changed_tokens"),
        ),
    }
    span_key, expr_key, token_keys = specs[sql_key]
    return _hint_sql_excerpt(
        _unit_hint_for_excerpt(unit),
        sql_key,
        span_key=span_key,
        expr_key=expr_key,
        token_keys=token_keys,
        limit=limit,
    )


def _unit_expr_excerpt(unit, expr_key, *, limit):
    token_keys = {
        "source_expr_sql": ("source_only", "actual_only", "expected_only"),
        "actual_expr_sql": ("actual_only", "target_actual_only", "target_actual_changed_tokens"),
        "expected_expr_sql": ("expected_only", "target_expected_only", "target_expected_changed_tokens"),
    }[expr_key]
    return _hint_expr_excerpt(
        _unit_hint_for_excerpt(unit),
        expr_key,
        token_keys=token_keys,
        limit=limit,
    )


def _unit_hint_for_excerpt(unit):
    hint = dict(unit)
    for expr_key, rule_key in (
        ("source_expr_sql", "source_rule_fragment_sql"),
        ("actual_expr_sql", "actual_rule_fragment_sql"),
        ("expected_expr_sql", "expected_rule_fragment_sql"),
    ):
        if hint.get(rule_key):
            hint[expr_key] = hint.get(rule_key)
    diff_anchor = hint.get("diff_anchor") if isinstance(hint.get("diff_anchor"), dict) else {}
    for key, anchor_key in (
        ("source_text_span", "source_text_span"),
        ("actual_text_span", "actual_text_span"),
        ("expected_text_span", "expected_text_span"),
    ):
        if not hint.get(key) and diff_anchor.get(anchor_key):
            hint[key] = diff_anchor.get(anchor_key)
    hint["mismatch"] = _diagnostic_mismatch(unit)
    return hint


def _transition_lines(unit, *, limit=4):
    lines = []
    transition = unit.get("transition_summary") if isinstance(unit.get("transition_summary"), dict) else {}
    for key in ("source_only", "actual_only", "expected_only"):
        value = transition.get(key)
        if value:
            lines.append(f"- {key}: {_prompt_text(value, limit=220)}")

    selected = unit.get("selected_target_diff") if isinstance(unit.get("selected_target_diff"), dict) else {}
    for key in ("actual_only", "expected_only", "actual_changed_tokens", "expected_changed_tokens"):
        value = selected.get(key)
        if value:
            lines.append(f"- target_{key}: {_prompt_text(value, limit=220)}")

    for item in unit.get("diff_summary") or []:
        if item:
            lines.append(f"- diff: {_prompt_text(item, limit=260)}")

    deduped = []
    seen = set()
    for line in lines:
        if line in seen:
            continue
        deduped.append(line)
        seen.add(line)
        if len(deduped) >= limit:
            break
    return deduped


# Memory query construction and payload compaction.
def _append_memory_query(
    task,
    info,
    split_manifest=None,
    focus_rule_group_id=None,
    *,
    max_groups=None,
    max_diagnostic_hints_per_category=None,
    stage_handoff=None,
):
    memory_query = _build_memory_query(
        info,
        split_manifest=split_manifest,
        focus_rule_group_id=focus_rule_group_id,
        max_groups=max_groups,
        max_diagnostic_hints_per_category=max_diagnostic_hints_per_category,
        stage_handoff=stage_handoff,
    )
    return MEMORY_QUERY_SECTION_PROMPT.format(
        task=task.rstrip(),
        memory_query=memory_query,
    )


def _build_memory_query(
    info,
    split_manifest=None,
    focus_rule_group_id=None,
    *,
    max_groups=None,
    max_diagnostic_hints_per_category=None,
    stage_handoff=None,
):
    stage_handoff = _sanitize_stage_handoff_for_split_manifest(stage_handoff, split_manifest)
    current_case_id = (
        info.get("rule_name")
        or info.get("case_id")
        or info.get("test_name")
        or info.get("parent_case_id")
    )
    current_sql_pair_hash = sql_pair_hash(
        src_dialect=info.get("src_db"),
        tgt_dialect=info.get("tgt_db"),
        source_sql=info.get("src_sql"),
        expected_sql=info.get("tgt_sql"),
    )
    payload = {
        "schema": "autodialect_memory_query_v1",
        "src_dialect": info.get("src_db"),
        "tgt_dialect": info.get("tgt_db"),
        "source_path": _source_path(info),
        "target_path": _target_path(info),
    }
    if current_case_id:
        payload["current_case_id"] = str(current_case_id)
    if current_sql_pair_hash:
        payload["current_sql_pair_hash"] = current_sql_pair_hash
    if split_manifest:
        groups = list(split_manifest.get("groups", []))
        focus_group_ids = _normalize_focus_group_ids(focus_rule_group_id)
        if focus_group_ids:
            focus_group_id_set = set(focus_group_ids)
            groups = [group for group in groups if group.get("rule_group_id") in focus_group_id_set]
            split_ids = {split_id for group in groups for split_id in (group.get("split_ids") or [])}
            hard_splits = [
                split
                for split in split_manifest.get("splits", [])
                if split.get("split_id") in split_ids and _is_hard_split(split)
            ]
            reliable_group_ids = {
                str(split.get("rule_group_id"))
                for split in hard_splits
                if split.get("rule_group_id") not in (None, "")
            }
            groups = [
                group
                for group in groups
                if str(group.get("rule_group_id") or "") in reliable_group_ids
            ]
            units = [
                _memory_unit(split)
                for split in hard_splits
            ]
            payload.update(
                {
                    "groups": [
                        {
                            "rule_group_id": group.get("rule_group_id"),
                            "rule_signature": _truncate_memory_value(
                                _group_signature_from_hard_units(
                                    group,
                                    [
                                        split
                                        for split in hard_splits
                                        if split.get("rule_group_id") == group.get("rule_group_id")
                                    ],
                                    limit=5000,
                                ),
                                string_limit=5000,
                            ),
                        }
                        for group in groups
                    ],
                    "units": units,
                }
            )
        else:
            hint_limit = _diagnostic_hint_limit(max_diagnostic_hints_per_category)
            workset = build_rule_workset(
                split_manifest,
                max_groups=_overview_group_limit(max_groups),
                max_diagnostic_hints_per_category=hint_limit,
            )
            diagnostic_hints = _memory_diagnostic_hints_from_workset(workset)
            payload.update(
                {
                    "source_sql": _memory_parent_sql_excerpt(
                        info.get("src_sql"),
                        diagnostic_hints,
                        side="source",
                    ),
                    "expected_sql": _memory_parent_sql_excerpt(
                        info.get("tgt_sql"),
                        diagnostic_hints,
                        side="expected",
                    ),
                    "diagnostic_hint_policy": {
                        "scope": "full_sql_retrieval_hints_only",
                        "not_task_scope": True,
                        "max_diagnostic_hints_per_category": hint_limit,
                    },
                    "diagnostic_hints": diagnostic_hints,
                }
            )
    else:
        payload.update(
            {
                "source_sql": _memory_parent_sql_excerpt(info.get("src_sql"), [], side="source"),
                "expected_sql": _memory_parent_sql_excerpt(info.get("tgt_sql"), [], side="expected"),
            }
        )
    residual = _memory_repair_residual_from_handoff(stage_handoff)
    if residual:
        payload["repair_residual"] = residual
    return _dump_memory_query(payload, limit=5000)


def _memory_repair_residual_from_handoff(stage_handoff):
    if not isinstance(stage_handoff, dict):
        return {}
    stage_handoff = model_facing_diagnostics(stage_handoff)
    validation = stage_handoff.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    repair_target = validation.get("repair_target")
    repair_target = repair_target if isinstance(repair_target, dict) else {}
    residual = {
        "previous_stage": (stage_handoff.get("previous_stage") or {}).get("stage")
        if isinstance(stage_handoff.get("previous_stage"), dict)
        else None,
        "failed_stage": validation.get("failed_stage"),
        "repair_reason": repair_target.get("reason"),
        "rule_group_id": repair_target.get("rule_group_id") or validation.get("rule_group_id"),
        "actual_sql": validation.get("actual_sql"),
        "expected_sql": validation.get("expected_sql"),
        "diff_summary": validation.get("diff_summary"),
    }
    failed_splits = repair_target.get("failed_splits")
    if isinstance(failed_splits, list) and failed_splits:
        residual["failed_splits"] = [
            {
                key: _truncate_memory_value(split.get(key), string_limit=350)
                for key in ("split_id", "rule_group_id", "actual_sql", "expected_sql", "diff_summary")
                if isinstance(split, dict) and split.get(key) not in (None, [], {})
            }
            for split in failed_splits[:3]
            if isinstance(split, dict)
        ]
    failed_evidence = validation.get("failed_group_evidence")
    if isinstance(failed_evidence, list) and failed_evidence:
        residual["failed_group_evidence"] = [
            {
                key: _truncate_memory_value(item.get(key), string_limit=260)
                for key in (
                    "rule_group_id",
                    "split_id",
                    "actual_sql",
                    "expected_sql",
                    "diff_summary",
                    "source_node_sql",
                    "actual_node_sql",
                    "expected_node_sql",
                    "actual_only",
                    "expected_only",
                    "transition_summary",
                )
                if isinstance(item, dict) and item.get(key) not in (None, [], {})
            }
            for item in failed_evidence[:3]
            if isinstance(item, dict)
        ]
    worktree = stage_handoff.get("worktree")
    if isinstance(worktree, dict):
        changed_files = worktree.get("changed_files") or worktree.get("sqlglot_changed_files")
        if changed_files:
            residual["changed_files"] = _truncate_memory_value(changed_files, string_limit=200)
    group_progress = stage_handoff.get("group_progress")
    if isinstance(group_progress, dict):
        for key in ("passed_groups", "repair_groups", "attempted_groups"):
            if group_progress.get(key):
                residual[key] = _truncate_memory_value(group_progress.get(key), string_limit=200)
    return {key: value for key, value in residual.items() if value not in (None, "", [], {})}


def _memory_unit(unit):
    selected_diff = unit.get("selected_target_diff") if isinstance(unit.get("selected_target_diff"), dict) else {}
    compact_selected_diff = {
        key: _truncate_memory_value(selected_diff.get(key), string_limit=300)
        for key in (
            "source_node_sql",
            "actual_node_sql",
            "expected_node_sql",
            "actual_only",
            "expected_only",
            "actual_changed_tokens",
            "expected_changed_tokens",
            "basis",
        )
        if selected_diff.get(key) not in (None, [], {})
    }
    item = {
        "split_id": unit.get("split_id"),
        "rule_group_id": unit.get("rule_group_id"),
        "slice_type": unit.get("slice_type"),
        "selected_ancestor_kind": unit.get("selected_ancestor_kind"),
        "source_sql_origin": unit.get("source_sql_origin"),
        "expected_unit_source": unit.get("expected_unit_source"),
        "source_sql": _unit_sql_excerpt(unit, "source_sql", limit=600),
        "actual_sql": _unit_sql_excerpt(unit, "actual_sql", limit=600),
        "expected_sql": _unit_sql_excerpt(unit, "expected_sql", limit=600),
        "source_expr_sql": _unit_expr_excerpt(unit, "source_expr_sql", limit=220),
        "actual_expr_sql": _unit_expr_excerpt(unit, "actual_expr_sql", limit=220),
        "expected_expr_sql": _unit_expr_excerpt(unit, "expected_expr_sql", limit=220),
        "selected_target_diff": compact_selected_diff,
        "transition_summary": _truncate_memory_value(unit.get("transition_summary"), string_limit=220),
    }
    return {key: value for key, value in item.items() if value not in (None, [], {})}


def _memory_diagnostic_hints_from_workset(workset):
    memory_hints = []
    for hint in _diagnostic_hints_from_workset(workset):
        item = {
            key: hint.get(key)
            for key in ("hint_id", "category_hint", "slice")
            if hint.get(key) not in (None, [], {})
        }
        source_sql = _hint_sql_excerpt(
            hint,
            "source_sql",
            span_key="source_text_span",
            expr_key="source_expr",
            token_keys=("source_only", "actual_only", "expected_only"),
            limit=600,
        )
        current_sql = _hint_sql_excerpt(
            hint,
            "current_output",
            span_key="current_text_span",
            expr_key="current_expr",
            token_keys=("actual_only", "target_actual_only", "target_actual_changed_tokens"),
            limit=600,
        )
        expected_sql = _hint_sql_excerpt(
            hint,
            "expected_output",
            span_key="expected_text_span",
            expr_key="expected_expr",
            token_keys=("expected_only", "target_expected_only", "target_expected_changed_tokens"),
            limit=600,
        )
        source_expr = _hint_expr_excerpt(
            hint,
            "source_expr",
            token_keys=("source_only", "actual_only", "expected_only"),
            limit=220,
        )
        current_expr = _hint_expr_excerpt(
            hint,
            "current_expr",
            token_keys=("actual_only", "target_actual_only", "target_actual_changed_tokens"),
            limit=220,
        )
        expected_expr = _hint_expr_excerpt(
            hint,
            "expected_expr",
            token_keys=("expected_only", "target_expected_only", "target_expected_changed_tokens"),
            limit=220,
        )
        for key, value in (
            ("source_sql", source_sql),
            ("current_output", current_sql),
            ("expected_output", expected_sql),
            ("source_expr", source_expr),
            ("current_expr", current_expr),
            ("expected_expr", expected_expr),
        ):
            if value not in (None, "", [], {}):
                item[key] = value
        mismatch = hint.get("mismatch")
        if isinstance(mismatch, dict):
            compact_mismatch = {
                key: _truncate_memory_value(mismatch.get(key), string_limit=220)
                for key in (
                    "source_only",
                    "actual_only",
                    "expected_only",
                    "target_actual_only",
                    "target_expected_only",
                    "target_actual_changed_tokens",
                    "target_expected_changed_tokens",
                    "diff_summary",
                )
                if mismatch.get(key) not in (None, [], {})
            }
            if compact_mismatch:
                item["mismatch"] = compact_mismatch
        if item:
            memory_hints.append(item)
    return memory_hints


def _memory_parent_sql_excerpt(text, hints, *, side, limit=900):
    if text in (None, "", [], {}):
        return ""
    full_text = _prompt_text_full(text)
    if len(full_text) <= limit:
        return full_text
    markers = []
    if side == "source":
        expr_key = "source_expr"
        token_keys = ("source_only", "actual_only", "expected_only")
    else:
        expr_key = "expected_expr"
        token_keys = ("expected_only", "target_expected_only", "target_expected_changed_tokens")
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        expr = hint.get(expr_key)
        if expr:
            markers.append(expr)
        markers.extend(_mismatch_tokens(hint.get("mismatch"), token_keys))
    return _marker_excerpt(full_text, markers, limit=limit)


def _dump_memory_query(payload, limit=5000):
    payload = model_facing_diagnostics(payload)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) <= limit:
        return text
    original_json_chars = len(text)
    for string_limit in (700, 300, 120):
        compact = _compact_memory_query_payload(
            payload,
            original_json_chars=original_json_chars,
            string_limit=string_limit,
        )
        text = json.dumps(compact, ensure_ascii=False, indent=2)
        if len(text) <= limit:
            return text
    compact = _compact_memory_query_payload(
        payload,
        original_json_chars=original_json_chars,
        string_limit=80,
    )
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    minimal = {
        key: compact.get(key)
        for key in (
            "schema",
            "src_dialect",
            "tgt_dialect",
            "path",
            "source_path",
            "target_path",
            "current_case_id",
            "current_sql_pair_hash",
        )
        if compact.get(key) not in (None, [], {})
    }
    minimal["metadata_truncated"] = True
    minimal["original_json_chars"] = original_json_chars
    if compact.get("groups"):
        minimal["groups"] = [
            {"rule_group_id": group.get("rule_group_id")}
            for group in compact.get("groups", [])
            if group.get("rule_group_id")
        ][:3]
    if compact.get("repair_residual"):
        residual = compact.get("repair_residual") or {}
        minimal["repair_residual"] = {
            key: residual.get(key)
            for key in (
                "failed_stage",
                "repair_reason",
                "rule_group_id",
                "diff_summary",
                "failed_splits",
            )
            if isinstance(residual, dict) and residual.get(key) not in (None, [], {})
        }
    if compact.get("diagnostic_hints"):
        minimal["diagnostic_hints"] = [
            {
                key: hint.get(key)
                for key in ("hint_id", "category_hint", "source_expr", "expected_expr")
                if hint.get(key) not in (None, [], {})
            }
            for hint in compact.get("diagnostic_hints", [])
        ][:6]
    if compact.get("units"):
        minimal["units"] = [
            {
                key: unit.get(key)
                for key in ("split_id", "rule_group_id", "source_expr_sql", "expected_expr_sql")
                if unit.get(key) not in (None, [], {})
            }
            for unit in compact.get("units", [])
        ][:6]
    return json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))


def _compact_memory_query_payload(payload, *, original_json_chars=0, string_limit=700):
    compact = {
        key: _truncate_memory_value(payload.get(key), string_limit=string_limit)
        for key in (
            "schema",
            "src_dialect",
            "tgt_dialect",
            "path",
            "source_path",
            "target_path",
            "path_hints",
            "current_case_id",
            "current_sql_pair_hash",
        )
        if payload.get(key) not in (None, [], {})
    }
    if original_json_chars:
        compact["metadata_truncated"] = True
        compact["original_json_chars"] = original_json_chars

    for residual_key in ("repair_target", "repair_residual"):
        repair_target = payload.get(residual_key)
        if not isinstance(repair_target, dict):
            continue
        compact[residual_key] = {
            key: _truncate_memory_value(repair_target.get(key), string_limit=string_limit)
            for key in (
                "stage",
                "previous_stage",
                "failed_stage",
                "rule_group_id",
                "reason",
                "repair_reason",
                "actual_sql",
                "expected_sql",
                "diff_summary",
                "failed_splits",
                "failed_group_evidence",
                "changed_files",
                "passed_groups",
                "repair_groups",
                "attempted_groups",
            )
            if repair_target.get(key) not in (None, [], {})
        }

    groups = []
    for group in payload.get("groups") or []:
        if not isinstance(group, dict):
            continue
        item = {
            key: _truncate_memory_value(group.get(key), string_limit=string_limit)
            for key in ("rule_group_id", "rule_signature")
            if group.get(key) not in (None, [], {})
        }
        if item:
            groups.append(item)
    if groups:
        compact["groups"] = groups

    diagnostic_hints = []
    hint_fields = (
        "hint_id",
        "category_hint",
        "slice",
        "source_sql",
        "current_output",
        "expected_output",
        "source_expr",
        "current_expr",
        "expected_expr",
        "mismatch",
    )
    for hint in payload.get("diagnostic_hints") or []:
        if not isinstance(hint, dict):
            continue
        item = {
            key: _truncate_memory_value(hint.get(key), string_limit=string_limit)
            for key in hint_fields
            if hint.get(key) not in (None, [], {})
        }
        if item:
            diagnostic_hints.append(item)
    if diagnostic_hints:
        compact["diagnostic_hints"] = diagnostic_hints

    units = []
    unit_fields = (
        "split_id",
        "rule_group_id",
        "selected_ancestor_kind",
        "slice_type",
        "source_sql_origin",
        "expected_unit_source",
        "source_sql",
        "actual_sql",
        "expected_sql",
        "source_expr_sql",
        "actual_expr_sql",
        "expected_expr_sql",
        "selected_target_diff",
        "transition_summary",
    )
    for unit in payload.get("units") or []:
        if not isinstance(unit, dict):
            continue
        item = {
            key: _truncate_memory_value(unit.get(key), string_limit=string_limit)
            for key in unit_fields
            if unit.get(key) not in (None, [], {})
        }
        if item:
            units.append(item)
    if units:
        compact["units"] = units

    for key in ("source_sql", "expected_sql"):
        if payload.get(key):
            compact[key] = _truncate_memory_value(payload.get(key), string_limit=string_limit)
    return compact


def _truncate_memory_value(value, string_limit=700):
    if isinstance(value, str):
        scrubbed = _scrub_legacy_family_text(value)
        return scrubbed if len(scrubbed) <= string_limit else scrubbed[:string_limit] + "...[truncated]"
    if isinstance(value, list):
        return [_truncate_memory_value(item, string_limit=string_limit) for item in value[:8]]
    if isinstance(value, dict):
        return {
            key: _truncate_memory_value(item, string_limit=string_limit)
            for key, item in value.items()
            if not _is_legacy_family_key(key) and item not in (None, [], {})
        }
    return value


# Prompt-safe text and excerpt helpers.
def _prompt_text(value, *, limit=900):
    if value in (None, [], {}):
        return ""
    cleaned = model_facing_diagnostics(_strip_legacy_family_metadata(value))
    if isinstance(cleaned, str):
        text = cleaned.strip()
    else:
        text = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _prompt_text_full(value):
    if isinstance(value, str):
        return value
    cleaned = model_facing_diagnostics(_strip_legacy_family_metadata(value))
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def _indent_text(text, *, indent="  "):
    if not text:
        return ""
    return "\n".join(f"{indent}{line}" for line in str(text).splitlines())


def _hint_sql_excerpt(hint, sql_key, *, span_key, expr_key, token_keys, limit):
    raw = hint.get(sql_key)
    if raw in (None, "", [], {}):
        return ""
    text = _prompt_text_full(raw)
    if len(text) <= limit:
        return text

    span = hint.get(span_key)
    span_range = _span_range(span, text)
    if span_range:
        span_start, span_end = span_range
        markers = []
        expr = hint.get(expr_key)
        if expr:
            markers.append(expr)
        markers.extend(_mismatch_tokens(hint.get("mismatch"), token_keys))
        if span_end - span_start > limit:
            marker_excerpt = _marker_excerpt_in_range(text, markers, span_start, span_end, limit=limit)
            if marker_excerpt:
                return marker_excerpt
        return _window_excerpt(text, span_start, span_end, limit=limit)

    markers = []
    span_value = _span_text(span)
    if span_value:
        markers.append(span_value)
    expr = hint.get(expr_key)
    if expr:
        markers.append(expr)
    markers.extend(_mismatch_tokens(hint.get("mismatch"), token_keys))
    return _marker_excerpt(text, markers, limit=limit)


def _hint_expr_excerpt(hint, expr_key, *, token_keys, limit):
    value = hint.get(expr_key)
    if value in (None, "", [], {}):
        return ""
    text = _prompt_text_full(value)
    if len(text) <= limit:
        return text
    markers = _mismatch_tokens(hint.get("mismatch"), token_keys)
    return _marker_excerpt(text, markers, limit=limit)


def _span_text(span):
    if isinstance(span, dict):
        return span.get("text") or span.get("boundary_text")
    if isinstance(span, str):
        return span
    return ""


def _span_range(span, text):
    if not isinstance(span, dict):
        return None
    try:
        start = int(span.get("start"))
        end = int(span.get("end"))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start or start >= len(text):
        return None
    end = min(end, len(text))
    span_fragment = _span_text(span)
    if span_fragment and text[start:end] != span_fragment:
        return None
    return start, end


def _window_excerpt(text, start, end, *, limit):
    if len(text) <= limit:
        return text
    start = max(0, start)
    end = min(len(text), max(end, start))
    focus_len = end - start
    if focus_len >= limit:
        return text[start : start + limit] + "...[truncated]"
    context_budget = max(0, limit - focus_len)
    left_budget = context_budget // 2
    right_budget = context_budget - left_budget
    window_start = max(0, start - left_budget)
    window_end = min(len(text), end + right_budget)
    if window_start == 0:
        window_end = min(len(text), limit)
    elif window_end == len(text):
        window_start = max(0, len(text) - limit)
    excerpt_start = window_start
    excerpt_end = window_end
    boundaries = " \n\t,;()"
    if window_start > 0:
        while excerpt_start < start and text[excerpt_start] not in boundaries:
            excerpt_start += 1
        if excerpt_start < start and text[excerpt_start] in boundaries:
            excerpt_start += 1
    if window_end < len(text):
        while excerpt_end > end and text[excerpt_end - 1] not in boundaries:
            excerpt_end -= 1
    excerpt = text[excerpt_start:excerpt_end]
    if window_start > 0:
        excerpt = "...[truncated]" + excerpt
    if window_end < len(text):
        excerpt = excerpt + "...[truncated]"
    return excerpt


def _head_tail_excerpt(text, *, limit):
    if len(text) <= limit:
        return text
    head = max(1, (limit - 18) // 2)
    tail = max(1, limit - 18 - head)
    return text[:head] + "...[truncated]" + text[-tail:]


def _marker_excerpt(text, markers, *, limit):
    if len(text) <= limit:
        return text
    for marker in markers:
        marker_text = str(marker or "").strip()
        if not marker_text:
            continue
        if len(marker_text) > limit:
            continue
        index = text.find(marker_text)
        if index >= 0:
            return _window_excerpt(text, index, index + len(marker_text), limit=limit)
    return _head_tail_excerpt(text, limit=limit)


def _marker_excerpt_in_range(text, markers, start, end, *, limit):
    for marker in markers:
        marker_text = str(marker or "").strip()
        if not marker_text:
            continue
        if len(marker_text) > limit:
            continue
        index = text.find(marker_text, start, end)
        if index >= 0:
            return _window_excerpt(text, index, index + len(marker_text), limit=limit)
    return ""


def _mismatch_tokens(mismatch, keys):
    if not isinstance(mismatch, dict):
        return []
    tokens = []
    for key in keys:
        value = mismatch.get(key)
        if isinstance(value, list):
            tokens.extend(str(item) for item in value if item not in (None, "", [], {}))
        elif value not in (None, "", [], {}):
            tokens.append(str(value))
    return tokens
