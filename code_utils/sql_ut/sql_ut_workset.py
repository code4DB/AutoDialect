"""Prompt workset selection for split AutoDialect SQL units.

This module consumes completed split manifests and prepares bounded prompt
evidence:
1. Hard-split filtering: select hard assertion split records from the manifest.
2. Priority scoring: rank groups and splits by validation failures,
   expected-token changes, text distance, changed-token volume, and signal
   diversity.
3. Prompt compaction: choose representative diagnostic hints and trim nested
   boundary/ancestor metadata.
4. Output contract: return a small workset consumed by prompt_utils.py for
   full-SQL diagnostics and memory-query payloads.

Raw split generation belongs in sql_ut_splitter_worker.py. Manifest schema,
grouping, and cache validity helpers belong in sql_ut_splitter.py.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from code_utils.constants import AUTODIALECT_MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT
from code_utils.sql_ut.sql_ut_splitter import split_is_reliable_rule_evidence

MAX_PROMPT_RULE_GROUPS = 3
MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT = AUTODIALECT_MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT
MAX_DISTANCE_TEXT_CHARS = 2000

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|`[^`]*`|\"[^\"]*\"|'[^']*'|\d+(?:\.\d+)?|[<>=!]+|[(),*/+\-]")


# 1. Hard-split filtering and token normalization.
def _is_hard_split(split: dict[str, Any]) -> bool:
    return split_is_reliable_rule_evidence(split)


def _list_value(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _tokenize(value: Any) -> list[str]:
    tokens: list[str] = []
    for item in _list_value(value):
        if item in (None, "", [], {}):
            continue
        text = str(item)
        tokens.extend(token.lower() for token in _TOKEN_RE.findall(text))
    return tokens


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        deduped.append(token)
        seen.add(token)
    return deduped


def _changed_tokens(split: dict[str, Any], side: str) -> list[str]:
    token_keys = (
        ("expected_only", "expected_changed_tokens")
        if side == "expected"
        else ("actual_only", "actual_changed_tokens")
    )
    tokens: list[str] = []
    for section_key in ("transition_summary", "selected_target_diff", "diff_anchor", "target_diff"):
        section = split.get(section_key)
        if not isinstance(section, dict):
            continue
        for token_key in token_keys:
            tokens.extend(_tokenize(section.get(token_key)))
    return _dedupe_tokens(tokens)


# 2. Group/split priority scoring and representative selection.
def _text_distance(actual: str, expected: str) -> int:
    if not actual and not expected:
        return 0
    if len(actual) > MAX_DISTANCE_TEXT_CHARS:
        half = MAX_DISTANCE_TEXT_CHARS // 2
        actual = actual[:half] + actual[-half:]
    if len(expected) > MAX_DISTANCE_TEXT_CHARS:
        half = MAX_DISTANCE_TEXT_CHARS // 2
        expected = expected[:half] + expected[-half:]
    ratio = SequenceMatcher(None, actual or "", expected or "").ratio()
    return int((1.0 - ratio) * 1000)


def _split_diff_distance(split: dict[str, Any]) -> int:
    actual = split.get("actual_rule_fragment_sql") or split.get("actual_expr_sql") or split.get("actual_sql") or ""
    expected = split.get("expected_rule_fragment_sql") or split.get("expected_expr_sql") or split.get("expected_sql") or ""
    return _text_distance(str(actual), str(expected))


def _failure_signal(split: dict[str, Any]) -> int:
    score = 0
    for key in (
        "validation_failed_count",
        "failed_validation_count",
        "failed_split_count",
        "failure_count",
        "hard_failure_count",
    ):
        try:
            score += int(split.get(key) or 0)
        except (TypeError, ValueError):
            continue
    if split.get("passed") is False or split.get("validation_passed") is False:
        score += 1
    if split.get("repair_target") or split.get("failed_stage"):
        score += 1
    return score


def _source_path_bucket(path: Any) -> str:
    return re.sub(r"\[\d+\]", "[]", str(path or ""))


def _split_diversity_key(split: dict[str, Any]) -> tuple[Any, ...]:
    expected_tokens = _changed_tokens(split, "expected")
    actual_tokens = _changed_tokens(split, "actual")
    source_expr_tokens = _tokenize(
        split.get("source_rule_fragment_sql") or split.get("source_expr_sql") or split.get("source_sql")
    )
    expected_expr_tokens = _tokenize(
        split.get("expected_rule_fragment_sql") or split.get("expected_expr_sql") or split.get("expected_sql")
    )
    token_key = tuple((expected_tokens or actual_tokens or expected_expr_tokens or source_expr_tokens)[:3])
    return (
        str(split.get("slice_type") or ""),
        str(split.get("selected_ancestor_kind") or split.get("cut_strategy") or ""),
        token_key,
        _source_path_bucket(split.get("source_ast_path")),
    )


def _split_priority(split: dict[str, Any]) -> tuple[Any, ...]:
    expected_tokens = _changed_tokens(split, "expected")
    actual_tokens = _changed_tokens(split, "actual")
    diff_summary = split.get("diff_summary") or []
    return (
        _failure_signal(split),
        not split.get("benchmark_equal", False),
        len(expected_tokens),
        _split_diff_distance(split),
        len(actual_tokens) + len(expected_tokens),
        bool(diff_summary),
        -len(split.get("source_sql", "") or ""),
    )


def _group_priority(group: dict[str, Any]) -> tuple[Any, ...]:
    splits = group.get("splits") or []
    split_priorities = [_split_priority(split) for split in splits]
    diversity_count = len({_split_diversity_key(split) for split in splits})
    return (
        max(split_priorities) if split_priorities else (),
        diversity_count,
        group.get("total_units") or 0,
    )


def _select_representative_splits_for_hints(splits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select split records that will be rendered as diagnostic hints."""
    if limit <= 0:
        return []
    indexed = list(enumerate(splits))
    indexed.sort(key=lambda item: (_split_priority(item[1]), -item[0]), reverse=True)
    if len(splits) <= limit:
        return [split for _index, split in indexed]
    selected: list[dict[str, Any]] = []
    selected_indexes: set[int] = set()
    seen_diversity_keys: set[tuple[Any, ...]] = set()
    for index, split in indexed:
        diversity_key = _split_diversity_key(split)
        if diversity_key in seen_diversity_keys:
            continue
        seen_diversity_keys.add(diversity_key)
        selected.append(split)
        selected_indexes.add(index)
        if len(selected) >= limit:
            return selected
    for index, split in indexed:
        if index in selected_indexes:
            continue
        selected.append(split)
        selected_indexes.add(index)
        if len(selected) >= limit:
            break
    return selected


# 3. Prompt-safe evidence compaction.
def _compact_ancestor_candidates(split: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    compact = []
    for candidate in list(split.get("ancestor_candidates") or [])[:limit]:
        item = {
            "candidate_kind": candidate.get("candidate_kind"),
            "candidate_path": candidate.get("candidate_path"),
            "source_node_type": candidate.get("source_node_type"),
            "expected_node_type": candidate.get("expected_node_type"),
            "parseable": candidate.get("parseable"),
            "score": candidate.get("score"),
            "ancestor_coverage": candidate.get("ancestor_coverage"),
            "source_sql": candidate.get("source_sql"),
            "expected_sql": candidate.get("expected_sql"),
        }
        for key in ("source_sql", "expected_sql"):
            value = item.get(key)
            if isinstance(value, str) and len(value) > 500:
                item[key] = value[:500] + "...[truncated]"
        compact.append({key: value for key, value in item.items() if value not in (None, [], {})})
    return compact


def _truncate_nested_text(value: Any, *, limit: int = 500) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    if isinstance(value, list):
        return [_truncate_nested_text(item, limit=limit) for item in value[:8]]
    if isinstance(value, dict):
        return {
            key: _truncate_nested_text(item, limit=limit)
            for key, item in value.items()
            if item not in (None, [], {})
        }
    return value


def _compact_boundary_trace(split: dict[str, Any]) -> dict[str, Any]:
    trace = {
        "source_text_span": split.get("source_text_span"),
        "actual_text_span": split.get("actual_text_span"),
        "expected_text_span": split.get("expected_text_span"),
        "source_canonical": split.get("source_canonical"),
        "actual": split.get("actual"),
        "expected_raw": split.get("expected_raw"),
        "expected_canonical": split.get("expected_canonical"),
        "target_diff": split.get("target_diff"),
    }
    compact = _truncate_nested_text(trace)
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


# 4. Public workset output contract.
def build_rule_workset(
    split_manifest: dict[str, Any],
    max_groups: int = MAX_PROMPT_RULE_GROUPS,
    max_diagnostic_hints_per_category: int = MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT,
) -> dict[str, Any]:
    """Build bounded, representative split evidence for the rule prompt.

    Only reliable hard-assertion splits participate. A demoted split is unsafe
    as a local rule contract (for example because it lost raw surface syntax,
    spans multiple SELECT scopes, or contains unrelated alias context), so it
    must not re-enter the prompt or memory as a diagnostic fallback. When no
    reliable split remains, the full parent SQL is the sole model-facing
    contract. Groups are prioritized by failure and diff signals, then diverse
    representative splits are selected before filling remaining hint slots.
    """
    manifest_splits = list(split_manifest.get("splits", []))
    hard_splits = [split for split in manifest_splits if _is_hard_split(split)]
    splits = hard_splits
    splits_by_group: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        splits_by_group.setdefault(split.get("rule_group_id"), []).append(split)

    groups = []
    manifest_groups = {group.get("rule_group_id"): group for group in split_manifest.get("groups", [])}
    for group_id, group_splits in splits_by_group.items():
        group = manifest_groups.get(group_id, {})
        groups.append(
            {
                "rule_group_id": group_id,
                "total_units": len(group_splits),
                "splits": group_splits,
            }
        )

    groups.sort(key=_group_priority, reverse=True)

    shown_groups = []
    for group in groups[:max_groups]:
        selected_splits = _select_representative_splits_for_hints(
            group["splits"],
            max_diagnostic_hints_per_category,
        )
        shown_groups.append(
            {
                "rule_group_id": group["rule_group_id"],
                "total_source_units": group["total_units"],
                "hint_source_splits": [
                    {
                        "split_id": split.get("split_id"),
                        "slice_type": split.get("slice_type"),
                        "skip_reason": split.get("skip_reason"),
                        "source_ast_path": split.get("source_ast_path"),
                        "source_text_span": split.get("source_text_span"),
                        "boundary_trace": _compact_boundary_trace(split),
                        "diff_anchor": split.get("diff_anchor"),
                        "unit_completion": split.get("unit_completion"),
                        "cut_strategy": split.get("cut_strategy"),
                        "dependency_merged": split.get("dependency_merged"),
                        "lift_reason": split.get("lift_reason"),
                        "selected_ancestor_kind": split.get("selected_ancestor_kind"),
                        "selected_ancestor_path": split.get("selected_ancestor_path"),
                        "selected_ancestor_coverage": split.get("selected_ancestor_coverage"),
                        "selected_target_diff": split.get("selected_target_diff"),
                        "ancestor_lift_score": split.get("ancestor_lift_score"),
                        "ancestor_candidates": _compact_ancestor_candidates(split),
                        "source_sql": split.get("source_sql"),
                        "actual_sql": split.get("actual_sql"),
                        "expected_sql": split.get("expected_sql"),
                        "source_sql_origin": split.get("source_sql_origin"),
                        "expected_unit_source": split.get("expected_unit_source"),
                        "actual_sql_origin": split.get("actual_sql_origin"),
                        "source_canonical_sql": split.get("source_canonical_sql"),
                        "actual_canonical_sql": split.get("actual_canonical_sql"),
                        "expected_canonical_sql": split.get("expected_canonical_sql"),
                        "source_expr_sql": split.get("source_expr_sql"),
                        "actual_expr_sql": split.get("actual_expr_sql"),
                        "expected_expr_sql": split.get("expected_expr_sql"),
                        "source_rule_fragment_sql": split.get("source_rule_fragment_sql"),
                        "actual_rule_fragment_sql": split.get("actual_rule_fragment_sql"),
                        "expected_rule_fragment_sql": split.get("expected_rule_fragment_sql"),
                        "transition_summary": split.get("transition_summary"),
                        "diff_summary": split.get("diff_summary", [])[:8],
                    }
                    for split in selected_splits
                ],
            }
        )

    return {
        "max_diagnostic_hints_per_category": max_diagnostic_hints_per_category,
        "split_scope": "hard_assertion_splits",
        "hint_selection": {
            "strategy": "priority_limited_diverse_diagnostic_hints",
            "priority_order": [
                "validation_failure_signal",
                "changed_expected",
                "expected_only_token_count",
                "actual_expected_diff_distance",
                "changed_token_count",
                "has_diff_summary",
                "shorter_source_sql",
                "stable_manifest_order",
            ],
            "diversity_before_fill": "slice_type_ancestor_changed_tokens_and_ast_bucket",
        },
        "total_group_count": len(groups),
        "shown_groups": shown_groups,
    }
