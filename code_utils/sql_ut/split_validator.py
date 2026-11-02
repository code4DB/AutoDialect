"""Deterministic split/group/variant validation helpers.

1. Unit validation: transpile one parent, split, or variant SQL unit and compare
   expected output with strict and quote-insensitive diagnostics.
2. Scope selection: ``full_manifest`` validates the complete parent SQL plus all
   hard split evidence; ``focused_group`` skips parent validation and restricts
   split/variant checks to the requested rule group ids for intermediate agent
   steps.
3. Hard split semantics: ``hard_assertion=True`` means the splitter produced a
   parseable source/expected unit that may fail validation. ``False`` marks
   diagnostic-only or unreliable split evidence; it is counted as skipped and
   cannot make a group fail.
4. Failure reporting: build repair targets, failed-group summaries, and repair
   hints consumed by completion validation.
5. Manifest validation: validate parent SQL, hard split units, and split-local
   variants according to the requested scope. Parent SQL has no variants.
"""

from __future__ import annotations

import re
import time
from typing import Any

from code_utils.common_tool import _SqlglotWorker, compact_output_fields, diff_summary, sql_key
from code_utils.sql_ut.sql_ut_splitter import (
    AST_CANONICAL_RENDERING_DIAGNOSTIC_REASON,
    RAW_BOUNDARY_AST_ANCHOR_MISMATCH_DIAGNOSTIC_REASON,
    split_has_raw_boundary_ast_anchor_mismatch,
    split_is_reliable_rule_evidence,
    split_uses_ast_canonical_rendering,
)


# 1. Unit validation helpers.
def _quote_insensitive_key(sql: str) -> str:
    return re.sub(r"`([^`]*)`|\"([^\"]*)\"", lambda match: match.group(1) or match.group(2) or "", sql_key(sql))


def _quote_sensitive_hint() -> str:
    return (
        "Strict SQL comparison failed only because identifier quoting differs. "
        "Keep quote-insensitive results diagnostic-only and fix target identifier/backtick rendering."
    )


def _validate_one(
    worker: _SqlglotWorker,
    source_sql: str,
    expected_sql: str,
    src: str,
    tgt: str,
) -> dict[str, Any]:
    """Transpile and strictly compare one source/expected SQL unit.

    Quote-insensitive equality is retained only as repair diagnostics; it never
    changes the strict pass result. Transpile exceptions become timed failure
    records rather than escaping the manifest validation loop.
    """
    started = time.monotonic()
    try:
        actual = worker.transpile(source_sql, read=src, write=tgt)
        strict_passed = sql_key(actual) == sql_key(expected_sql)
        quote_insensitive_passed = _quote_insensitive_key(actual) == _quote_insensitive_key(expected_sql)
        item = {
            "passed": strict_passed,
            "actual_sql": actual,
            "expected_sql": expected_sql,
            "duration_s": round(time.monotonic() - started, 4),
        }
        if not strict_passed:
            item["diff_summary"] = diff_summary(actual, expected_sql)
            if quote_insensitive_passed:
                item["quote_insensitive_passed"] = True
                item["diagnostic_reason"] = "quote_only_difference"
                item["repair_hint"] = _quote_sensitive_hint()
        return item
    except Exception as exc:
        return {
            "passed": False,
            "actual_sql": str(exc),
            "expected_sql": expected_sql,
            "diff_summary": [f"{type(exc).__name__}: {exc}"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "duration_s": round(time.monotonic() - started, 4),
        }


def _variant_is_hard_failure(item: dict[str, Any]) -> bool:
    if item.get("passed"):
        return False
    if item.get("hard_assertion") is False:
        return False
    return True


def _split_is_hard_assertion(split: dict[str, Any]) -> bool:
    return split_is_reliable_rule_evidence(split)


# 2. Focus and selection helpers.
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


def _variant_fingerprint(variant: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(variant.get("mutation_type") or ""),
        sql_key(str(variant.get("source_sql") or "")),
        sql_key(str(variant.get("expected_sql") or "")),
    )


def _variant_priority(variant: dict[str, Any]) -> tuple[int, int, str]:
    mutation_type = str(variant.get("mutation_type") or "")
    quote_sensitive = 1 if "quote" in mutation_type or "`" in str(variant.get("expected_sql") or "") else 0
    sql_len = len(str(variant.get("source_sql") or "")) + len(str(variant.get("expected_sql") or ""))
    return (quote_sensitive, -sql_len, mutation_type)


def _select_variants_for_validation(split: dict[str, Any]) -> list[dict[str, Any]]:
    """Dedupe hard variants before validation."""
    raw_variants = list(split.get("variants", []) or [])
    hard_variants = [variant for variant in raw_variants if variant.get("hard_assertion", True)]
    deduped_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for variant in hard_variants:
        key = _variant_fingerprint(variant)
        current = deduped_by_key.get(key)
        if current is None or _variant_priority(variant) > _variant_priority(current):
            deduped_by_key[key] = variant

    return sorted(deduped_by_key.values(), key=_variant_priority, reverse=True)


# 3. Failure reporting helpers.
def _split_failure_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_id": item.get("split_id"),
        "rule_group_id": item.get("rule_group_id"),
        "actual_sql": item.get("actual_sql"),
        "expected_sql": item.get("expected_sql"),
        "diff_summary": item.get("diff_summary", []),
        "quote_insensitive_passed": item.get("quote_insensitive_passed"),
        "diagnostic_reason": item.get("diagnostic_reason"),
        "repair_hint": item.get("repair_hint"),
        "error_type": item.get("error_type"),
        "error": item.get("error"),
    }


def _manifest_build_failed(manifest: dict[str, Any]) -> bool:
    status = manifest.get("status")
    if status and status != "split_ok":
        return True
    if manifest.get("error_type") or manifest.get("error"):
        return True
    split_quality = manifest.get("split_quality") or {}
    return split_quality.get("reason") == "split_script_failed"


def _pass_with_split_warning(
    result: dict[str, Any],
    *,
    reason: str,
    warning_target: dict[str, Any],
    repair_hint: str,
    validation_started: float,
) -> dict[str, Any]:
    result.update(
        {
            "passed": True,
            "failed_stage": None,
            "warning_stage": "ut_split",
            "warning_reason": reason,
            "validation_warning": {
                "stage": "ut_split",
                "reason": reason,
                "split_quality": result.get("split_quality", {}),
                "warning_target": warning_target,
                "message": repair_hint,
            },
            "repair_hint": repair_hint,
            "repair_target": {},
            "duration_s": round(time.monotonic() - validation_started, 4),
        }
    )
    return result


def _fail_parent_result(result: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    parent_hint = parent.get("repair_hint") or (
        "Split UTs did not explain the remaining parent mismatch; fix the integration-level SQL difference."
    )
    repair_target = {
        "stage": "parent",
        "reason": "parent_failed_after_split_groups",
        "actual_sql": parent.get("actual_sql"),
        "expected_sql": parent.get("expected_sql"),
        "diff_summary": parent.get("diff_summary", []),
        "quote_insensitive_passed": parent.get("quote_insensitive_passed"),
        "diagnostic_reason": parent.get("diagnostic_reason"),
        "repair_hint": parent.get("repair_hint"),
        "instruction": "Split UTs did not explain the remaining mismatch. Repair only the parent integration difference.",
    }
    result.update(
        {
            "passed": False,
            "failed_stage": "parent",
            "actual_sql": parent.get("actual_sql"),
            "expected_sql": parent.get("expected_sql"),
            "diff_summary": parent.get("diff_summary", []),
            "repair_hint": parent_hint,
            "repair_target": repair_target,
        }
    )
    return result


def _variant_failure_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_id": item.get("variant_id"),
        "mutation_type": item.get("mutation_type"),
        "split_id": item.get("split_id"),
        "rule_group_id": item.get("rule_group_id"),
        "actual_sql": item.get("actual_sql"),
        "expected_sql": item.get("expected_sql"),
        "diff_summary": item.get("diff_summary", []),
        "quote_insensitive_passed": item.get("quote_insensitive_passed"),
        "diagnostic_reason": item.get("diagnostic_reason"),
        "repair_hint": item.get("repair_hint"),
        "error_type": item.get("error_type"),
        "error": item.get("error"),
    }


def _repair_hint_from_failure(item: dict[str, Any]) -> str:
    """Choose a focused repair hint from the failed unit's concrete evidence."""
    evidence = " ".join(
        str(value or "")
        for value in (
            item.get("actual_sql"),
            item.get("expected_sql"),
            " ".join(item.get("diff_summary") or []),
            item.get("transition_summary"),
            item.get("selected_target_diff"),
        )
    ).lower()
    if "divide" in evidence or " / " in evidence or "+/" in evidence or "-/" in evidence:
        return "Fix the arithmetic/operator rewrite shown by this split, including divide(...) and required casts."
    if any(
        token in evidence
        for token in (
            "parsedatetimeornull",
            "formatdatetime",
            "to_char",
            "to_timestamp",
            "%y",
            "%g",
            "%v",
            "%i",
            "yyyy",
            "hh24",
            "toyear",
            "toquarter",
        )
    ):
        return "Fix date/time function and format-token mapping shown by this split."
    if item.get("quote_insensitive_passed"):
        return "Fix identifier quote rendering; strict SQL comparison treats backticks and double quotes as different."
    if "nullable" in evidence or "decimal" in evidence or "float" in evidence:
        return "Fix target type mapping and Nullable(...) wrapper spelling shown by this split."
    if "coalesce" in evidence:
        return "Fix the target-side wrapper difference shown by this split; avoid adding or preserving unwanted COALESCE wrappers."
    if "json" in evidence:
        return "Fix the JSON function/name mapping shown by this split."
    return "Fix the focused SQL unit using the actual/expected diff; do not rely on the parent SQL to hide the mismatch."


# 4. Public manifest validation entrypoint.
def validate_manifest(
    *,
    package_root: str,
    sys_path_backup: list[str],
    manifest: dict[str, Any] | None,
    parent_source_sql: str,
    parent_expected_sql: str,
    src_dialect: str,
    tgt_dialect: str,
    validate_variants: bool = True,
    validation_scope: str = "full_manifest",
    focus_rule_group_id: str | None = None,
    focus_rule_group_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate parent SQL, hard split units, groups, and optional variants.

    ``full_manifest`` validates parent SQL, then all hard splits, then variants
    attached to hard splits that passed. ``focused_group`` skips parent SQL and
    validates only hard splits/variants for the requested groups. Split variants
    are local generalization checks; parent SQL has no variant generation.

    A split is hard unless ``hard_assertion`` is explicitly false. The splitter
    sets false for unparseable or otherwise unreliable units, which become
    skipped diagnostics rather than correctness failures. Failures include a
    stage-specific repair target; degraded split coverage may return a passing
    result with a warning after the parent passes. The input manifest and legacy
    result fields are not mutated.
    """
    validation_started = time.monotonic()
    validation_scope = validation_scope or "full_manifest"
    if validation_scope not in {"full_manifest", "focused_group"}:
        validation_scope = "full_manifest"
    focused_group_ids = (
        _normalize_group_ids(focus_rule_group_ids or focus_rule_group_id)
        if validation_scope == "focused_group"
        else []
    )
    focused_group_id = focused_group_ids[0] if len(focused_group_ids) == 1 else None
    focused_group_id_set = set(focused_group_ids)
    result: dict[str, Any] = {
        "passed": False,
        "failed_stage": None,
        "parent": {},
        "split_results": [],
        "group_results": [],
        "variant_hard_failures": [],
        "skipped_results": [],
        "failed_groups": [],
        "repair_target": {},
        "validation_scope": validation_scope,
        "focus_rule_group_id": focused_group_id,
        "focus_rule_group_ids": focused_group_ids,
        "generated_variant_count": 0,
        "deduped_variant_count": 0,
        "validated_variant_count": 0,
        "split_quality": {},
    }
    worker = _SqlglotWorker(package_root, sys_path_backup)
    try:
        manifest = manifest or {}

        if validation_scope == "focused_group" and not focused_group_ids:
            repair_target = {
                "stage": "split_group",
                "reason": "focused_group_id_missing",
                "instruction": (
                    "Focused-group validation requires at least one rule group id. "
                    "Report this AutoDialect scheduling/spec error instead of falling back to full validation."
                ),
            }
            result.update(
                {
                    "passed": False,
                    "failed_stage": "split_group",
                    "parent": {
                        "skipped": True,
                        "reason": "focused_group_id_missing",
                    },
                    "repair_hint": "Focused-group validation was requested without a rule group id.",
                    "repair_target": repair_target,
                }
            )
            return result

        if focused_group_ids:
            parent = {
                "skipped": True,
                "reason": "focused_group_validation",
                "instruction": "Parent SQL is validated only after all planned rule groups pass.",
            }
        else:
            parent = _validate_one(
                worker,
                parent_source_sql,
                parent_expected_sql,
                src_dialect,
                tgt_dialect,
            )
        result["parent"] = parent

        split_results = []
        variant_results = []
        manifest_splits = [
            split for split in manifest.get("splits", [])
            if not focused_group_id_set or split.get("rule_group_id") in focused_group_id_set
        ]
        manifest_groups = [
            group for group in manifest.get("groups", [])
            if not focused_group_id_set or group.get("rule_group_id") in focused_group_id_set
        ]
        split_count = len(manifest_splits)
        skipped_split_count = 0
        for split in manifest_splits:
            split_variants = list(split.get("variants", []))
            result["generated_variant_count"] += len(split_variants)
            hard_assertion = _split_is_hard_assertion(split)
            if not hard_assertion:
                skipped_split_count += 1
                result["skipped_results"].append(
                    {
                        "split_id": split.get("split_id"),
                        "rule_group_id": split.get("rule_group_id"),
                        "hard_assertion": False,
                        "reason": (
                            split.get("skip_reason")
                            or split.get("parse_warning")
                            or (
                                AST_CANONICAL_RENDERING_DIAGNOSTIC_REASON
                                if split_uses_ast_canonical_rendering(split)
                                else (
                                    RAW_BOUNDARY_AST_ANCHOR_MISMATCH_DIAGNOSTIC_REASON
                                    if split_has_raw_boundary_ast_anchor_mismatch(split)
                                    else "skipped_split"
                                )
                            )
                        ),
                    }
                )
                continue
            item = _validate_one(
                worker,
                split.get("source_sql", ""),
                split.get("expected_sql", ""),
                src_dialect,
                tgt_dialect,
            )
            item.update(
                {
                    "split_id": split.get("split_id"),
                    "rule_group_id": split.get("rule_group_id"),
                    "hard_assertion": hard_assertion,
                    "transition_summary": split.get("transition_summary"),
                    "selected_target_diff": split.get("selected_target_diff"),
                }
            )
            split_results.append(item)
            variants_to_validate = _select_variants_for_validation(split)
            result["deduped_variant_count"] += len(variants_to_validate)
            if not item.get("passed"):
                continue
            for variant in (variants_to_validate if validate_variants else []):
                variant_item = _validate_one(
                    worker,
                    variant.get("source_sql", ""),
                    variant.get("expected_sql", ""),
                    src_dialect,
                    tgt_dialect,
                )
                variant_item.update(
                    {
                        "variant_id": variant.get("variant_id"),
                        "mutation_type": variant.get("mutation_type"),
                        "split_id": split.get("split_id"),
                        "rule_group_id": split.get("rule_group_id"),
                        "hard_assertion": variant.get("hard_assertion", True),
                    }
                )
                variant_results.append(variant_item)
        result["split_results"] = split_results
        result["validated_variant_count"] = len(variant_results)
        variant_hard_failures = [item for item in variant_results if _variant_is_hard_failure(item)]
        result["variant_hard_failures"] = [_variant_failure_summary(item) for item in variant_hard_failures[:5]]
        result["split_quality"] = {
            "split_truncated": bool(manifest.get("split_truncated")),
            "split_count": split_count,
            "hard_assertion_count": split_count - skipped_split_count,
            "skipped_split_count": skipped_split_count,
            "hard_assertion_ratio": ((split_count - skipped_split_count) / split_count) if split_count else 0.0,
            "skipped_split_ratio": (skipped_split_count / split_count) if split_count else 0.0,
            "parent_fallback_count": (manifest.get("split_quality") or {}).get("parent_fallback_count"),
            "parent_fallback_ratio": (manifest.get("split_quality") or {}).get("parent_fallback_ratio"),
            "ast_canonical_rendering_count": (
                manifest.get("split_quality") or {}
            ).get("ast_canonical_rendering_count"),
            "ast_canonical_rendering_ratio": (
                manifest.get("split_quality") or {}
            ).get("ast_canonical_rendering_ratio"),
            "raw_boundary_ast_anchor_mismatch_count": (
                manifest.get("split_quality") or {}
            ).get("raw_boundary_ast_anchor_mismatch_count"),
            "raw_boundary_ast_anchor_mismatch_ratio": (
                manifest.get("split_quality") or {}
            ).get("raw_boundary_ast_anchor_mismatch_ratio"),
            "source_text_span_missing_count": (manifest.get("split_quality") or {}).get("source_text_span_missing_count"),
            "source_text_span_missing_ratio": (manifest.get("split_quality") or {}).get("source_text_span_missing_ratio"),
            "cut_strategy_counts": (manifest.get("split_quality") or {}).get("cut_strategy_counts", {}),
            "expected_status": manifest.get("expected_status"),
            "expected_error": manifest.get("expected_error"),
            "reasons": (manifest.get("split_quality") or {}).get("reasons", []),
            "degraded": bool(
                _manifest_build_failed(manifest)
                or manifest.get("split_truncated")
                or (split_count and skipped_split_count == split_count)
                or manifest.get("expected_status") == "error"
                or (manifest.get("split_quality") or {}).get("degraded")
            ),
        }

        if _manifest_build_failed(manifest):
            if not focused_group_ids and not parent.get("passed"):
                return _fail_parent_result(result, parent)
            warning_target = {
                "stage": "ut_split",
                "reason": "split_manifest_build_failed",
                "status": manifest.get("status"),
                "error_type": manifest.get("error_type"),
                "error": manifest.get("error"),
                "split_quality": result["split_quality"],
                "instruction": (
                    "Parent SQL passed, but UT split construction failed. Treat this as split coverage "
                    "degradation, not as a rule correctness failure."
                ),
            }
            repair_hint = (
                "UT split construction failed after parent SQL passed; record split coverage as degraded "
                "and inspect splitter/parser support separately."
            )
            if not focused_group_ids and parent.get("passed"):
                return _pass_with_split_warning(
                    result,
                    reason="split_manifest_build_failed",
                    warning_target=warning_target,
                    repair_hint=repair_hint,
                    validation_started=validation_started,
                )
            result.update(
                {
                    "passed": False,
                    "failed_stage": "ut_split",
                    "error_type": manifest.get("error_type"),
                    "error": manifest.get("error"),
                    "repair_hint": "UT split construction failed before rule correctness could be proven.",
                    "repair_target": warning_target,
                }
            )
            return result

        group_results = []
        for group in manifest_groups:
            group_split_ids = set(group.get("split_ids", []))
            group_items = [item for item in split_results if item.get("split_id") in group_split_ids]
            hard_items = [item for item in group_items if item.get("hard_assertion")]
            passed = all(item.get("passed") for item in hard_items) if hard_items else not focused_group_ids
            failed_items = [item for item in group_items if not item.get("passed")]
            group_results.append(
                {
                    "rule_group_id": group.get("rule_group_id"),
                    "passed": passed,
                    "split_ids": group.get("split_ids", []),
                    "validated_split_count": len(group_items),
                    "hard_split_count": len(hard_items),
                    "failed_split_count": len(failed_items),
                }
            )
        result["group_results"] = group_results
        failed_groups = []
        for group in group_results:
            failed_items = [
                item for item in split_results
                if item.get("rule_group_id") == group.get("rule_group_id") and not item.get("passed")
            ]
            if not failed_items:
                continue
            summary = {
                "rule_group_id": group.get("rule_group_id"),
                "failed_split_count": len(failed_items),
                "failures": [
                    {
                        "split_id": item.get("split_id"),
                        "actual_sql": item.get("actual_sql"),
                        "expected_sql": item.get("expected_sql"),
                        "diff_summary": item.get("diff_summary", []),
                    }
                    for item in failed_items[:3]
                ],
            }
            failed_groups.append(summary)
        result["failed_groups"] = failed_groups[:5]

        if focused_group_ids:
            found_group_ids = {group.get("rule_group_id") for group in manifest_groups}
            missing_group_ids = [
                group_id for group_id in focused_group_ids
                if group_id not in found_group_ids
            ]
        else:
            missing_group_ids = []
        if missing_group_ids:
            repair_target = {
                "stage": "split_group",
                "rule_group_id": missing_group_ids[0] if len(missing_group_ids) == 1 else None,
                "rule_group_ids": focused_group_ids,
                "missing_rule_group_ids": missing_group_ids,
                "reason": "focused_group_not_found",
                "instruction": "The focused rule group is missing from the split manifest; report this AutoDialect scheduling error.",
            }
            result.update(
                {
                    "passed": False,
                    "failed_stage": "split_group",
                    "rule_group_id": missing_group_ids[0] if len(missing_group_ids) == 1 else None,
                    "repair_hint": "Focused group validation could not find the requested rule group.",
                    "repair_target": repair_target,
                }
            )
            return result

        if focused_group_ids and result["split_quality"]["hard_assertion_count"] == 0:
            reason = (
                "focused_group_no_split_units"
                if split_count == 0
                else "focused_group_no_hard_units"
            )
            repair_target = {
                "stage": "ut_split",
                "rule_group_id": focused_group_id,
                "rule_group_ids": focused_group_ids,
                "reason": reason,
                "split_quality": result["split_quality"],
                "skipped_results": result["skipped_results"][:5],
                "instruction": (
                    "Focused group validation did not contain any hard split unit. "
                    "Treat this as an AutoDialect split/scheduling issue instead of accepting the group."
                ),
            }
            result.update(
                {
                    "passed": False,
                    "failed_stage": "ut_split",
                    "rule_group_id": focused_group_id,
                    "rule_group_ids": focused_group_ids,
                    "repair_hint": "Focused group validation had no hard split units to prove correctness.",
                    "repair_target": repair_target,
                }
            )
            return result

        failed_split = next(
            (item for item in split_results if item.get("hard_assertion") and not item.get("passed")),
            None,
        )
        if failed_split is not None:
            failed_group_items = [
                item for item in split_results
                if item.get("rule_group_id") == failed_split.get("rule_group_id") and not item.get("passed")
            ]
            failed_split_hint = failed_split.get("repair_hint") or _repair_hint_from_failure(failed_split)
            grouped_focus = len(focused_group_ids) > 1
            repair_target = {
                "stage": "split_group",
                "rule_group_id": failed_split.get("rule_group_id"),
                "rule_group_ids": focused_group_ids if grouped_focus else [],
                "reason": "hard_split_failed",
                "quote_insensitive_passed": failed_split.get("quote_insensitive_passed"),
                "failed_splits": [_split_failure_summary(item) for item in failed_group_items[:5]],
                "instruction": (
                    "Repair failed groups in the current focused batch. The repair_target is the "
                    "first failing group; use failed_groups for other failing groups in this batch."
                    if grouped_focus
                    else "Repair only this rule group first. Validation will still run all splits in the group."
                ),
            }
            result.update(
                {
                    "passed": False,
                    "failed_stage": "split_group",
                    "rule_group_id": failed_split.get("rule_group_id"),
                    "rule_group_ids": focused_group_ids if grouped_focus else [],
                    "actual_sql": failed_split.get("actual_sql"),
                    "expected_sql": failed_split.get("expected_sql"),
                    "diff_summary": failed_split.get("diff_summary", []),
                    "repair_hint": failed_split_hint,
                    "repair_target": repair_target,
                }
            )
            return result

        if not focused_group_ids and not parent.get("passed"):
            return _fail_parent_result(result, parent)

        if split_count and skipped_split_count == split_count:
            warning_target = {
                "stage": "ut_split",
                "reason": "no_reliable_split_units",
                "split_quality": result["split_quality"],
                "skipped_results": result["skipped_results"][:5],
                "instruction": (
                    "The parent SQL passed, but every split UT was skipped. Treat this as split coverage "
                    "degradation, not as a rule correctness failure."
                ),
            }
            repair_hint = (
                "All split UTs were skipped after parent SQL passed; record split coverage as degraded "
                "and inspect splitter support separately."
            )
            if not focused_group_ids and parent.get("passed"):
                return _pass_with_split_warning(
                    result,
                    reason="no_reliable_split_units",
                    warning_target=warning_target,
                    repair_hint=repair_hint,
                    validation_started=validation_started,
                )
            result.update(
                {
                    "passed": False,
                    "failed_stage": "ut_split",
                    "repair_hint": "All split UTs were skipped before rule correctness could be proven.",
                    "repair_target": warning_target,
                }
            )
            return result

        failed_variant = next((item for item in variant_hard_failures), None)
        if failed_variant is not None:
            variant_hint = failed_variant.get("repair_hint") or (
                "The parent SQL passed, but a strict variant failed; avoid hardcoding identifiers, literals, or aliases."
            )
            repair_target = {
                "stage": "variant",
                "rule_group_id": failed_variant.get("rule_group_id"),
                "reason": "non_quote_only_variant_failed",
                "quote_insensitive_passed": failed_variant.get("quote_insensitive_passed"),
                "diagnostic_reason": failed_variant.get("diagnostic_reason"),
                "failed_variants": [_variant_failure_summary(failed_variant)],
                "instruction": "Repair only the overfit/generalization mismatch shown by this variant. Quote differences are SQL-sensitive and must not be ignored.",
            }
            result.update(
                {
                    "passed": False,
                    "failed_stage": "variant",
                    "rule_group_id": failed_variant.get("rule_group_id"),
                    "actual_sql": failed_variant.get("actual_sql"),
                    "expected_sql": failed_variant.get("expected_sql"),
                    "diff_summary": failed_variant.get("diff_summary", []),
                    "repair_hint": variant_hint,
                    "repair_target": repair_target,
                }
            )
            return result

        if (
            not focused_group_ids
            and parent.get("passed")
            and result.get("split_quality", {}).get("degraded")
        ):
            warning_target = {
                "stage": "ut_split",
                "reason": "degraded_split_quality",
                "split_quality": result["split_quality"],
                "instruction": (
                    "The parent SQL passed and hard split/variant checks did not fail, but split coverage "
                    "was degraded. Treat this as a coverage warning, not as a rule correctness failure."
                ),
            }
            return _pass_with_split_warning(
                result,
                reason="degraded_split_quality",
                warning_target=warning_target,
                repair_hint=(
                    "Split coverage was degraded after parent SQL passed; inspect splitter/parser coverage "
                    "separately from rule correctness."
                ),
                validation_started=validation_started,
            )

        result["passed"] = True
        result["duration_s"] = round(time.monotonic() - validation_started, 4)
        return result
    finally:
        result.setdefault("duration_s", round(time.monotonic() - validation_started, 4))
        compact_result = compact_output_fields(result)
        result.clear()
        result.update(compact_result)
        worker.close()
