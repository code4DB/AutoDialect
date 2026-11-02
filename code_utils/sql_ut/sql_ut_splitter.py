"""Public split-manifest builder for AutoDialect SQL UTs.

This module is the main-process API between rule_main/rule_eval and the
isolated split worker:
1. Manifest entry point: build_split_manifest() calls
   sql_ut_splitter_worker.py and assembles schema/version metadata,
   transition/diff summaries, benchmark equality, split-quality summaries, and
   normalized split records.
2. Rule grouping: build_rule_group_id() and its helpers derive stable group ids,
   signatures, and semantic grouping hints.
3. Hard assertion normalization: mark split units as hard only when their
   source/expected SQL units are parseable; unreliable units remain
   diagnostic-only skipped evidence.
4. Variant filtering: keep only hard-safe split-local mutation variants before
   downstream validation.
5. Cache and diagnostic helpers: expose is_current_split_manifest() and
   compact_split_manifest() for reuse guards and compact diagnostics.

AST alignment and raw split generation belong in sql_ut_splitter_worker.py.
Prompt-sized representative evidence selection belongs in sql_ut_workset.py.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from code_utils.execution_runtime import run_python_json_file
from code_utils.common_tool import compact_output_fields, diff_summary, sql_key, transition_summary
from code_utils.constants import AUTODIALECT_MAX_SYNTHETIC_VARIANT_SQL_CHARS, AUTODIALECT_SPLIT_HELPER_TIMEOUT_S
from code_utils.sql_abstract import abstract_sql

SPLIT_MANIFEST_VERSION = 14
SPLIT_MANIFEST_SCHEMA = f"autodialect_split_manifest_v{SPLIT_MANIFEST_VERSION}"
DEFAULT_SPLIT_HELPER_TIMEOUT_S = AUTODIALECT_SPLIT_HELPER_TIMEOUT_S

AST_CANONICAL_RENDERING_DIAGNOSTIC_REASON = "surface_contract_lost_in_ast_rendering"
RAW_BOUNDARY_AST_ANCHOR_MISMATCH_DIAGNOSTIC_REASON = "raw_boundary_ast_anchor_mismatch"
_SQL_UNIT_ORIGIN_FIELDS = (
    "source_sql_origin",
    "actual_sql_origin",
    "expected_unit_source",
)


_SPLIT_WORKER_PATH = str(Path(__file__).with_name("sql_ut_splitter_worker.py").resolve())


_DEFAULT_VARIANT_LIMIT_PER_SPLIT = 3
_MAX_SYNTHETIC_VARIANT_SQL_CHARS = AUTODIALECT_MAX_SYNTHETIC_VARIANT_SQL_CHARS
_SPECIAL_NUMERIC_VARIANT_VALUES = {"0", "1"}

_VARIANT_PRIORITY = {
    "rename_first_column_reference": 40,
    "replace_first_numeric_literal": 30,
    "rename_projection_alias_quoted": 20,
    "rename_projection_alias_unquoted": 10,
}


_NUMERIC_LITERAL_RE = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SQL_IDENTIFIER_COMPONENT = r'(?:`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_$]*)'
_QUALIFIED_IDENTIFIER_RE = re.compile(
    rf"(?=({_SQL_IDENTIFIER_COMPONENT})\s*\.\s*({_SQL_IDENTIFIER_COMPONENT}))"
)
_LEADING_WITH_RE = re.compile(r"(?:^|\n)\s*WITH\b", re.IGNORECASE)
_FORMAT_TOKEN_RE = re.compile(
    r"'((?:YYYY|YY|MM|DD|HH24|MI|SS|IYYY|IW|%[A-Za-z]|[-_: /])+)'",
    re.IGNORECASE,
)
_TYPE_TOKEN_RE = re.compile(
    r"\b(?:REAL|DOUBLE\s+PRECISION|FLOAT(?:32|64)?|DATE(?:TIME)?(?:64)?|TIMESTAMP|INT(?:32|64)?|BIGINT|VARCHAR|STRING)\b",
    re.IGNORECASE,
)


# 1. Worker orchestration and manifest assembly.
def build_split_manifest(
    *,
    project_path: str,
    src_sql: str,
    expected_sql: str,
    src_dialect: str,
    tgt_dialect: str,
    parent_case_id: str,
) -> dict[str, Any]:
    """Build the versioned split manifest for one parent SQL case.

    Split generation runs in the isolated worker subprocess. This function
    enriches successful splits with transition/diff evidence, filters unsafe
    variants, assigns stable rule groups, and records coverage quality. Worker
    failures return a degraded manifest rather than raising or inventing units.
    """
    started = time.monotonic()
    result = run_python_json_file(
        _SPLIT_WORKER_PATH,
        stdin_json={
            "package_root": project_path,
            "src_sql": src_sql,
            "expected_sql": expected_sql,
            "src_dialect": src_dialect,
            "tgt_dialect": tgt_dialect,
        },
        timeout=DEFAULT_SPLIT_HELPER_TIMEOUT_S,
        normalize=True,
    )
    duration_s = round(time.monotonic() - started, 4)
    manifest: dict[str, Any] = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "schema_version": SPLIT_MANIFEST_VERSION,
        "parent_case_id": parent_case_id,
        "status": result.get("status", "split_failed"),
        "duration_s": duration_s,
        "actual_sql": result.get("actual_sql"),
        "source_root": result.get("source_root"),
        "expected_root": result.get("expected_root"),
        "expected_status": result.get("expected_status"),
        "expected_error": result.get("expected_error"),
        "select_count": result.get("select_count"),
        "processed_select_count": result.get("processed_select_count"),
        "source_text_span_count": result.get("source_text_span_count"),
        "expected_select_count": result.get("expected_select_count"),
        "actual_select_count": result.get("actual_select_count"),
        "split_truncated": result.get("split_truncated", False),
        "split_quality": {},
        "splits": [],
        "groups": [],
        "error_type": result.get("error_type"),
        "error": result.get("error"),
    }
    if not result.get("ok"):
        manifest["split_quality"] = {
            "degraded": True,
            "reason": "split_script_failed",
            "split_truncated": False,
            "split_count": 0,
            "hard_assertion_count": 0,
            "skipped_split_count": 0,
            "expected_status": manifest.get("expected_status"),
            "expected_error": manifest.get("expected_error"),
        }
        return _compact_split_manifest_output(manifest)

    result_splits = list(result.get("splits", []))
    result_splits_by_id = {
        split.get("split_id"): split
        for split in result_splits
        if isinstance(split, dict) and split.get("split_id")
    }
    filtered_splits = []
    for split in result_splits:
        if split.get("slice_type") == "surface_provenance_guard_slice":
            trigger_split_ids = split.get("preservation_trigger_split_ids") or []
            if not any(
                split_is_reliable_rule_evidence(result_splits_by_id.get(trigger_split_id))
                for trigger_split_id in trigger_split_ids
            ):
                continue
        filtered_splits.append(split)

    groups: dict[str, dict[str, Any]] = {}
    for split in filtered_splits:
        split["hard_assertion"] = bool(split.get("hard_assertion", True))
        if not split["hard_assertion"]:
            split["skip_reason"] = split.get("skip_reason") or split.get("parse_warning") or "unreliable_split"
        split["transition_summary"] = transition_summary(
            split.get("source_rule_fragment_sql") or split.get("source_expr_sql") or split.get("source_sql", ""),
            split.get("actual_rule_fragment_sql") or split.get("actual_expr_sql") or split.get("actual_sql", ""),
            split.get("expected_rule_fragment_sql") or split.get("expected_expr_sql") or split.get("expected_sql", ""),
        )
        split["diff_summary"] = diff_summary(split.get("actual_sql", ""), split.get("expected_sql", ""))
        split["benchmark_equal"] = sql_key(split.get("actual_sql", "")) == sql_key(split.get("expected_sql", ""))
        _apply_hard_unit_eligibility(split)
        _filter_variants_for_split(split)
        group_id, signature = build_rule_group_id(
            source_expr_sql=split.get("source_expr_sql") or split.get("source_sql", ""),
            expected_expr_sql=split.get("expected_expr_sql") or split.get("expected_sql", ""),
            source_surface_sql=split.get("source_rule_fragment_sql"),
            expected_surface_sql=split.get("expected_rule_fragment_sql"),
            src_dialect=src_dialect,
            tgt_dialect=tgt_dialect,
        )
        split["rule_group_id"] = group_id
        split["rule_signature"] = signature
        manifest["splits"].append(split)
        group = groups.setdefault(
            group_id,
            {
                "rule_group_id": group_id,
                "rule_signature": signature if split_is_reliable_rule_evidence(split) else None,
                "split_ids": [],
                "hard_assertion_count": 0,
                "skipped_split_count": 0,
            },
        )
        group["split_ids"].append(split["split_id"])
        if split_is_reliable_rule_evidence(split):
            group["hard_assertion_count"] += 1
            group["rule_signature"] = group.get("rule_signature") or signature
        else:
            group["skipped_split_count"] += 1

    manifest["status"] = "split_ok"
    manifest["groups"] = list(groups.values())
    split_count = len(manifest["splits"])
    hard_assertion_count = sum(1 for split in manifest["splits"] if split.get("hard_assertion"))
    parent_fallback_count = sum(1 for split in manifest["splits"] if split.get("slice_type") == "parent")
    ast_canonical_rendering_count = sum(
        1 for split in manifest["splits"] if split_uses_ast_canonical_rendering(split)
    )
    raw_boundary_ast_anchor_mismatch_count = sum(
        1 for split in manifest["splits"] if split_has_raw_boundary_ast_anchor_mismatch(split)
    )
    source_text_span_missing_count = sum(1 for split in manifest["splits"] if not split.get("source_text_span"))
    cut_strategy_counts: dict[str, int] = {}
    for split in manifest["splits"]:
        cut_strategy = split.get("cut_strategy") or "unknown"
        cut_strategy_counts[cut_strategy] = cut_strategy_counts.get(cut_strategy, 0) + 1
    degraded_reasons = []
    if manifest.get("split_truncated"):
        degraded_reasons.append("split_truncated")
    if manifest.get("expected_status") == "error":
        degraded_reasons.append("expected_parse_error")
    if split_count and hard_assertion_count == 0:
        degraded_reasons.append("all_splits_skipped")
    manifest["split_quality"] = {
        "degraded": bool(degraded_reasons),
        "reasons": degraded_reasons,
        "split_truncated": bool(manifest.get("split_truncated")),
        "split_count": split_count,
        "hard_assertion_count": hard_assertion_count,
        "skipped_split_count": split_count - hard_assertion_count,
        "parent_fallback_count": parent_fallback_count,
        "parent_fallback_ratio": (parent_fallback_count / split_count) if split_count else 0.0,
        "ast_canonical_rendering_count": ast_canonical_rendering_count,
        "ast_canonical_rendering_ratio": (
            ast_canonical_rendering_count / split_count
        ) if split_count else 0.0,
        "raw_boundary_ast_anchor_mismatch_count": raw_boundary_ast_anchor_mismatch_count,
        "raw_boundary_ast_anchor_mismatch_ratio": (
            raw_boundary_ast_anchor_mismatch_count / split_count
        ) if split_count else 0.0,
        "source_text_span_missing_count": source_text_span_missing_count,
        "source_text_span_missing_ratio": (source_text_span_missing_count / split_count) if split_count else 0.0,
        "cut_strategy_counts": cut_strategy_counts,
        "expected_status": manifest.get("expected_status"),
        "expected_error": manifest.get("expected_error"),
    }
    return _compact_split_manifest_output(manifest)


def _compact_split_manifest_output(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compact diagnostics while preserving required completion-spec lists."""
    compact = compact_output_fields(manifest)
    compact.setdefault("splits", [])
    compact.setdefault("groups", [])
    return compact


# 2. Rule grouping helpers used by manifest assembly.
def _fallback_abstract(sql: str) -> str:
    sql = re.sub(r"`[^`]*`|\"[^\"]*\"", "identifier", sql)
    sql = re.sub(r"'[^']*'", "'?'", sql)
    sql = re.sub(r"\b\d+(?:\.\d+)?\b", "0", sql)
    sql = re.sub(r"\s+", " ", sql).strip().lower()
    return sql


def _abstract_sql_fragment(sql: str, dialect: str) -> str:
    """Abstract a SQL fragment by wrapping it in SELECT when needed."""
    candidates = [sql]
    stripped = sql.strip()
    if not stripped.lower().startswith("select"):
        candidates.append(f"SELECT {stripped}")

    for candidate in candidates:
        try:
            return abstract_sql(candidate, read=dialect)
        except Exception:
            continue
    return _fallback_abstract(sql)


def _protected_token_signature(*sql_parts: str) -> str:
    """Keep tokens that distinguish different rules and must not be abstracted away."""
    tokens: list[str] = []
    for sql in sql_parts:
        tokens.extend(token.upper() for token in _FORMAT_TOKEN_RE.findall(sql or ""))
        tokens.extend(re.sub(r"\s+", " ", token.upper()) for token in _TYPE_TOKEN_RE.findall(sql or ""))
        lower = (sql or "").lower()
        for op in (" div ", " / ", "divide(", "parsedatetimeornull", "formatdatetime", "toyear", "toquarter"):
            if op in lower:
                tokens.append(op.strip().upper())
    return "|".join(dict.fromkeys(tokens))


def _function_tokens(*sql_parts: str) -> str:
    tokens: list[str] = []
    for sql in sql_parts:
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\s*(?=\()", sql or ""):
            tokens.append(token.upper())
    return "|".join(dict.fromkeys(tokens))


def _semantic_hints(source_expr_sql: str, expected_expr_sql: str) -> list[str]:
    """Return broad rule hints derived from the actual source/expected diff."""
    protected = _protected_token_signature(source_expr_sql, expected_expr_sql)
    functions = _function_tokens(source_expr_sql, expected_expr_sql)
    lower = f"{source_expr_sql}\n{expected_expr_sql}".lower()
    hints: list[str] = []
    if " div " in lower or " / " in lower or "divide(" in lower:
        hints.append("operator=DIVIDE")
    if "`" in expected_expr_sql or '"' in source_expr_sql or '"' in expected_expr_sql:
        hints.append("operator=IDENTIFIER_QUOTE")
    if functions:
        hints.append(f"functions={functions}")
    if protected:
        hints.append(f"protected={protected}")
    return [hint for hint in hints if hint and not hint.endswith("=")]


def build_rule_group_id(
    *,
    source_expr_sql: str,
    expected_expr_sql: str,
    source_surface_sql: str | None = None,
    expected_surface_sql: str | None = None,
    src_dialect: str,
    tgt_dialect: str,
) -> tuple[str, str]:
    """Return stable group id plus human-readable signature."""
    source_abs = _abstract_sql_fragment(source_expr_sql, src_dialect)
    expected_abs = _abstract_sql_fragment(expected_expr_sql, tgt_dialect)
    protected = _protected_token_signature(
        source_expr_sql,
        expected_expr_sql,
        source_surface_sql or "",
        expected_surface_sql or "",
    )
    functions = _function_tokens(
        source_expr_sql,
        expected_expr_sql,
        source_surface_sql or "",
        expected_surface_sql or "",
    )
    semantic_hints = _semantic_hints(source_expr_sql, expected_expr_sql)
    structural_signature = "\n".join(
        [
            f"source={source_abs}",
            f"expected={expected_abs}",
            f"protected={protected}",
            f"functions={functions}",
        ]
    )
    identity_signature = (
        "\n".join(f"hint={hint}" for hint in semantic_hints)
        if semantic_hints
        else structural_signature
    )
    digest = hashlib.sha1(identity_signature.encode("utf-8", errors="ignore")).hexdigest()[:12]
    signature = f"identity={identity_signature}\n{structural_signature}"
    return f"rule_{digest}", signature


# 3. Variant filtering helpers.
def _mask_single_quoted_literals(sql: str, replacement: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(sql):
        if sql[index] != "'":
            output.append(sql[index])
            index += 1
            continue
        output.append(replacement)
        index += 1
        while index < len(sql):
            if sql[index] == "'":
                if index + 1 < len(sql) and sql[index + 1] == "'":
                    index += 2
                    continue
                index += 1
                break
            index += 1
    return "".join(output)


def _collapse_sql_space(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip()).rstrip(";")


def _strip_identifier_quotes(identifier: str) -> str:
    text = (identifier or "").strip()
    if len(text) >= 2 and (
        (text[0] == text[-1] and text[0] in {'"', "`"})
        or (text[0] == "[" and text[-1] == "]")
    ):
        text = text[1:-1]
    return text.lower()


_TABLE_ALIAS_AS_RE = re.compile(
    r"(?:\b(FROM|JOIN)\s+|,\s*)"
    r"(?P<table>(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w$]*))?)"
    r"\s+(?P<as>AS\s+)?"
    r"(?P<alias>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w$]*)"
    r"(?=\s+(?:ON|USING|WHERE|JOIN|INNER|LEFT|RIGHT|FULL|CROSS|GROUP|ORDER|HAVING|LIMIT|QUALIFY|UNION|$)|\s*,|\)|$)",
    re.IGNORECASE,
)


def _table_alias_as_styles(sql: str) -> list[tuple[str, str, bool]]:
    masked = _mask_single_quoted_literals(sql or "", "'__STR__'")
    styles: list[tuple[str, str, bool]] = []
    for match in _TABLE_ALIAS_AS_RE.finditer(masked):
        table = ".".join(
            _strip_identifier_quotes(part)
            for part in re.split(r"\s*\.\s*", match.group("table"))
        )
        alias = _strip_identifier_quotes(match.group("alias"))
        if table and alias:
            styles.append((table, alias, bool(match.group("as"))))
    return styles


def _table_alias_as_context_style_differs(actual_sql: str, expected_sql: str) -> bool:
    actual_styles = _table_alias_as_styles(actual_sql)
    expected_styles = _table_alias_as_styles(expected_sql)
    if not actual_styles or not expected_styles:
        return False
    actual_by_pair: dict[tuple[str, str], list[bool]] = {}
    expected_by_pair: dict[tuple[str, str], list[bool]] = {}
    for table, alias, has_as in actual_styles:
        actual_by_pair.setdefault((table, alias), []).append(has_as)
    for table, alias, has_as in expected_styles:
        expected_by_pair.setdefault((table, alias), []).append(has_as)
    return any(
        set(actual_by_pair[pair]) != set(expected_by_pair[pair])
        for pair in actual_by_pair.keys() & expected_by_pair.keys()
    )


_SELECT_ONE_FROM_RE = re.compile(r"^\s*SELECT\s+1\s+FROM\b", re.IGNORECASE)


def _uses_select_one_from(sql: str) -> bool:
    return bool(_SELECT_ONE_FROM_RE.search(_mask_single_quoted_literals(sql or "", "'__STR__'")))


_UNION_RE = re.compile(r"\bUNION(?:\s+ALL)?\b", re.IGNORECASE)
_SELECT_RE = re.compile(r"\bSELECT\b", re.IGNORECASE)
_QUOTED_IDENTIFIER_OR_COMMENT_RE = re.compile(
    r"`(?:``|[^`])*`|\"(?:\"\"|[^\"])*\"|\[[^\]]*\]|--[^\r\n]*|/\*.*?\*/",
    re.DOTALL,
)


def _contains_hard_scope_boundary(sql: str) -> bool:
    masked = _mask_single_quoted_literals(sql or "", "'__STR__'")
    masked = _QUOTED_IDENTIFIER_OR_COMMENT_RE.sub(" __NON_CODE__ ", masked)
    return bool(
        _LEADING_WITH_RE.search(masked)
        or _UNION_RE.search(masked)
        or len(_SELECT_RE.findall(masked)) > 1
    )


def _mark_split_diagnostic_only(split: dict[str, Any], reason: str) -> None:
    if split.get("hard_assertion") is False:
        split["skip_reason"] = split.get("skip_reason") or reason
        return
    split["hard_assertion"] = False
    split["skip_reason"] = split.get("skip_reason") or reason
    split["diagnostic_only_reason"] = reason


def split_uses_ast_canonical_rendering(split: dict[str, Any]) -> bool:
    """Return whether any unit side was serialized from a canonical AST."""
    return any(
        (
            "ast_canonical_" in str(split.get(field) or "")
            or "select_one_projection_aligned" in str(split.get(field) or "")
        )
        for field in _SQL_UNIT_ORIGIN_FIELDS
    )


def _normalized_identifier_component(value: str) -> str:
    value = (value or "").strip()
    if len(value) >= 2 and (value[0], value[-1]) in {
        ("`", "`"),
        ('"', '"'),
        ("[", "]"),
    }:
        value = value[1:-1]
    return value.lower()


def _qualified_identifier_references(sql: str) -> set[tuple[str, str]]:
    """Extract qualified references without treating string contents as SQL."""
    masked = _mask_single_quoted_literals(sql or "", "''")
    return {
        (_normalized_identifier_component(match.group(1)), _normalized_identifier_component(match.group(2)))
        for match in _QUALIFIED_IDENTIFIER_RE.finditer(masked)
    }


def _raw_and_canonical_qualified_references_disagree(raw_sql: str, canonical_sql: str) -> bool:
    """Return a mismatch only when both representations provide strong evidence."""
    raw_references = _qualified_identifier_references(raw_sql)
    canonical_references = _qualified_identifier_references(canonical_sql)
    return bool(raw_references and canonical_references and raw_references != canonical_references)


def split_has_raw_boundary_ast_anchor_mismatch(split: dict[str, Any]) -> bool:
    """Detect a raw projection boundary attached to the wrong AST expression.

    This deliberately checks only qualified column references. Function names,
    aliases, identifier case, quoting, and ordinary dialect rewrites are not
    used as mismatch evidence. A one-sided or unqualified expression therefore
    remains undecided instead of being rejected.
    """
    if split.get("slice_type") != "projection_slice":
        return False

    source_raw = split.get("source_raw") or {}
    source_canonical = split.get("source_canonical") or {}
    actual = split.get("actual") or {}
    expected_raw = split.get("expected_raw") or {}
    expected_canonical = split.get("expected_canonical") or {}
    source_raw_sql = source_raw.get("boundary_text") or (split.get("source_text_span") or {}).get("text")
    source_anchor_sql = source_canonical.get("anchor_node_sql") or split.get("source_expr_sql")
    actual_raw_sql = actual.get("raw_boundary_text") or (split.get("actual_text_span") or {}).get("text")
    actual_anchor_sql = actual.get("node_sql_canonical") or split.get("actual_expr_sql")
    expected_raw_sql = expected_raw.get("boundary_text") or (split.get("expected_text_span") or {}).get("text")
    expected_anchor_sql = expected_canonical.get("node_sql") or split.get("expected_expr_sql")

    return _raw_and_canonical_qualified_references_disagree(
        source_raw_sql or "",
        source_anchor_sql or "",
    ) or _raw_and_canonical_qualified_references_disagree(
        actual_raw_sql or "",
        actual_anchor_sql or "",
    ) or _raw_and_canonical_qualified_references_disagree(
        expected_raw_sql or "",
        expected_anchor_sql or "",
    )


def split_is_reliable_rule_evidence(split: dict[str, Any]) -> bool:
    """Return whether a split is safe for validation and model-facing rules."""
    if not isinstance(split, dict) or not bool(split.get("hard_assertion", True)):
        return False
    if split.get("diagnostic_only_reason") or split.get("skip_reason"):
        return False
    # Recompute every current eligibility rule instead of trusting a persisted
    # hard_assertion bit. This keeps stale manifests from reviving units that a
    # newer splitter would demote (complex scopes, alias context, AST rendering,
    # raw/AST misalignment, or asymmetric placeholders).
    return _hard_unit_ineligibility_reason(split) is None


def _hard_unit_ineligibility_reason(split: dict[str, Any]) -> str | None:
    source_sql = split.get("source_sql") or ""
    actual_sql = split.get("actual_sql") or ""
    expected_sql = split.get("expected_sql") or ""

    if split.get("slice_type") == "surface_provenance_guard_slice":
        trigger_split_ids = split.get("preservation_trigger_split_ids")
        if (
            not isinstance(trigger_split_ids, list)
            or not trigger_split_ids
            or not all(isinstance(split_id, str) and split_id for split_id in trigger_split_ids)
        ):
            return "surface_provenance_guard_without_trigger"

    # Canonical AST serialization is never an exact surface contract. This
    # applies to both explicit fallbacks and expression units: serializers can
    # normalize functions, aliases, quotes, and parentheses, while a mismatched
    # SELECT alignment can attach an otherwise valid expression to the wrong
    # FROM/JOIN context.
    if split_uses_ast_canonical_rendering(split):
        return AST_CANONICAL_RENDERING_DIAGNOSTIC_REASON

    # A raw projection boundary can still be paired with the wrong SELECT AST
    # expression when repeated branches are aligned by ordinal. Do not expose
    # such internally inconsistent units as rules or exact validation targets.
    if split_has_raw_boundary_ast_anchor_mismatch(split):
        return RAW_BOUNDARY_AST_ANCHOR_MISMATCH_DIAGNOSTIC_REASON

    # Hard units must be local. CTE, UNION, and nested-SELECT units are useful
    # diagnostics, but mix scopes/rules already covered by parent validation.
    if any(_contains_hard_scope_boundary(sql) for sql in (source_sql, expected_sql)):
        return "complex_scope_unit_diagnostic_only"

    # SELECT 1 placeholders are only reliable when both sides use the same
    # abstraction. Otherwise the agent can fix the JOIN/WHERE rule and still
    # fail on an artificial projection mismatch.
    if _uses_select_one_from(source_sql) != _uses_select_one_from(expected_sql):
        return "asymmetric_projection_placeholder"

    # Demote only unresolved table-alias AS drift in the current translation.
    # Comparing source to expected would incorrectly reject ordinary dialect
    # changes that the current translator already renders correctly.
    if (
        _table_alias_as_context_style_differs(actual_sql, expected_sql)
        and not _table_alias_diff_is_rule_relevant(split)
    ):
        return "table_alias_as_context_style_mismatch"

    return None


def _apply_hard_unit_eligibility(split: dict[str, Any]) -> None:
    reason = _hard_unit_ineligibility_reason(split)
    if reason:
        _mark_split_diagnostic_only(split, reason)


def _numeric_literals_outside_strings(sql: str) -> list[str]:
    return _NUMERIC_LITERAL_RE.findall(_mask_single_quoted_literals(sql or "", "''"))


def _numeric_skeleton(sql: str) -> str:
    masked = _mask_single_quoted_literals(sql or "", "'__STR__'")
    return _collapse_sql_space(_NUMERIC_LITERAL_RE.sub("__NUM__", masked))


def _identifier_tokens(sql: str) -> list[str]:
    masked = _mask_single_quoted_literals(sql or "", "'__STR__'")
    return _IDENTIFIER_RE.findall(masked)


def _identifier_count(sql: str, identifier: str | None) -> int:
    if not identifier:
        return 0
    return sum(1 for token in _identifier_tokens(sql) if token == identifier)


def _identifier_skeleton(sql: str) -> str:
    masked = _mask_single_quoted_literals(sql or "", "'__STR__'")
    return _collapse_sql_space(_IDENTIFIER_RE.sub("__ID__", masked))


def _single_identifier_substitution(
    original_sql: str,
    variant_sql: str,
    replacement: str,
) -> tuple[bool, str | None]:
    original_tokens = _identifier_tokens(original_sql)
    variant_tokens = _identifier_tokens(variant_sql)
    if len(original_tokens) != len(variant_tokens):
        return False, None
    diffs = [
        (left, right)
        for left, right in zip(original_tokens, variant_tokens)
        if left != right
    ]
    if not diffs or any(right != replacement for _, right in diffs):
        return False, None
    original_identifiers = {left for left, _ in diffs}
    if len(original_identifiers) != 1:
        return False, None
    return True, next(iter(original_identifiers))


def _numeric_literal_kind(value: str) -> str:
    unsigned = value[1:] if value.startswith("-") else value
    return "decimal" if "." in unsigned else "integer"


def _numeric_literal_sign(value: str) -> str:
    return "-" if value.startswith("-") else "+"


def _single_numeric_variant_is_safe(variant: dict[str, Any], split: dict[str, Any]) -> tuple[bool, str]:
    """Accept only one-to-one numeric mutations that preserve SQL structure.

    Source and expected sides must replace the same unique literal with a value
    of the same numeric kind and sign. The reason string explains rejection or
    records the accepted mutation contract.
    """
    source_sql = split.get("source_sql") or ""
    expected_sql = split.get("expected_sql") or ""
    variant_source_sql = variant.get("source_sql") or ""
    variant_expected_sql = variant.get("expected_sql") or ""
    source_numbers = _numeric_literals_outside_strings(source_sql)
    expected_numbers = _numeric_literals_outside_strings(expected_sql)
    variant_source_numbers = _numeric_literals_outside_strings(variant_source_sql)
    variant_expected_numbers = _numeric_literals_outside_strings(variant_expected_sql)
    if not source_numbers or not expected_numbers:
        return False, "numeric_literal_missing"
    if len(source_numbers) != len(variant_source_numbers) or len(expected_numbers) != len(variant_expected_numbers):
        return False, "numeric_literal_count_changed"
    source_changed_indexes = [
        index
        for index, (original, changed) in enumerate(zip(source_numbers, variant_source_numbers))
        if original != changed
    ]
    expected_changed_indexes = [
        index
        for index, (original, changed) in enumerate(zip(expected_numbers, variant_expected_numbers))
        if original != changed
    ]
    if len(source_changed_indexes) != 1 or source_changed_indexes != expected_changed_indexes:
        return False, "numeric_variant_replacement_not_aligned"
    changed_index = source_changed_indexes[0]
    source_value = source_numbers[changed_index]
    expected_value = expected_numbers[changed_index]
    if source_value != expected_value:
        return False, "numeric_literal_not_aligned"
    if source_value in _SPECIAL_NUMERIC_VARIANT_VALUES:
        return False, "numeric_literal_special_value"
    if source_numbers.count(source_value) != 1 or expected_numbers.count(expected_value) != 1:
        return False, "numeric_literal_not_one_to_one"
    replacement_value = variant_source_numbers[changed_index]
    if replacement_value != variant_expected_numbers[changed_index]:
        return False, "numeric_variant_replacement_not_aligned"
    if replacement_value == source_value:
        return False, "numeric_variant_replacement_unchanged"
    if replacement_value in source_numbers or replacement_value in expected_numbers:
        return False, "numeric_variant_replacement_not_fresh"
    if _numeric_literal_kind(replacement_value) != _numeric_literal_kind(source_value):
        return False, "numeric_variant_replacement_kind_changed"
    if _numeric_literal_sign(replacement_value) != _numeric_literal_sign(source_value):
        return False, "numeric_variant_replacement_sign_changed"
    if _numeric_skeleton(source_sql) != _numeric_skeleton(variant_source_sql):
        return False, "numeric_source_structure_changed"
    if _numeric_skeleton(expected_sql) != _numeric_skeleton(variant_expected_sql):
        return False, "numeric_expected_structure_changed"
    return True, "single_numeric_literal_one_to_one"


def _identifier_variant_is_safe(variant: dict[str, Any], split: dict[str, Any], replacement: str) -> tuple[bool, str]:
    source_sql = split.get("source_sql") or ""
    expected_sql = split.get("expected_sql") or ""
    variant_source_sql = variant.get("source_sql") or ""
    variant_expected_sql = variant.get("expected_sql") or ""
    if replacement not in variant_source_sql or replacement not in variant_expected_sql:
        return False, "identifier_replacement_missing"
    source_ok, source_identifier = _single_identifier_substitution(source_sql, variant_source_sql, replacement)
    expected_ok, expected_identifier = _single_identifier_substitution(expected_sql, variant_expected_sql, replacement)
    if not source_ok:
        return False, "identifier_source_structure_changed"
    if not expected_ok:
        return False, "identifier_expected_structure_changed"
    if source_identifier != expected_identifier:
        return False, "identifier_not_aligned"
    if _identifier_count(variant_source_sql, source_identifier) or _identifier_count(
        variant_expected_sql,
        expected_identifier,
    ):
        return False, "identifier_original_still_present"
    if _identifier_skeleton(source_sql) != _identifier_skeleton(variant_source_sql):
        return False, "identifier_source_structure_changed"
    if _identifier_skeleton(expected_sql) != _identifier_skeleton(variant_expected_sql):
        return False, "identifier_expected_structure_changed"
    return True, "single_identifier_one_to_one"


def _variant_has_synthetic_alias_noise(variant: dict[str, Any]) -> bool:
    combined = (variant.get("source_sql", "") or "") + "\n" + (variant.get("expected_sql", "") or "")
    return "autodialect_probe" in combined


def _variant_sql_prompt_safety_issue(variant: dict[str, Any]) -> str | None:
    source_sql = variant.get("source_sql", "") or ""
    expected_sql = variant.get("expected_sql", "") or ""
    combined = f"{source_sql}\n{expected_sql}"
    # Synthetic variants are anti-overfit hints, not real benchmark variants.
    # Keep them short and local so they do not distract the agent with full CTEs
    # or unrelated parent-query structure.
    if len(combined) > _MAX_SYNTHETIC_VARIANT_SQL_CHARS:
        return "variant_sql_too_large"
    masked = _mask_single_quoted_literals(combined, "'__STR__'")
    if _LEADING_WITH_RE.search(masked):
        return "variant_contains_cte"
    return None


def _alias_diff_is_rule_relevant(split: dict[str, Any]) -> bool:
    selected = split.get("selected_target_diff") or {}
    target_diff = split.get("target_diff") or {}
    path = " ".join(
        str(value or "")
        for value in (
            selected.get("candidate_path"),
            selected.get("target_diff_node_path"),
            split.get("target_diff_node_path"),
        )
    ).lower()
    changed_tokens = " ".join(
        str(token)
        for token in (
            list(selected.get("actual_changed_tokens") or [])
            + list(selected.get("expected_changed_tokens") or [])
            + list(target_diff.get("actual_changed_tokens") or [])
            + list(target_diff.get("expected_changed_tokens") or [])
        )
    )
    return "alias" in path or "`" in changed_tokens or '"' in changed_tokens


def _table_alias_diff_is_rule_relevant(split: dict[str, Any]) -> bool:
    """Return whether the selected rule itself targets a relation alias.

    Projection aliases and identifier-quote changes must not exempt unrelated
    FROM/JOIN alias-AS context drift. The broader alias helper remains in use
    for projection-alias mutation filtering.
    """
    selected = split.get("selected_target_diff") or {}
    target_diff = split.get("target_diff") or {}
    path = " ".join(
        str(value or "")
        for value in (
            selected.get("candidate_path"),
            selected.get("target_diff_node_path"),
            split.get("target_diff_node_path"),
        )
    ).lower()
    node_types = " ".join(
        str(value or "")
        for value in (
            selected.get("source_node_type"),
            selected.get("actual_node_type"),
            selected.get("expected_node_type"),
            target_diff.get("source_node_type"),
            target_diff.get("actual_node_type"),
            target_diff.get("expected_node_type"),
            split.get("cut_node_type"),
            split.get("target_diff_node_type"),
        )
    ).lower()
    return "tablealias" in node_types or (
        "alias" in path
        and any(scope in path for scope in ("from", "join", "table"))
    )


def _variant_is_stable_for_split(variant: dict[str, Any], split: dict[str, Any]) -> tuple[bool, str]:
    mutation_type = variant.get("mutation_type")

    if _variant_has_synthetic_alias_noise(variant):
        return False, "synthetic_alias_noise"
    prompt_safety_issue = _variant_sql_prompt_safety_issue(variant)
    if prompt_safety_issue:
        return False, prompt_safety_issue
    if mutation_type in {"rename_projection_alias_unquoted", "rename_projection_alias_quoted"}:
        if not _alias_diff_is_rule_relevant(split):
            return False, "alias_not_part_of_selected_diff"
        ok, reason = _identifier_variant_is_safe(variant, split, "autodialect_variant_alias")
        if not ok:
            return False, reason
        variant["hard_assertion"] = True
        return True, reason
    if mutation_type == "rename_first_column_reference":
        ok, reason = _identifier_variant_is_safe(variant, split, "autodialect_variant_col")
        if not ok:
            return False, reason
        variant["hard_assertion"] = True
        return True, reason
    if mutation_type == "replace_first_numeric_literal":
        ok, reason = _single_numeric_variant_is_safe(variant, split)
        if not ok:
            return False, reason
        variant["hard_assertion"] = True
        return True, reason
    return False, "unsupported_variant_mutation"


def _variant_rank(variant: dict[str, Any], split: dict[str, Any]) -> tuple[int, int, int]:
    mutation_type = variant.get("mutation_type")
    transition = split.get("transition_summary") or {}
    selected = split.get("selected_target_diff") or {}
    evidence = " ".join(
        str(token)
        for token in (
            list(transition.get("source_only") or [])
            + list(transition.get("actual_only") or [])
            + list(transition.get("expected_only") or [])
            + list(selected.get("actual_changed_tokens") or [])
            + list(selected.get("expected_changed_tokens") or [])
        )
    ).lower()
    score = _VARIANT_PRIORITY.get(mutation_type, 0)
    if mutation_type == "replace_first_numeric_literal" and ("/" in evidence or "divide" in evidence):
        score += 20
    if mutation_type == "rename_first_column_reference":
        score += 10
    if mutation_type in {"rename_projection_alias_unquoted", "rename_projection_alias_quoted"} and _alias_diff_is_rule_relevant(split):
        score += 20
    return (score, -len(variant.get("source_sql", "") or ""), -len(variant.get("expected_sql", "") or ""))


def _filter_variants_for_split(split: dict[str, Any]) -> None:
    """Keep only hard-safe variants from split-local mutation candidates."""
    original_variants = list(split.get("variants") or [])
    limit = _DEFAULT_VARIANT_LIMIT_PER_SPLIT
    selected = []
    for variant in original_variants:
        ok, _reason = _variant_is_stable_for_split(variant, split)
        if ok:
            selected.append(variant)
    selected.sort(key=lambda variant: _variant_rank(variant, split), reverse=True)
    diverse = []
    seen_mutation_types = set()
    for variant in selected:
        mutation_type = variant.get("mutation_type")
        if mutation_type in seen_mutation_types:
            continue
        seen_mutation_types.add(mutation_type)
        diverse.append(variant)
        if len(diverse) >= limit:
            break
    if len(diverse) < limit:
        for variant in selected:
            if variant in diverse:
                continue
            diverse.append(variant)
            if len(diverse) >= limit:
                break
    split["variants"] = diverse
    split["variant_generation"] = {
        "candidate_count": len(original_variants),
    }


# 4. Cache and diagnostic helpers.
def is_current_split_manifest(manifest: Any) -> bool:
    """Return whether a cached split manifest was produced by the current splitter schema."""
    structurally_current = (
        isinstance(manifest, dict)
        and manifest.get("status") == "split_ok"
        and manifest.get("schema_version") == SPLIT_MANIFEST_VERSION
        and manifest.get("schema") == SPLIT_MANIFEST_SCHEMA
        and isinstance(manifest.get("splits"), list)
        and isinstance(manifest.get("groups"), list)
    )
    if not structurally_current:
        return False
    splits = manifest.get("splits", [])
    groups = manifest.get("groups", [])
    split_ids = [str(split.get("split_id") or "") for split in splits if isinstance(split, dict)]
    group_ids = [str(group.get("rule_group_id") or "") for group in groups if isinstance(group, dict)]
    if (
        len(split_ids) != len(splits)
        or not all(split_ids)
        or len(set(split_ids)) != len(split_ids)
        or len(group_ids) != len(groups)
        or not all(group_ids)
        or len(set(group_ids)) != len(group_ids)
    ):
        return False
    group_by_id = {
        str(group.get("rule_group_id")): group
        for group in groups
    }
    split_by_id = {str(split.get("split_id")): split for split in splits}
    for split in splits:
        group_id = str(split.get("rule_group_id") or "")
        group = group_by_id.get(group_id)
        if group is None or split.get("split_id") not in (group.get("split_ids") or []):
            return False
        if split.get("slice_type") == "surface_provenance_guard_slice":
            trigger_splits = [
                split_by_id.get(str(trigger_split_id))
                for trigger_split_id in (split.get("preservation_trigger_split_ids") or [])
            ]
            if not any(
                trigger_split is not None
                and trigger_split.get("slice_type") != "surface_provenance_guard_slice"
                and split_is_reliable_rule_evidence(trigger_split)
                for trigger_split in trigger_splits
            ):
                return False
    for group_id, group in group_by_id.items():
        listed_split_ids = list(group.get("split_ids") or [])
        if len(set(listed_split_ids)) != len(listed_split_ids):
            return False
        owned_splits = [
            split
            for split in splits
            if str(split.get("rule_group_id") or "") == group_id
        ]
        if set(listed_split_ids) != {split.get("split_id") for split in owned_splits}:
            return False
        hard_count = sum(split_is_reliable_rule_evidence(split) for split in owned_splits)
        if (
            group.get("hard_assertion_count") != hard_count
            or group.get("skipped_split_count") != len(owned_splits) - hard_count
        ):
            return False
    return all(
        isinstance(split, dict)
        and (
            (
                split.get("hard_assertion") is False
                and bool(split.get("skip_reason") or split.get("diagnostic_only_reason"))
            )
            or (
                split.get("hard_assertion") is True
                and split_is_reliable_rule_evidence(split)
                and isinstance(split.get("variants", []), list)
                and all(
                    isinstance(variant, dict)
                    and variant.get("hard_assertion") is True
                    and variant.get("variant_id") not in (None, "")
                    and _variant_is_stable_for_split(dict(variant), split)[0]
                    for variant in split.get("variants", [])
                )
            )
        )
        for split in splits
    )


def compact_split_manifest(manifest: dict[str, Any], max_splits: int = 6) -> str:
    """Render a compact split/group summary for prompts."""
    if manifest.get("status") != "split_ok":
        return f"UT split unavailable: {manifest.get('error_type')}: {manifest.get('error')}"
    reliable_splits = [
        split
        for split in manifest.get("splits", [])
        if isinstance(split, dict) and split_is_reliable_rule_evidence(split)
    ]
    lines = [
        f"UT reliable splits: {len(reliable_splits)}; total splits: {len(manifest.get('splits', []))}",
    ]
    for split in reliable_splits[:max_splits]:
        lines.append(
            f"- {split.get('split_id')} group={split.get('rule_group_id')} "
            f"path={split.get('source_ast_path')} "
            f"hard_assertion={split.get('hard_assertion', True)}"
        )
        if split.get("diff_summary"):
            lines.append(f"  diff: {' '.join(split['diff_summary'][:6])}")
        if split.get("source_expr_sql") and split.get("expected_expr_sql"):
            source_expr = " ".join(str(split.get("source_expr_sql", "")).split())[:180]
            actual_expr = " ".join(str(split.get("actual_expr_sql", "")).split())[:180]
            expected_expr = " ".join(str(split.get("expected_expr_sql", "")).split())[:180]
            lines.append(f"  source: {source_expr}")
            lines.append(f"  actual: {actual_expr}")
            lines.append(f"  expected: {expected_expr}")
    return "\n".join(lines)
