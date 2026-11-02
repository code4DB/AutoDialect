"""Isolated split-generation worker for AutoDialect SQL UTs.

This file is executed as a subprocess by sql_ut_splitter.py and should remain a
subprocess boundary:
1. Input and SQLGlot isolation: read one stdin JSON payload, import the target
   package_root, and parse/transpile source, actual, and expected SQL.
2. Alignment and unit helpers: align SELECT trees/raw spans and construct
   complete parseable source/actual/expected SQL units.
3. Evidence and candidate helpers: derive diff anchors, boundary metadata,
   ancestor candidates, and variant candidates.
4. Split materialization and output: create projection/join/clause/fallback
   records and emit one sentinel-prefixed JSON result.

Main-process manifest schema/grouping/variant filtering belongs in
sql_ut_splitter.py. Prompt workset selection belongs in sql_ut_workset.py.
"""

import difflib
import json
import re
import sys


def emit(obj):
    print("__AUTODIALECT_JSON__:" + json.dumps(obj, ensure_ascii=False))


def main() -> None:
    """Build raw split records from one stdin payload in an isolated process.

    The worker aligns source, actual, and expected SQL trees/text spans, emits
    complete parseable units plus mutation candidates, and falls back to a
    parent unit when no changed child can be trusted. Every path emits exactly
    one sentinel-prefixed success or structured failure object; exceptions do
    not escape to the parent process.
    """
    try:
        # 1. Input and SQLGlot isolation.
        payload = json.loads(sys.stdin.read() or "{}")
        package_root = payload["package_root"]
        src_sql = payload["src_sql"]
        expected_sql = payload["expected_sql"]
        src_dialect = payload["src_dialect"]
        tgt_dialect = payload["tgt_dialect"]
        sys.path.insert(0, package_root)

        import sqlglot
        from sqlglot import exp
        from sqlglot.tokens import Tokenizer

        source_ast = sqlglot.parse_one(src_sql, read=src_dialect)
        try:
            actual_sql = sqlglot.transpile(src_sql, read=src_dialect, write=tgt_dialect)[0]
            actual_status = "ok"
        except Exception as exc:
            actual_sql = str(exc)
            actual_status = "error"

        expected_status = "ok"
        expected_error = None
        try:
            expected_ast = sqlglot.parse_one(expected_sql, read=tgt_dialect)
        except Exception as exc:
            expected_ast = None
            expected_status = "error"
            expected_error = "%s: %s" % (type(exc).__name__, exc)
        actual_ast = None
        if actual_status == "ok":
            try:
                actual_ast = sqlglot.parse_one(actual_sql, read=tgt_dialect)
            except Exception:
                actual_ast = None

        splits = []
        split_keys = set()
        residual_quote_split_keys = set()
        preservation_candidates = []
        split_limits = {"max": 96, "truncated": False}
        source_select_spans = []
        actual_select_spans = []
        expected_select_spans = []

        # 2. SQL/text alignment and complete-unit helpers.
        def normalized(sql):
            return "".join((sql or "").strip().rstrip(";").split()).lower()

        def diff_token_records(sql):
            normalized_sql = (sql or "").strip().rstrip(";")
            display_tokens = re.findall(
                r"`[^`]*`|\"[^\"]*\"|'(?:''|[^'])*'|[A-Za-z_][A-Za-z0-9_]*|"
                r"\d+\.\d+|\d+|<=|>=|<>|!=|::|[+\-*/%()]|[^\s]",
                normalized_sql,
            )
            records = []
            cursor = 0
            for token in display_tokens:
                key = re.sub(r"\s+", "", token).lower()
                records.append((key, token, cursor, cursor + len(key)))
                cursor += len(key)
            return records

        def overlapping_record_indexes(records, start, end):
            if start >= end:
                return []
            return [
                index
                for index, (_key, _display, token_start, token_end) in enumerate(records)
                if token_end > start and token_start < end
            ]

        def comparison_char_opcodes(actual_key, expected_key):
            prefix = 0
            shared_length = min(len(actual_key), len(expected_key))
            while prefix < shared_length and actual_key[prefix] == expected_key[prefix]:
                prefix += 1

            actual_end = len(actual_key)
            expected_end = len(expected_key)
            while (
                actual_end > prefix
                and expected_end > prefix
                and actual_key[actual_end - 1] == expected_key[expected_end - 1]
            ):
                actual_end -= 1
                expected_end -= 1

            matcher = difflib.SequenceMatcher(
                None,
                actual_key[prefix:actual_end],
                expected_key[prefix:expected_end],
                autojunk=False,
            )
            return [
                (tag, prefix + i1, prefix + i2, prefix + j1, prefix + j2)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                if tag != "equal"
            ]

        def changed_token_summary(actual_fragment, expected_fragment):
            if normalized(actual_fragment) == normalized(expected_fragment):
                return {
                    "actual_only": [],
                    "expected_only": [],
                    "actual_token_count": len(diff_token_records(actual_fragment)),
                    "expected_token_count": len(diff_token_records(expected_fragment)),
                }
            actual_records = diff_token_records(actual_fragment)
            expected_records = diff_token_records(expected_fragment)
            actual_key = "".join(record[0] for record in actual_records)
            expected_key = "".join(record[0] for record in expected_records)
            actual_only = []
            expected_only = []
            seen_actual = set()
            seen_expected = set()
            for _tag, i1, i2, j1, j2 in comparison_char_opcodes(actual_key, expected_key):
                for index in overlapping_record_indexes(actual_records, i1, i2):
                    if index not in seen_actual:
                        actual_only.append(actual_records[index][1])
                        seen_actual.add(index)
                for index in overlapping_record_indexes(expected_records, j1, j2):
                    if index not in seen_expected:
                        expected_only.append(expected_records[index][1])
                        seen_expected.add(index)
            return {
                "actual_only": actual_only[:12],
                "expected_only": expected_only[:12],
                "actual_token_count": len(actual_records),
                "expected_token_count": len(expected_records),
            }

        def quote_identifier(name, quote):
            if quote == "`":
                return "`%s`" % (name or "").replace("`", "``")
            if quote == '"':
                return '"%s"' % (name or "").replace('"', '""')
            return name or ""

        def raw_projection_alias(text):
            match = re.search(
                r"(?is)\bAS\s+(`((?:``|[^`])*)`|\"((?:\"\"|[^\"])*)\"|([A-Za-z_][A-Za-z0-9_$]*))\s*$",
                text or "",
            )
            if not match:
                return None
            token = match.group(1)
            if token.startswith("`"):
                name = (match.group(2) or "").replace("``", "`")
                quote = "`"
            elif token.startswith('"'):
                name = (match.group(3) or "").replace('""', '"')
                quote = '"'
            else:
                name = match.group(4) or ""
                quote = ""
            if not name:
                return None
            return {
                "name": name,
                "canonical_name": name.lower(),
                "quote": quote,
                "token": token,
            }

        def projection_without_explicit_as(text):
            """Normalize only a trailing projection-alias AS keyword.

            This is intentionally narrower than SQL normalization: CAST AS,
            table aliases, expression spelling, and alias quoting remain
            untouched.
            """
            return re.sub(
                r"(?is)\s+AS\s+(?P<alias>`(?:``|[^`])+`|\"(?:\"\"|[^\"])+\"|[A-Za-z_][A-Za-z0-9_$]*)\s*$",
                r" \g<alias>",
                text or "",
            )

        def projection_differs_only_by_alias_as(actual_text, expected_text):
            if normalized(actual_text) == normalized(expected_text):
                return False
            return normalized(projection_without_explicit_as(actual_text)) == normalized(
                projection_without_explicit_as(expected_text)
            )

        def node_sql(node, dialect):
            return node.sql(dialect=dialect) if node is not None else ""

        def node_complexity(node):
            if node is None:
                return {"node_count": 0, "function_count": 0, "has_subquery": False, "has_window": False}
            node_count = 0
            function_count = 0
            has_subquery = False
            has_window = False
            try:
                for child in node.walk():
                    node_count += 1
                    if isinstance(child, exp.Func):
                        function_count += 1
                    if isinstance(child, (exp.Subquery, exp.Select)) and child is not node:
                        has_subquery = True
                    if isinstance(child, exp.Window):
                        has_window = True
            except Exception:
                pass
            return {
                "node_count": node_count,
                "function_count": function_count,
                "has_subquery": has_subquery,
                "has_window": has_window,
            }

        def diff_anchor(source_node, actual_node, expected_node, source_span, *, cut_kind, lift_reason, actual_span=None, expected_span=None, target_diff_node=None, source_diff_node=None, ancestor_lift_score=None, target_diff_path=None, ancestor_coverage=None):
            actual_fragment = (actual_span or {}).get("text") or (node_sql(actual_node, tgt_dialect) if actual_node is not None else actual_sql)
            expected_fragment = (expected_span or {}).get("text") or (node_sql(expected_node, tgt_dialect) if expected_node is not None else expected_sql)
            token_summary = changed_token_summary(actual_fragment, expected_fragment)
            return {
                "strategy": "actual_expected_raw_boundary_diff_to_source_ast_cut",
                "cut_kind": cut_kind,
                "source_ast_node_type": source_node.__class__.__name__ if source_node is not None else None,
                "source_ast_sql": node_sql(source_node, src_dialect),
                "source_diff_node_type": source_diff_node.__class__.__name__ if source_diff_node is not None else None,
                "source_diff_node_sql": node_sql(source_diff_node, src_dialect) if source_diff_node is not None else None,
                "target_diff_node_type": target_diff_node.__class__.__name__ if target_diff_node is not None else None,
                "target_diff_node_sql": node_sql(target_diff_node, tgt_dialect) if target_diff_node is not None else None,
                "target_diff_node_path": target_diff_path,
                "source_text_span": source_span,
                "actual_text_span": actual_span,
                "expected_text_span": expected_span,
                "diff_basis": "actual_raw_boundary_text_vs_expected_raw_boundary_text" if actual_span or expected_span else "actual_expected_canonical_node_sql",
                "actual_changed_tokens": token_summary["actual_only"],
                "expected_changed_tokens": token_summary["expected_only"],
                "actual_token_count": token_summary["actual_token_count"],
                "expected_token_count": token_summary["expected_token_count"],
                "ancestor_lift": {
                    "lifted_to": "Select",
                    "reason": lift_reason,
                    "completion": "complete_parseable_sql_unit",
                    "score": ancestor_lift_score,
                    "coverage": ancestor_coverage,
                },
                "source_node_complexity": node_complexity(source_node),
            }

        def build_select_spans(sql_text, dialect):
            try:
                tokens = Tokenizer(dialect=dialect).tokenize(sql_text)
            except Exception:
                return []

            def token_text(token):
                return (getattr(token, "text", "") or "").upper()

            def token_type_name(token):
                token_type = getattr(token, "token_type", None)
                return getattr(token_type, "name", "") or ""

            def clause_kind(token):
                """Return the raw SELECT-boundary kind across SQLGlot token versions.

                Older SQLGlot versions tokenize ``GROUP BY`` and ``ORDER BY`` as
                single GROUP_BY / ORDER_BY tokens.  Comparing only ``token.text``
                with GROUP / ORDER silently loses the raw boundary and forces the
                splitter through AST serialization, which can change exact surface
                contracts such as an implicit projection alias.
                """
                token_type = token_type_name(token)
                text = token_text(token)
                if token_type == "GROUP_BY" or text in {"GROUP", "GROUP BY"}:
                    return "group"
                if token_type == "ORDER_BY" or text in {"ORDER", "ORDER BY"}:
                    return "order"
                if token_type == "WHERE" or text == "WHERE":
                    return "where"
                if token_type == "HAVING" or text == "HAVING":
                    return "having"
                if token_type == "QUALIFY" or text == "QUALIFY":
                    return "qualify"
                # FETCH is SQLGlot's source-side limit expression. Treat it as
                # the same raw boundary as LIMIT so ORDER does not absorb it.
                if token_type in {"LIMIT", "FETCH"} or text in {"LIMIT", "FETCH"}:
                    return "limit"
                return None

            def is_lparen(token):
                return token_text(token) == "(" or getattr(token, "token_type", None).name == "L_PAREN"

            def is_rparen(token):
                return token_text(token) == ")" or getattr(token, "token_type", None).name == "R_PAREN"

            def is_comma(token):
                return token_text(token) == "," or getattr(token, "token_type", None).name == "COMMA"

            def token_end(token):
                return getattr(token, "end", getattr(token, "start", 0)) + 1

            def trim_span(start, end):
                while start < end and sql_text[start].isspace():
                    start += 1
                while end > start and sql_text[end - 1].isspace():
                    end -= 1
                return {"start": start, "end": end, "text": sql_text[start:end]}

            depths = []
            depth = 0
            for token in tokens:
                if is_rparen(token):
                    depth = max(0, depth - 1)
                depths.append(depth)
                if is_lparen(token):
                    depth += 1

            select_spans = []
            for index, token in enumerate(tokens):
                if token_text(token) != "SELECT":
                    continue
                select_depth = depths[index]
                from_index = None
                end_index = len(tokens)
                projection_end_index = None
                for scan in range(index + 1, len(tokens)):
                    current = token_text(tokens[scan])
                    if depths[scan] < select_depth:
                        end_index = scan
                        break
                    if depths[scan] == select_depth and current in {"UNION", "EXCEPT", "INTERSECT"}:
                        end_index = scan
                        break
                    if depths[scan] == select_depth and current == "FROM":
                        from_index = scan
                        projection_end_index = scan
                        break
                if projection_end_index is None:
                    projection_end_index = end_index
                    for scan in range(index + 1, end_index):
                        current = token_text(tokens[scan])
                        if depths[scan] == select_depth and clause_kind(tokens[scan]):
                            projection_end_index = scan
                            break

                projection_spans = []
                projection_start = token_end(token)
                item_start = projection_start
                for scan in range(index + 1, projection_end_index):
                    if depths[scan] == select_depth and is_comma(tokens[scan]):
                        projection_spans.append(trim_span(item_start, getattr(tokens[scan], "start", item_start)))
                        item_start = token_end(tokens[scan])
                projection_spans.append(trim_span(item_start, getattr(tokens[projection_end_index], "start", len(sql_text)) if projection_end_index < len(tokens) else len(sql_text)))

                clause_scan_end = end_index
                if from_index is not None and end_index == len(tokens):
                    for scan in range(from_index + 1, len(tokens)):
                        current = token_text(tokens[scan])
                        if depths[scan] < select_depth or (depths[scan] == select_depth and current in {"UNION", "EXCEPT", "INTERSECT"}):
                            clause_scan_end = scan
                            break

                clauses = {}
                first_clause_index = clause_scan_end
                from_base_span = None
                from_join_span = None
                if from_index is not None:
                    scan = from_index + 1
                    while scan < clause_scan_end:
                        current = token_text(tokens[scan])
                        current_clause = clause_kind(tokens[scan])
                        if depths[scan] == select_depth and current_clause:
                            first_clause_index = min(first_clause_index, scan)
                            start = getattr(tokens[scan], "start", 0)
                            next_scan = scan + 1
                            while next_scan < clause_scan_end:
                                next_clause = clause_kind(tokens[next_scan])
                                if depths[next_scan] == select_depth and next_clause:
                                    break
                                next_scan += 1
                            end = getattr(tokens[next_scan], "start", len(sql_text)) if next_scan < len(tokens) else len(sql_text)
                            clauses[current_clause] = trim_span(start, end)
                            scan = next_scan
                            continue
                        scan += 1

                    from_join_end = getattr(tokens[first_clause_index], "start", len(sql_text)) if first_clause_index < len(tokens) else (
                        getattr(tokens[clause_scan_end], "start", len(sql_text)) if clause_scan_end < len(tokens) else len(sql_text)
                    )
                    from_base_end = from_join_end
                    for scan in range(from_index + 1, first_clause_index):
                        if depths[scan] == select_depth and token_text(tokens[scan]) == "JOIN":
                            from_base_end = getattr(tokens[scan], "start", from_join_end)
                            break
                    from_base_span = trim_span(getattr(tokens[from_index], "start", 0), from_base_end)
                    from_join_span = trim_span(getattr(tokens[from_index], "start", 0), from_join_end)

                select_spans.append({
                    "select_start": getattr(token, "start", 0),
                    "select_end": getattr(tokens[clause_scan_end], "start", len(sql_text)) if clause_scan_end < len(tokens) else len(sql_text),
                    "select_sql": sql_text[
                        getattr(token, "start", 0):
                        getattr(tokens[clause_scan_end], "start", len(sql_text)) if clause_scan_end < len(tokens) else len(sql_text)
                    ],
                    "from_base_span": from_base_span,
                    "from_join_span": from_join_span,
                    "projection_spans": projection_spans,
                    "clauses": clauses,
                })
            return select_spans

        def align_select_spans(ast_selects, raw_spans, dialect):
            span_selects = []
            for span in raw_spans:
                try:
                    parsed = sqlglot.parse_one(span.get("select_sql") or "", read=dialect)
                    parsed_selects = select_nodes(parsed)
                    span_selects.append(parsed_selects[0] if parsed_selects else None)
                except Exception:
                    span_selects.append(None)
            matches = match_selects(ast_selects, span_selects)
            aligned = [None] * len(ast_selects)
            for source_idx, (span_idx, _span_select, _score) in matches.items():
                if span_idx < len(raw_spans):
                    aligned[source_idx] = raw_spans[span_idx]
            return aligned

        def boundary_text_span(aligned_spans, select_ordinal, kind, index=None, *, source_name):
            if select_ordinal is None or select_ordinal >= len(aligned_spans):
                return None
            select_span = aligned_spans[select_ordinal]
            if not select_span:
                return None
            if kind == "projection":
                spans = select_span.get("projection_spans") or []
                if index is None or index >= len(spans):
                    return None
                span = dict(spans[index])
            else:
                span = dict((select_span.get("clauses") or {}).get(kind) or {})
            if not span:
                return None
            span.update({
                "kind": kind,
                "select_ordinal": select_ordinal,
                "source": source_name,
            })
            if index is not None:
                span["index"] = index
            return span

        def source_text_span(select_ordinal, kind, index=None):
            return boundary_text_span(source_select_spans, select_ordinal, kind, index, source_name="source_raw_token_boundary")

        def actual_text_span(select_ordinal, kind, index=None):
            return boundary_text_span(actual_select_spans, select_ordinal, kind, index, source_name="actual_raw_token_boundary")

        def expected_text_span(select_ordinal, kind, index=None):
            return boundary_text_span(expected_select_spans, select_ordinal, kind, index, source_name="expected_raw_token_boundary")

        def raw_completed_unit(
            aligned_spans,
            select_ordinal,
            kind,
            boundary_span,
            fallback_unit,
            dialect,
            origin_prefix,
            dependency_projection_indexes=None,
            dependency_projection_texts=None,
            context_free=False,
        ):
            if select_ordinal is None or select_ordinal >= len(aligned_spans):
                return fallback_unit, "%s_ast_canonical_fallback" % origin_prefix
            select_span = aligned_spans[select_ordinal]
            if not select_span or not boundary_span or not boundary_span.get("text"):
                return fallback_unit, "%s_ast_canonical_fallback" % origin_prefix
            from_base = (select_span.get("from_base_span") or {}).get("text")
            from_join = (select_span.get("from_join_span") or {}).get("text")
            clauses = select_span.get("clauses") or {}
            boundary_text = boundary_span.get("text")
            projection_spans = select_span.get("projection_spans") or []
            if dependency_projection_texts is not None:
                dependency_projections = list(dependency_projection_texts)
            else:
                dependency_projections = []
                for projection_index in dependency_projection_indexes or []:
                    if 0 <= projection_index < len(projection_spans):
                        projection_text = (projection_spans[projection_index] or {}).get("text")
                        if projection_text:
                            dependency_projections.append(projection_text)
            dependency_select_list = ", ".join(dependency_projections) or "1"
            if context_free and kind in ("where", "group", "order"):
                unit = "SELECT %s %s" % (dependency_select_list, boundary_text)
                if kind == "order":
                    limit_text = (clauses.get("limit") or {}).get("text")
                    if limit_text:
                        unit = "%s %s" % (unit, limit_text)
            elif kind == "projection":
                unit = "SELECT %s %s" % (boundary_text, from_base) if from_base else "SELECT %s" % boundary_text
                limit_text = (clauses.get("limit") or {}).get("text")
                if limit_text:
                    unit = "%s %s" % (unit, limit_text)
            elif kind in ("where", "qualify") and from_join:
                unit = "SELECT %s %s %s" % (dependency_select_list, from_join, boundary_text)
            elif kind == "group" and from_join:
                unit = "SELECT %s %s %s" % (dependency_select_list, from_join, boundary_text)
            elif kind == "having" and from_join:
                group_text = (clauses.get("group") or {}).get("text")
                unit = "SELECT %s %s %s %s" % (
                    dependency_select_list,
                    from_join,
                    group_text or "",
                    boundary_text,
                )
            elif kind == "order" and from_join:
                unit = "SELECT %s %s %s" % (dependency_select_list, from_join, boundary_text)
                limit_text = (clauses.get("limit") or {}).get("text")
                if limit_text:
                    unit = "%s %s" % (unit, limit_text)
            else:
                return fallback_unit, "%s_ast_canonical_fallback" % origin_prefix
            # Keep raw token text intact. In particular, global whitespace
            # collapsing would change string literal values such as 'x  y'.
            # SQLGlot accepts the original newlines/spacing in these units.
            unit = unit.strip()
            try:
                sqlglot.parse_one(unit, read=dialect)
                if origin_prefix == "expected":
                    unit = preserve_expected_identifier_quote(unit)
                completion = "raw_boundary_context_free_completed_unit" if context_free else "raw_boundary_completed_unit"
                return unit, "%s_%s" % (origin_prefix, completion)
            except Exception:
                return fallback_unit, "%s_ast_canonical_fallback_parse_failed" % origin_prefix

        def raw_source_unit(
            select_ordinal,
            kind,
            boundary_span,
            fallback_unit,
            dependency_projection_indexes=None,
            context_free=False,
        ):
            return raw_completed_unit(
                source_select_spans,
                select_ordinal,
                kind,
                boundary_span,
                fallback_unit,
                src_dialect,
                "source",
                dependency_projection_indexes,
                context_free=context_free,
            )

        def raw_actual_unit(
            actual_select_ordinal,
            kind,
            boundary_span,
            fallback_unit,
            dependency_projection_indexes=None,
            dependency_projection_texts=None,
            context_free=False,
        ):
            return raw_completed_unit(
                actual_select_spans,
                actual_select_ordinal,
                kind,
                boundary_span,
                fallback_unit,
                tgt_dialect,
                "actual",
                dependency_projection_indexes,
                dependency_projection_texts,
                context_free=context_free,
            )

        def raw_expected_unit(
            expected_select_ordinal,
            kind,
            boundary_span,
            fallback_unit,
            dependency_projection_indexes=None,
            context_free=False,
        ):
            return raw_completed_unit(
                expected_select_spans,
                expected_select_ordinal,
                kind,
                boundary_span,
                fallback_unit,
                tgt_dialect,
                "expected",
                dependency_projection_indexes,
                context_free=context_free,
            )

        def raw_projection_body_span(boundary_span, projection_expr, candidate_path):
            if not boundary_span or not boundary_span.get("text"):
                return None
            if not (
                isinstance(projection_expr, exp.Alias)
                and candidate_path
                and len(candidate_path) == 1
                and candidate_path[0][0] == "this"
                and candidate_path[0][1] is None
            ):
                return None
            text = boundary_span.get("text") or ""
            match = re.search(
                r"(?is)\s+AS\s+(`((?:``|[^`])*)`|\"((?:\"\"|[^\"])*)\"|([A-Za-z_][A-Za-z0-9_$]*))\s*$",
                text,
            )
            if not match:
                return None
            body_text = text[: match.start()].rstrip()
            if not body_text:
                return None
            span = dict(boundary_span)
            span["end"] = span.get("start", 0) + len(body_text)
            span["text"] = body_text
            span["source"] = "%s_projection_alias_body" % (boundary_span.get("source") or "raw")
            return span

        # 3. Diff evidence, matching, variant, and ancestor-candidate helpers.
        def four_route_metadata(
            source_node,
            actual_node,
            expected_node,
            source_span,
            actual_span,
            expected_span,
            source_unit,
            actual_unit,
            expected_unit,
            *,
            cut_kind,
            source_canonical_unit=None,
            actual_canonical_unit=None,
            expected_canonical_unit=None,
        ):
            return {
                "source_raw": {
                    "boundary_text": (source_span or {}).get("text"),
                    "span": source_span,
                    "unit_sql": source_unit,
                },
                "source_canonical": {
                    "unit_sql": source_canonical_unit or source_unit,
                    "anchor_node_sql": node_sql(source_node, src_dialect),
                },
                "actual": {
                    "raw_boundary_text": (actual_span or {}).get("text"),
                    "unit_sql": actual_unit,
                    "canonical_unit_sql": actual_canonical_unit or actual_unit,
                    "node_sql_canonical": node_sql(actual_node, tgt_dialect) if actual_node is not None else None,
                },
                "expected_raw": {
                    "boundary_text": (expected_span or {}).get("text"),
                    "unit_sql": expected_unit,
                },
                "expected_canonical": {
                    "unit_sql": expected_canonical_unit or expected_unit,
                    "node_sql": node_sql(expected_node, tgt_dialect) if expected_node is not None else None,
                },
                "target_diff": {
                    "basis": "actual_raw_boundary_text_vs_expected_raw_boundary_text" if actual_span or expected_span else "actual_expected_canonical_node_sql",
                    "cut_kind": cut_kind,
                    **changed_token_summary(
                        (actual_span or {}).get("text") or (node_sql(actual_node, tgt_dialect) if actual_node is not None else actual_unit),
                        (expected_span or {}).get("text") or (node_sql(expected_node, tgt_dialect) if expected_node is not None else expected_unit),
                    ),
                },
            }

        def selected_target_diff_metadata(actual_node, expected_node, source_node, actual_unit, expected_unit, source_unit, *, candidate_kind, candidate_path):
            actual_fragment = node_sql(actual_node, tgt_dialect) if actual_node is not None else actual_unit
            expected_fragment = node_sql(expected_node, tgt_dialect) if expected_node is not None else expected_unit
            return {
                "basis": "actual_expected_selected_ancestor_canonical_sql",
                "candidate_kind": candidate_kind,
                "candidate_path": candidate_path,
                "source_node_type": source_node.__class__.__name__ if source_node is not None else None,
                "actual_node_type": actual_node.__class__.__name__ if actual_node is not None else None,
                "expected_node_type": expected_node.__class__.__name__ if expected_node is not None else None,
                "source_node_sql": node_sql(source_node, src_dialect) if source_node is not None else source_unit,
                "actual_node_sql": actual_fragment,
                "expected_node_sql": expected_fragment,
                **changed_token_summary(actual_fragment, expected_fragment),
            }

        def preserve_expected_identifier_quote(sql):
            # The expected SQL text is the benchmark contract. Some SQLGlot target
            # generators round-trip quoted identifiers with a different quote character.
            # Only restore identifiers that are actually backtick-quoted in the parent
            # expected SQL; this avoids converting unrelated double-quoted text.
            if "`" not in expected_sql or '"' not in (sql or ""):
                return sql
            expected_backtick_names = {
                match.group(1).replace("``", "`")
                for match in re.finditer(r"`((?:``|[^`])*)`", expected_sql)
            }
            if not expected_backtick_names:
                return sql
            expected_backtick_names_by_lower = {name.lower(): name for name in expected_backtick_names}

            def replace_if_expected_identifier(match):
                raw_name = match.group(1).replace('""', '"')
                canonical_name = (
                    raw_name
                    if raw_name in expected_backtick_names
                    else expected_backtick_names_by_lower.get(raw_name.lower())
                )
                if canonical_name is None:
                    return match.group(0)
                return "`%s`" % canonical_name.replace("`", "``")

            return re.sub(r'"((?:""|[^"\n])*)"', replace_if_expected_identifier, sql)

        def changed_from_expected(actual_node, expected_node):
            if expected_node is None:
                return False
            if actual_node is None:
                return True
            return normalized(node_sql(actual_node, tgt_dialect)) != normalized(node_sql(expected_node, tgt_dialect))

        def alias_body(node):
            if isinstance(node, exp.Alias):
                return node.this
            return node

        def select_nodes(ast):
            if ast is None:
                return []
            return [node for node in ast.walk() if isinstance(node, exp.Select)]

        def select_signature(select):
            if select is None:
                return {}
            tables = []
            try:
                direct_tables = set()
                for table in select.find_all(exp.Table):
                    parent = table.parent
                    nearest_select = None
                    while parent is not None:
                        if isinstance(parent, exp.Select):
                            nearest_select = parent
                            break
                        parent = parent.parent
                    if nearest_select is select and table.name:
                        direct_tables.add(table.name.lower())
                tables = sorted(direct_tables)
            except Exception:
                tables = []
            return {
                "projection_count": len(list(select.expressions or [])),
                "tables": tables,
                "has_where": bool(select.args.get("where")),
                "has_group": bool(select.args.get("group")),
                "has_having": bool(select.args.get("having")),
                "has_order": bool(select.args.get("order")),
                "join_count": len(list(select.args.get("joins") or [])),
            }

        def select_score(left, right):
            if left is None or right is None:
                return -1
            left_sig = select_signature(left)
            right_sig = select_signature(right)
            score = 0
            score += max(0, 8 - abs(left_sig["projection_count"] - right_sig["projection_count"]))
            score += max(0, 4 - abs(left_sig["join_count"] - right_sig["join_count"]))
            for key in ("has_where", "has_group", "has_having", "has_order"):
                if left_sig[key] == right_sig[key]:
                    score += 2
            common_tables = set(left_sig["tables"]) & set(right_sig["tables"])
            score += min(8, len(common_tables) * 4)
            if left_sig["tables"] and right_sig["tables"] and not common_tables:
                score -= 8
            return score

        def match_selects(source_selects, other_selects):
            matched = {}
            used = set()
            for source_idx, source_select in enumerate(source_selects):
                best_idx = None
                best_score = -1
                for other_idx, other_select in enumerate(other_selects):
                    if other_idx in used:
                        continue
                    score = select_score(source_select, other_select)
                    if source_idx == other_idx:
                        score += 3
                    if score > best_score:
                        best_idx = other_idx
                        best_score = score
                if best_idx is not None and best_score >= 4:
                    matched[source_idx] = (best_idx, other_selects[best_idx], best_score)
                    used.add(best_idx)
            return matched

        def select_path(select_ordinal, suffix=None):
            base = "Select[%s]" % select_ordinal
            return "%s.%s" % (base, suffix) if suffix else base

        def alias_name(expr):
            try:
                return expr.alias
            except Exception:
                return None

        def identifier_names(node):
            names = set()
            if node is None:
                return names
            try:
                for column in node.find_all(exp.Column):
                    if column.name:
                        names.add(column.name.lower())
            except Exception:
                pass
            try:
                for ident in node.find_all(exp.Identifier):
                    if ident.name:
                        names.add(ident.name.lower())
            except Exception:
                pass
            return names

        def context_expression_indexes(select, context_node, context_kind=None):
            # Keep projections needed by alias-sensitive clauses such as ORDER/QUALIFY.
            names = identifier_names(context_node)
            expressions = list(select.expressions or [])
            selected = []
            for index, expr in enumerate(expressions):
                # A bare column's ``alias`` property is its own name in some
                # SQLGlot versions. It is not an alias dependency and retaining
                # it duplicates unrelated projection context.
                alias = alias_name(expr) if isinstance(expr, exp.Alias) else None
                if alias and alias.lower() in names:
                    selected.append(index)
            # Positional references are only direct GROUP BY / ORDER BY numeric
            # expressions. Numeric literals nested in CASE predicates or
            # function arguments are values, not projection ordinals.
            if context_kind in {"group", "order"}:
                try:
                    for item in list(context_node.expressions or []):
                        ordinal_expr = item.this if isinstance(item, exp.Ordered) else item
                        if (
                            isinstance(ordinal_expr, exp.Literal)
                            and not ordinal_expr.is_string
                            and str(ordinal_expr.this).isdigit()
                        ):
                            ordinal = int(ordinal_expr.this)
                            if 1 <= ordinal <= len(expressions):
                                selected.append(ordinal - 1)
                except Exception:
                    pass
            return list(dict.fromkeys(selected))

        def context_expressions(select, context_node, context_kind=None):
            expressions = list(select.expressions or [])
            return [
                expressions[index].copy()
                for index in context_expression_indexes(select, context_node, context_kind=context_kind)
            ]

        def dependency_projection_texts(select_spans, select_ordinal, projection_indexes):
            if select_ordinal is None or select_ordinal >= len(select_spans):
                return None
            projection_spans = (select_spans[select_ordinal] or {}).get("projection_spans") or []
            texts = []
            for projection_index in projection_indexes or []:
                if projection_index < 0 or projection_index >= len(projection_spans):
                    return None
                text = (projection_spans[projection_index] or {}).get("text")
                if not text:
                    return None
                texts.append(text)
            return texts

        def harmonized_actual_dependency_projections(
            actual_select_ordinal,
            actual_projection_indexes,
            expected_select_ordinal,
            expected_projection_indexes,
        ):
            """Remove unrelated projection-alias AS drift from clause units.

            Clause units retain projections only as dependencies (for example,
            an ORDER BY alias). If the current and gold dependency expressions
            are otherwise exactly equal, use the gold AS style so repairing the
            selected clause cannot remain blocked by an unrelated surface rule.
            Parent SQL validation still checks that AS contract exactly.
            """
            actual_texts = dependency_projection_texts(
                actual_select_spans,
                actual_select_ordinal,
                actual_projection_indexes,
            )
            expected_texts = dependency_projection_texts(
                expected_select_spans,
                expected_select_ordinal,
                expected_projection_indexes,
            )
            if actual_texts is None or expected_texts is None or len(actual_texts) != len(expected_texts):
                return None, []
            harmonized = list(actual_texts)
            normalizations = []
            for index, (actual_text, expected_text) in enumerate(zip(actual_texts, expected_texts)):
                if not projection_differs_only_by_alias_as(actual_text, expected_text):
                    continue
                harmonized[index] = expected_text
                normalizations.append({
                    "kind": "projection_alias_as_dependency_context",
                    "dependency_index": index,
                    "actual_text": actual_text,
                    "replacement_text": expected_text,
                })
            return (harmonized if normalizations else None), normalizations

        def cte_aliases(root):
            names = set()
            if root is None:
                return names
            try:
                for cte in root.find_all(exp.CTE):
                    alias = cte.alias
                    if alias:
                        names.add(alias.lower())
            except Exception:
                pass
            return names

        def table_names(node):
            names = set()
            if node is None:
                return names
            try:
                for table in node.find_all(exp.Table):
                    if table.name:
                        names.add(table.name.lower())
            except Exception:
                pass
            return names

        def attach_required_with(sliced, root):
            root_with = root.args.get("with") if root is not None else None
            if root_with is None or sliced.args.get("with") is not None:
                return
            if table_names(sliced) & cte_aliases(root):
                sliced.set("with", root_with.copy())

        def completed_select(select, dialect, *, root=None, expression=None, keep=None, join_index=None, context_node=None, context_kind=None):
            # Build a complete, parseable unit around a source-AST anchored cut node.
            if select is None:
                return actual_sql
            keep = set(keep or [])
            sliced = select.copy()
            if expression is not None:
                sliced.set("expressions", [expression.copy()])
            else:
                needed = context_expressions(select, context_node, context_kind=context_kind)
                sliced.set("expressions", needed or [exp.Literal.number(1)])

            for clause in ("where", "group", "having", "order", "qualify"):
                if clause not in keep:
                    sliced.set(clause, None)

            if join_index is None:
                if "joins" not in keep:
                    sliced.set("joins", [])
            else:
                sliced.set("joins", list(sliced.args.get("joins") or [])[: join_index + 1])
            attach_required_with(sliced, root)
            sql = sliced.sql(dialect=dialect)
            if dialect == tgt_dialect and root is expected_ast:
                return preserve_expected_identifier_quote(sql)
            return sql

        def transpile_source_unit(source_unit):
            try:
                return sqlglot.transpile(source_unit, read=src_dialect, write=tgt_dialect)[0]
            except Exception:
                return actual_sql

        select_one_from_re = re.compile(r"^\s*SELECT\s+1\s+FROM\b", re.IGNORECASE)

        def uses_select_one_from(unit):
            masked = re.sub(r"'(?:''|[^'])*'", "'__STR__'", unit or "")
            return bool(select_one_from_re.search(masked))

        def select_one_projection_unit(unit, dialect):
            try:
                parsed = sqlglot.parse_one(unit, read=dialect)
                select = parsed if isinstance(parsed, exp.Select) else next(parsed.find_all(exp.Select), None)
                if select is None:
                    return None
                select.set("expressions", [exp.Literal.number(1)])
                return parsed.sql(dialect=dialect)
            except Exception:
                return None

        def align_select_one_projection_units(
            source_unit,
            actual_unit,
            expected_unit,
            *,
            source_origin,
            actual_origin,
            expected_origin,
        ):
            """Make clause/join placeholder projections symmetric when possible."""
            select_one_states = [
                uses_select_one_from(unit)
                for unit in (source_unit, actual_unit, expected_unit)
            ]
            if all(select_one_states) or not any(select_one_states):
                return source_unit, actual_unit, expected_unit, source_origin, actual_origin, expected_origin
            aligned_source = select_one_projection_unit(source_unit, src_dialect)
            aligned_actual = select_one_projection_unit(actual_unit, tgt_dialect)
            aligned_expected = select_one_projection_unit(expected_unit, tgt_dialect)
            if not aligned_source or not aligned_actual or not aligned_expected:
                return source_unit, actual_unit, expected_unit, source_origin, actual_origin, expected_origin
            return (
                aligned_source,
                aligned_actual,
                aligned_expected,
                "%s_ast_canonical_select_one_projection_aligned" % (source_origin or "source_unit"),
                "%s_ast_canonical_select_one_projection_aligned" % (actual_origin or "actual_unit"),
                "%s_ast_canonical_select_one_projection_aligned" % (expected_origin or "expected_unit"),
            )

        def add_split(item):
            if len(splits) >= split_limits["max"]:
                split_limits["truncated"] = True
                return
            key = (
                item.get("source_ast_path"),
                item.get("target_anchor"),
                normalized(item.get("source_sql")),
                normalized(item.get("expected_sql")),
            )
            if key in split_keys:
                return
            item.setdefault("hard_assertion", True)
            for side, span_key, expr_key, origin_key in (
                ("source", "source_text_span", "source_expr_sql", "source_sql_origin"),
                ("actual", "actual_text_span", "actual_expr_sql", "actual_sql_origin"),
                ("expected", "expected_text_span", "expected_expr_sql", "expected_unit_source"),
            ):
                span = item.get(span_key) if isinstance(item.get(span_key), dict) else {}
                origin = str(item.get(origin_key) or "")
                raw_text = span.get("text") if "raw_boundary" in origin else None
                unit_sql = item.get("%s_sql" % side) or ""
                if raw_text and normalized(raw_text) not in normalized(unit_sql):
                    raw_projection_body = re.sub(
                        r"(?is)\s+AS\s+(?:`(?:``|[^`])+`|\"(?:\"\"|[^\"])+\"|[A-Za-z_][A-Za-z0-9_$]*)\s*$",
                        "",
                        raw_text,
                    ).rstrip()
                    raw_text = (
                        raw_projection_body
                        if raw_projection_body != span.get("text")
                        and normalized(raw_projection_body) in normalized(unit_sql)
                        else None
                    )
                item.setdefault(
                    "%s_rule_fragment_sql" % side,
                    raw_text or item.get(expr_key) or unit_sql,
                )
            try:
                sqlglot.parse_one(item["source_sql"], read=src_dialect)
                sqlglot.parse_one(item["expected_sql"], read=tgt_dialect)
            except Exception as exc:
                item["parse_warning"] = "%s: %s" % (type(exc).__name__, exc)
                item["hard_assertion"] = False
                item["skip_reason"] = item["parse_warning"]
            split_keys.add(key)
            splits.append(item)

        def expr_score(left, right, left_idx, right_idx):
            if left is None or right is None:
                return -1
            score = 0
            if left_idx == right_idx:
                score += 3
            if alias_name(left) and alias_name(left) == alias_name(right):
                score += 5
            if alias_body(left).__class__.__name__ == alias_body(right).__class__.__name__:
                score += 3
            common = identifier_names(left) & identifier_names(right)
            score += min(6, len(common) * 2)
            if normalized(node_sql(alias_body(left), src_dialect)) == normalized(node_sql(alias_body(right), tgt_dialect)):
                score += 4
            return score

        def match_expressions(source_exprs, expected_exprs):
            pairs = []
            used = set()
            for source_idx, source_expr in enumerate(source_exprs):
                best_idx = None
                best_score = -1
                for expected_idx, expected_expr in enumerate(expected_exprs):
                    if expected_idx in used:
                        continue
                    score = expr_score(source_expr, expected_expr, source_idx, expected_idx)
                    if score > best_score:
                        best_idx = expected_idx
                        best_score = score
                if best_idx is not None and best_score >= 3:
                    pairs.append((source_idx, best_idx, source_expr, expected_exprs[best_idx], best_score))
                    used.add(best_idx)
            return pairs

        def alias_variant(source_select, expected_select, source_expr, expected_expr, select_ordinal, idx):
            if source_expr.args.get("alias") is None or expected_expr.args.get("alias") is None:
                return []
            variant_alias = "autodialect_variant_alias"
            variants = []

            source_variant_expr = source_expr.copy()
            expected_variant_expr = expected_expr.copy()
            source_variant_expr.set("alias", exp.to_identifier(variant_alias))
            expected_variant_expr.set("alias", exp.to_identifier(variant_alias))
            variants.append({
                "variant_id": "alias_rename_unquoted_%s_%s" % (select_ordinal, idx),
                "mutation_type": "rename_projection_alias_unquoted",
                "source_sql": completed_select(source_select, src_dialect, root=source_ast, expression=source_variant_expr),
                "expected_sql": completed_select(expected_select, tgt_dialect, root=expected_ast, expression=expected_variant_expr),
                "hard_assertion": True,
            })

            source_quoted_expr = source_expr.copy()
            expected_quoted_expr = expected_expr.copy()
            source_quoted_expr.set("alias", exp.to_identifier(variant_alias, quoted=True))
            expected_quoted_expr.set("alias", exp.to_identifier(variant_alias, quoted=True))
            variants.append({
                "variant_id": "alias_rename_quoted_%s_%s" % (select_ordinal, idx),
                "mutation_type": "rename_projection_alias_quoted",
                "source_sql": completed_select(source_select, src_dialect, root=source_ast, expression=source_quoted_expr),
                "expected_sql": completed_select(expected_select, tgt_dialect, root=expected_ast, expression=expected_quoted_expr),
                "hard_assertion": True,
            })
            return variants

        def first_column(node):
            if node is None:
                return None
            try:
                return next(node.find_all(exp.Column), None)
            except Exception:
                return None

        def first_numeric_literal(node):
            if node is None:
                return None
            try:
                for literal in node.find_all(exp.Literal):
                    if not literal.is_string:
                        return literal
            except Exception:
                return None
            return None

        def matching_columns(node, template_column):
            if node is None or template_column is None:
                return []
            template_name = template_column.name
            template_table = template_column.table
            try:
                return [
                    column
                    for column in node.find_all(exp.Column)
                    if column.name == template_name and (not template_table or column.table == template_table)
                ]
            except Exception:
                return []

        def column_rename_variant(source_select, expected_select, source_expr, expected_expr, select_ordinal, idx):
            source_variant_expr = source_expr.copy()
            expected_variant_expr = expected_expr.copy()
            source_column = first_column(source_variant_expr)
            expected_column = first_column(expected_variant_expr)
            if source_column is None or expected_column is None:
                return []
            variant_name = "autodialect_variant_col"
            try:
                source_columns = matching_columns(source_variant_expr, source_column)
                expected_columns = matching_columns(expected_variant_expr, expected_column)
                if not source_columns or not expected_columns:
                    return []
                for column in source_columns:
                    column.set("this", exp.to_identifier(variant_name))
                for column in expected_columns:
                    column.set("this", exp.to_identifier(variant_name))
            except Exception:
                return []
            return [{
                "variant_id": "column_rename_%s_%s" % (select_ordinal, idx),
                "mutation_type": "rename_first_column_reference",
                "source_sql": completed_select(source_select, src_dialect, root=source_ast, expression=source_variant_expr),
                "expected_sql": completed_select(expected_select, tgt_dialect, root=expected_ast, expression=expected_variant_expr),
            }]

        def numeric_literal_variant(source_select, expected_select, source_expr, expected_expr, select_ordinal, idx):
            source_variant_expr = source_expr.copy()
            expected_variant_expr = expected_expr.copy()
            source_literal = first_numeric_literal(source_variant_expr)
            expected_literal = first_numeric_literal(expected_variant_expr)
            if source_literal is None or expected_literal is None:
                return []
            try:
                source_literal.replace(exp.Literal.number(7))
                expected_literal.replace(exp.Literal.number(7))
            except Exception:
                return []
            return [{
                "variant_id": "numeric_literal_%s_%s" % (select_ordinal, idx),
                "mutation_type": "replace_first_numeric_literal",
                "source_sql": completed_select(source_select, src_dialect, root=source_ast, expression=source_variant_expr),
                "expected_sql": completed_select(expected_select, tgt_dialect, root=expected_ast, expression=expected_variant_expr),
            }]

        def projection_variants(source_select, expected_select, source_expr, expected_expr, select_ordinal, idx):
            variants = []
            seen = set()
            for variant in (
                alias_variant(source_select, expected_select, source_expr, expected_expr, select_ordinal, idx)
                + column_rename_variant(source_select, expected_select, source_expr, expected_expr, select_ordinal, idx)
                + numeric_literal_variant(source_select, expected_select, source_expr, expected_expr, select_ordinal, idx)
            ):
                key = (variant.get("source_sql"), variant.get("expected_sql"))
                if key in seen:
                    continue
                seen.add(key)
                variants.append(variant)
            return variants

        def combined_projection_variants(source_select, expected_select, selected_source_expr, selected_expected_expr, select_ordinal, idx, *, selected_candidate_kind=None):
            variants = []
            seen = set()
            if selected_candidate_kind == "expression":
                variant_sources = projection_variants(
                    source_select,
                    expected_select,
                    selected_source_expr,
                    selected_expected_expr,
                    select_ordinal,
                    idx,
                )
            else:
                variant_sources = []
            for variant in variant_sources:
                key = (variant.get("variant_id"), variant.get("source_sql"), variant.get("expected_sql"))
                if key in seen:
                    continue
                seen.add(key)
                variants.append(variant)
            return variants

        def best_target_diff_node(actual_node, expected_node, source_node):
            if expected_node is None:
                return source_node, actual_node, expected_node, [], 0
            if actual_node is None:
                return source_node, actual_node, expected_node, [], 0
            if normalized(node_sql(actual_node, tgt_dialect)) == normalized(node_sql(expected_node, tgt_dialect)):
                return source_node, actual_node, expected_node, [], 0
            if actual_node.__class__ is not expected_node.__class__:
                return source_node, actual_node, expected_node, [], 0

            best = (source_node, actual_node, expected_node, [], 0)
            best_size = len(node_sql(expected_node, tgt_dialect))
            changed_candidates = []
            for arg_key, expected_value in expected_node.args.items():
                actual_value = actual_node.args.get(arg_key)
                source_value = source_node.args.get(arg_key) if source_node is not None else None
                if isinstance(expected_value, list) and isinstance(actual_value, list):
                    for index, expected_child in enumerate(expected_value):
                        if index >= len(actual_value) or not isinstance(expected_child, exp.Expression):
                            continue
                        source_child = source_value[index] if isinstance(source_value, list) and index < len(source_value) else None
                        changed_candidates.append((arg_key, index, actual_value[index], expected_child, source_child))
                elif isinstance(expected_value, exp.Expression) and isinstance(actual_value, exp.Expression):
                    source_child = source_value if isinstance(source_value, exp.Expression) else None
                    changed_candidates.append((arg_key, None, actual_value, expected_value, source_child))

            changed_candidates = [
                candidate for candidate in changed_candidates
                if normalized(node_sql(candidate[2], tgt_dialect)) != normalized(node_sql(candidate[3], tgt_dialect))
            ]
            non_alias_candidates = [candidate for candidate in changed_candidates if candidate[0] != "alias"]
            if non_alias_candidates:
                changed_candidates = non_alias_candidates
            changed_candidates.sort(key=lambda candidate: 0 if candidate[0] == "this" else 1)

            for arg_key, index, actual_child, expected_child, source_child in changed_candidates:
                child_source, child_actual, child_expected, child_path, child_depth = best_target_diff_node(
                    actual_child,
                    expected_child,
                    source_child,
                )
                child_size = len(node_sql(child_expected, tgt_dialect)) if child_expected is not None else best_size
                if child_size < best_size:
                    best = (child_source, child_actual, child_expected, [(arg_key, index)] + child_path, child_depth + 1)
                    best_size = child_size
            return best

        def format_arg_path(path):
            parts = []
            for arg_key, index in path or []:
                parts.append("%s[%s]" % (arg_key, index) if index is not None else arg_key)
            return ".".join(parts) if parts else None

        def node_at_arg_path(node, path):
            current = node
            for arg_key, index in path or []:
                if current is None:
                    return None
                value = current.args.get(arg_key)
                if isinstance(value, list):
                    if index is None or index >= len(value):
                        return None
                    current = value[index]
                else:
                    if index not in (None, 0):
                        return None
                    current = value
            return current

        def is_trivial_expression(node):
            return node is None or isinstance(node, (exp.Identifier, exp.Literal, exp.Column))

        def parseable_unit(sql, dialect):
            try:
                sqlglot.parse_one(sql, read=dialect)
                return True
            except Exception:
                return False

        def ancestor_coverage(candidate_path, target_diff_path):
            diff_path = list(target_diff_path or [])
            if candidate_path is None:
                return {
                    "covers_target_diff": True,
                    "coverage_kind": "boundary_unit",
                    "candidate_path": None,
                    "target_diff_path": format_arg_path(diff_path),
                }
            candidate = list(candidate_path or [])
            covers = bool(candidate) and diff_path[: len(candidate)] == candidate
            return {
                "covers_target_diff": covers,
                "coverage_kind": "ancestor_prefix" if covers else "not_ancestor_prefix",
                "candidate_path": format_arg_path(candidate),
                "target_diff_path": format_arg_path(diff_path),
            }

        def aliased_probe_expression(node):
            if node is None:
                return None
            # Expression-level UTs should not introduce a synthetic alias because
            # target generators may quote aliases differently, creating validation
            # noise unrelated to the rule being learned.
            return node.copy()

        def expression_select_unit(select, dialect, node, *, root):
            probe = aliased_probe_expression(node)
            if probe is None:
                return None
            return completed_select(select, dialect, root=root, expression=probe)

        def ancestor_lift_score(source_unit, expected_unit, source_span, actual_span, expected_span, diff_depth, *, dependency_merged, candidate_depth=0, candidate_kind="boundary", parseable=True):
            score = 0
            if parseable:
                score += 20
            if source_span:
                score += 3
            if actual_span and expected_span:
                score += 4
            if source_unit and expected_unit:
                score += 2
            if diff_depth:
                score += min(4, diff_depth)
            if candidate_depth:
                score += min(8, candidate_depth * 2)
            if candidate_kind == "expression":
                score += 8
            if dependency_merged:
                score += 1
            if source_unit and len(source_unit) < max(len(src_sql), 1):
                score += 1 + min(6, max(0, len(src_sql) - len(source_unit)) // 24)
            return score

        def raw_unit_origin(origin, prefix):
            return bool(origin and origin.startswith("%s_raw" % prefix))

        def raw_boundary_has_target_diff(actual_span, expected_span):
            if not actual_span or not expected_span:
                return False
            summary = changed_token_summary(actual_span.get("text"), expected_span.get("text"))
            return bool(summary["actual_only"] or summary["expected_only"])

        def raw_projection_alias_name_matches(actual_span, expected_span):
            actual_alias = raw_projection_alias((actual_span or {}).get("text"))
            expected_alias = raw_projection_alias((expected_span or {}).get("text"))
            return bool(
                actual_alias
                and expected_alias
                and actual_alias.get("canonical_name") == expected_alias.get("canonical_name")
            )

        def raw_boundary_contract_bonus(candidate_kind, source_origin, expected_origin, actual_span, expected_span, parseable):
            if candidate_kind != "projection_boundary":
                return 0
            if not parseable:
                return 0
            if not (raw_unit_origin(source_origin, "source") and raw_unit_origin(expected_origin, "expected")):
                return 0
            if raw_projection_alias_name_matches(actual_span, expected_span):
                return 0
            if not raw_boundary_has_target_diff(actual_span, expected_span):
                return 0
            return 16

        def projection_ancestor_candidates(
            source_select,
            expected_select,
            actual_select,
            source_expr,
            expected_expr,
            actual_expr,
            target_diff_path,
            *,
            select_ordinal,
            expected_select_ordinal,
            actual_select_ordinal,
            source_unit,
            expected_unit,
            actual_unit,
            source_canonical_unit,
            expected_canonical_unit,
            actual_canonical_unit,
            source_unit_source,
            actual_unit_source,
            source_span,
            actual_span,
            expected_span,
            expected_unit_source,
            diff_depth,
        ):
            candidates = []
            seen = set()
            path = list(target_diff_path or [])
            for cut_len in range(len(path), 0, -1):
                prefix = path[:cut_len]
                coverage = ancestor_coverage(prefix, path)
                if not coverage["covers_target_diff"]:
                    continue
                source_node = node_at_arg_path(source_expr, prefix)
                expected_node = node_at_arg_path(expected_expr, prefix)
                actual_node = node_at_arg_path(actual_expr, prefix) if actual_expr is not None else None
                if is_trivial_expression(source_node) or is_trivial_expression(expected_node):
                    continue
                if not changed_from_expected(actual_node, expected_node):
                    continue
                candidate_source_canonical_unit = expression_select_unit(source_select, src_dialect, source_node, root=source_ast)
                candidate_expected_canonical_unit = expression_select_unit(expected_select, tgt_dialect, expected_node, root=expected_ast)
                candidate_actual_canonical_unit = expression_select_unit(actual_select, tgt_dialect, actual_node, root=actual_ast) if actual_select is not None and actual_node is not None else actual_sql
                if not candidate_source_canonical_unit or not candidate_expected_canonical_unit:
                    continue
                candidate_source_span = raw_projection_body_span(source_span, source_expr, prefix)
                candidate_expected_span = raw_projection_body_span(expected_span, expected_expr, prefix)
                candidate_actual_span = raw_projection_body_span(actual_span, actual_expr, prefix)
                candidate_source_unit, candidate_source_unit_source = raw_source_unit(
                    select_ordinal,
                    "projection",
                    candidate_source_span,
                    candidate_source_canonical_unit,
                )
                candidate_expected_unit, candidate_expected_unit_source = raw_expected_unit(
                    expected_select_ordinal,
                    "projection",
                    candidate_expected_span,
                    candidate_expected_canonical_unit,
                )
                candidate_actual_unit, candidate_actual_unit_source = raw_actual_unit(
                    actual_select_ordinal,
                    "projection",
                    candidate_actual_span,
                    candidate_actual_canonical_unit,
                )
                parseable = parseable_unit(candidate_source_unit, src_dialect) and parseable_unit(candidate_expected_unit, tgt_dialect)
                key = (normalized(candidate_source_unit), normalized(candidate_expected_unit))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "candidate_kind": "expression",
                    "candidate_path": format_arg_path(prefix),
                    "source_node": source_node,
                    "expected_node": expected_node,
                    "actual_node": actual_node,
                    "source_sql": candidate_source_unit,
                    "expected_sql": candidate_expected_unit,
                    "actual_sql": candidate_actual_unit,
                    "source_canonical_sql": candidate_source_canonical_unit,
                    "expected_canonical_sql": candidate_expected_canonical_unit,
                    "actual_canonical_sql": candidate_actual_canonical_unit,
                    "source_sql_origin": candidate_source_unit_source,
                    "expected_unit_source": candidate_expected_unit_source,
                    "actual_sql_origin": candidate_actual_unit_source,
                    "source_text_span": source_span,
                    "actual_text_span": actual_span,
                    "expected_text_span": expected_span,
                    "unit_completion": "expression_select_context",
                    "dependency_merged": False,
                    "parseable": parseable,
                    "ancestor_coverage": coverage,
                    "score": ancestor_lift_score(
                        candidate_source_unit,
                        candidate_expected_unit,
                        candidate_source_span,
                        candidate_actual_span,
                        candidate_expected_span,
                        diff_depth,
                        dependency_merged=False,
                        candidate_depth=cut_len,
                        candidate_kind="expression",
                        parseable=parseable,
                    ),
                })

            boundary_parseable = parseable_unit(source_unit, src_dialect) and parseable_unit(expected_unit, tgt_dialect)
            boundary_score = ancestor_lift_score(
                source_unit,
                expected_unit,
                source_span,
                actual_span,
                expected_span,
                diff_depth,
                dependency_merged=False,
                candidate_kind="projection_boundary",
                parseable=boundary_parseable,
            )
            boundary_score += raw_boundary_contract_bonus(
                "projection_boundary",
                source_unit_source,
                expected_unit_source,
                actual_span,
                expected_span,
                boundary_parseable,
            )
            candidates.append({
                "candidate_kind": "projection_boundary",
                "candidate_path": None,
                "source_node": source_expr,
                "expected_node": expected_expr,
                "actual_node": actual_expr,
                "source_sql": source_unit,
                "expected_sql": expected_unit,
                "actual_sql": actual_unit,
                "source_canonical_sql": source_canonical_unit,
                "expected_canonical_sql": expected_canonical_unit,
                "actual_canonical_sql": actual_canonical_unit,
                "source_sql_origin": source_unit_source,
                "expected_unit_source": expected_unit_source,
                "actual_sql_origin": actual_unit_source,
                "source_text_span": source_span,
                "actual_text_span": actual_span,
                "expected_text_span": expected_span,
                "unit_completion": "select_from_join_context",
                "dependency_merged": False,
                "parseable": boundary_parseable,
                "ancestor_coverage": ancestor_coverage(None, path),
                "score": boundary_score,
            })
            candidates.sort(key=lambda item: (item["parseable"], item["score"], -len(item["source_sql"] or "")), reverse=True)
            return candidates

        def clause_ancestor_candidates(
            source_select,
            expected_select,
            actual_select,
            source_node,
            expected_node,
            actual_node,
            target_diff_path,
            *,
            source_unit,
            expected_unit,
            actual_unit,
            source_canonical_unit,
            expected_canonical_unit,
            actual_canonical_unit,
            source_unit_source,
            actual_unit_source,
            source_span,
            actual_span,
            expected_span,
            expected_unit_source,
            diff_depth,
            dependency_merged,
            unit_completion,
            allow_expression_candidates,
        ):
            candidates = []
            seen = set()
            path = list(target_diff_path or [])
            candidate_cut_lengths = range(len(path), 0, -1) if allow_expression_candidates else []
            for cut_len in candidate_cut_lengths:
                prefix = path[:cut_len]
                coverage = ancestor_coverage(prefix, path)
                if not coverage["covers_target_diff"]:
                    continue
                source_candidate = node_at_arg_path(source_node, prefix)
                expected_candidate = node_at_arg_path(expected_node, prefix)
                actual_candidate = node_at_arg_path(actual_node, prefix) if actual_node is not None else None
                if is_trivial_expression(source_candidate) or is_trivial_expression(expected_candidate):
                    continue
                if not changed_from_expected(actual_candidate, expected_candidate):
                    continue
                candidate_source_unit = expression_select_unit(source_select, src_dialect, source_candidate, root=source_ast)
                candidate_expected_unit = expression_select_unit(expected_select, tgt_dialect, expected_candidate, root=expected_ast)
                candidate_actual_unit = expression_select_unit(actual_select, tgt_dialect, actual_candidate, root=actual_ast) if actual_select is not None and actual_candidate is not None else actual_sql
                if not candidate_source_unit or not candidate_expected_unit:
                    continue
                parseable = parseable_unit(candidate_source_unit, src_dialect) and parseable_unit(candidate_expected_unit, tgt_dialect)
                key = (normalized(candidate_source_unit), normalized(candidate_expected_unit))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "candidate_kind": "expression",
                    "candidate_path": format_arg_path(prefix),
                    "source_node": source_candidate,
                    "expected_node": expected_candidate,
                    "actual_node": actual_candidate,
                    "source_sql": candidate_source_unit,
                    "expected_sql": candidate_expected_unit,
                    "actual_sql": candidate_actual_unit,
                    "source_canonical_sql": candidate_source_unit,
                    "expected_canonical_sql": candidate_expected_unit,
                    "actual_canonical_sql": candidate_actual_unit,
                    "source_sql_origin": "source_ast_canonical_expression_unit",
                    "expected_unit_source": "expected_ast_canonical_expression_unit",
                    "actual_sql_origin": "actual_ast_canonical_expression_unit",
                    "source_text_span": source_span,
                    "actual_text_span": actual_span,
                    "expected_text_span": expected_span,
                    "unit_completion": "expression_select_context",
                    "dependency_merged": False,
                    "parseable": parseable,
                    "ancestor_coverage": coverage,
                    "score": ancestor_lift_score(
                        candidate_source_unit,
                        candidate_expected_unit,
                        source_span,
                        actual_span,
                        expected_span,
                        diff_depth,
                        dependency_merged=False,
                        candidate_depth=cut_len,
                        candidate_kind="expression",
                        parseable=parseable,
                    ),
                })

            boundary_parseable = parseable_unit(source_unit, src_dialect) and parseable_unit(expected_unit, tgt_dialect)
            boundary_score = ancestor_lift_score(
                source_unit,
                expected_unit,
                source_span,
                actual_span,
                expected_span,
                diff_depth,
                dependency_merged=dependency_merged,
                candidate_kind="clause_boundary",
                parseable=boundary_parseable,
            )
            candidates.append({
                "candidate_kind": "clause_boundary",
                "candidate_path": None,
                "source_node": source_node,
                "expected_node": expected_node,
                "actual_node": actual_node,
                "source_sql": source_unit,
                "expected_sql": expected_unit,
                "actual_sql": actual_unit,
                "source_canonical_sql": source_canonical_unit,
                "expected_canonical_sql": expected_canonical_unit,
                "actual_canonical_sql": actual_canonical_unit,
                "source_sql_origin": source_unit_source,
                "expected_unit_source": expected_unit_source,
                "actual_sql_origin": actual_unit_source,
                "source_text_span": source_span,
                "actual_text_span": actual_span,
                "expected_text_span": expected_span,
                "unit_completion": unit_completion,
                "dependency_merged": dependency_merged,
                "parseable": boundary_parseable,
                "ancestor_coverage": ancestor_coverage(None, path),
                "score": boundary_score,
            })
            candidates.sort(key=lambda item: (item["parseable"], item["score"], -len(item["source_sql"] or "")), reverse=True)
            return candidates

        def public_candidate_summary(candidates):
            summary = []
            for item in candidates[:8]:
                summary.append({
                    "candidate_kind": item.get("candidate_kind"),
                    "candidate_path": item.get("candidate_path"),
                    "source_node_type": item.get("source_node").__class__.__name__ if item.get("source_node") is not None else None,
                    "expected_node_type": item.get("expected_node").__class__.__name__ if item.get("expected_node") is not None else None,
                    "parseable": item.get("parseable"),
                    "score": item.get("score"),
                    "source_sql": item.get("source_sql"),
                    "expected_sql": item.get("expected_sql"),
                    "source_sql_origin": item.get("source_sql_origin"),
                    "expected_unit_source": item.get("expected_unit_source"),
                    "source_canonical_sql": item.get("source_canonical_sql"),
                    "expected_canonical_sql": item.get("expected_canonical_sql"),
                    "ancestor_coverage": item.get("ancestor_coverage"),
                })
            return summary

        # 4. Split materialization, fallback handling, and result emission.
        def add_projection_splits(source_select, expected_select, actual_select, select_ordinal, expected_select_ordinal, actual_select_ordinal):
            source_exprs = list(source_select.expressions or [])
            expected_exprs = list(expected_select.expressions or [])
            actual_exprs = list(actual_select.expressions or []) if actual_select is not None else []

            def add_projection_quote_residual_split(
                idx,
                expected_idx,
                actual_idx,
                source_expr,
                expected_expr,
                actual_expr,
                match_score,
                source_span,
                actual_span,
                expected_span,
                *,
                canonical_split_covers_quote=False,
            ):
                if canonical_split_covers_quote:
                    return
                actual_quote_span = actual_span or source_span
                if not actual_quote_span or not expected_span or not expected_span.get("text"):
                    return
                actual_alias = raw_projection_alias(actual_quote_span.get("text"))
                expected_alias = raw_projection_alias(expected_span.get("text"))
                if not actual_alias or not expected_alias:
                    return
                if expected_alias.get("quote") not in {"`", '"'}:
                    return
                if actual_alias.get("canonical_name") != expected_alias.get("canonical_name"):
                    return
                source_alias = raw_projection_alias((source_span or {}).get("text"))
                if source_alias and source_alias.get("canonical_name") != expected_alias.get("canonical_name"):
                    source_alias = None
                # Keep this residual split conservative. Besides the legacy
                # double-quote-to-backtick target-style case, accept exact quote
                # preservation only when the raw source and raw expected contract
                # agree and the current target rendering changed that quote.
                preserves_raw_quote = bool(
                    source_alias
                    and source_alias.get("quote") == expected_alias.get("quote")
                )
                legacy_target_quote_change = bool(
                    expected_alias.get("quote") == "`"
                    and (
                        actual_alias.get("quote") == '"'
                        or (source_alias or {}).get("quote") == '"'
                    )
                )
                if not preserves_raw_quote and not legacy_target_quote_change:
                    return
                if actual_alias.get("quote") == expected_alias.get("quote"):
                    return
                key = (
                    select_ordinal,
                    "projection_alias_quote",
                    actual_alias.get("quote"),
                    expected_alias.get("quote"),
                )
                if key in residual_quote_split_keys:
                    return
                residual_quote_split_keys.add(key)

                alias_name = expected_alias["name"]
                source_token = (
                    source_alias.get("token")
                    if source_alias
                    else quote_identifier(alias_name, actual_alias.get("quote") or '"')
                )
                actual_token = quote_identifier(alias_name, actual_alias.get("quote") or '"')
                expected_token = quote_identifier(alias_name, expected_alias.get("quote"))
                source_expr_sql = "1 AS %s" % source_token
                actual_expr_sql = "1 AS %s" % actual_token
                expected_expr_sql = "1 AS %s" % expected_token
                source_unit = "SELECT %s" % source_expr_sql
                expected_unit = "SELECT %s" % expected_expr_sql
                try:
                    actual_unit = sqlglot.transpile(source_unit, read=src_dialect, write=tgt_dialect)[0]
                except Exception:
                    actual_unit = "SELECT %s" % actual_expr_sql
                token_summary = changed_token_summary(actual_expr_sql, expected_expr_sql)
                add_split({
                    "split_id": "select_%s_projection_%s_alias_quote" % (select_ordinal, idx),
                    "slice_type": "projection_alias_quote_slice",
                    "projection_index": idx,
                    "target_projection_index": expected_idx,
                    "actual_projection_index": actual_idx if actual_idx < len(actual_exprs) else None,
                    "projection_match_score": match_score,
                    "select_ordinal": select_ordinal,
                    "actual_select_ordinal": actual_select_ordinal,
                    "target_select_ordinal": expected_select_ordinal,
                    "source_sql": source_unit,
                    "expected_sql": expected_unit,
                    "expected_unit_source": "expected_raw_residual_alias_quote_unit",
                    "actual_sql": actual_unit,
                    "source_expr_sql": source_expr_sql,
                    "expected_expr_sql": expected_expr_sql,
                    "actual_expr_sql": actual_expr_sql,
                    "source_ast_path": select_path(select_ordinal, "expressions[%s].alias" % idx),
                    "source_text_span": source_span,
                    "actual_text_span": actual_quote_span,
                    "expected_text_span": expected_span,
                    "target_anchor": select_path(select_ordinal, "expressions[%s].alias" % expected_idx),
                    "source_raw": {
                        "boundary_text": (source_span or {}).get("text"),
                        "span": source_span,
                    },
                    "source_canonical": {
                        "unit_sql": source_unit,
                        "anchor_node_sql": source_expr_sql,
                    },
                    "actual": {
                        "raw_boundary_text": actual_quote_span.get("text"),
                        "unit_sql": actual_unit,
                        "node_sql_canonical": actual_expr_sql,
                    },
                    "expected_raw": {
                        "boundary_text": expected_span.get("text"),
                        "unit_sql": expected_unit,
                    },
                    "expected_canonical": {
                        "node_sql": expected_expr_sql,
                    },
                    "target_diff": {
                        "basis": "residual_actual_expected_raw_alias_quote_diff",
                        "cut_kind": "projection_alias_quote",
                        **token_summary,
                    },
                    "diff_anchor": {
                        "strategy": "residual_raw_alias_quote_diff_to_synthetic_projection_unit",
                        "cut_kind": "projection_alias_quote",
                        "source_ast_node_type": source_expr.__class__.__name__ if source_expr is not None else None,
                        "source_ast_sql": node_sql(source_expr, src_dialect),
                        "source_text_span": source_span,
                        "actual_text_span": actual_quote_span,
                        "expected_text_span": expected_span,
                        "diff_basis": "actual_raw_boundary_text_vs_expected_raw_boundary_text",
                        "actual_changed_tokens": token_summary["actual_only"],
                        "expected_changed_tokens": token_summary["expected_only"],
                        "actual_token_count": token_summary["actual_token_count"],
                        "expected_token_count": token_summary["expected_token_count"],
                        "ancestor_lift": {
                            "lifted_to": "ProjectionAlias",
                            "reason": "residual_quote_diff_not_covered_by_canonical_split",
                            "completion": "synthetic_parseable_alias_sql_unit",
                        },
                    },
                    "cut_node_type": "Alias",
                    "diff_node_type": "Identifier",
                    "target_diff_node_type": "Identifier",
                    "target_diff_node_path": "alias",
                    "selected_ancestor_kind": "projection_alias_quote",
                    "selected_ancestor_path": "alias",
                    "selected_target_diff": {
                        "basis": "residual_actual_expected_raw_alias_quote_diff",
                        "candidate_kind": "projection_alias_quote",
                        "candidate_path": "alias",
                        "source_node_type": "Identifier",
                        "actual_node_type": "Identifier",
                        "expected_node_type": "Identifier",
                        "source_node_sql": source_token,
                        "actual_node_sql": actual_token,
                        "expected_node_sql": expected_token,
                        **changed_token_summary(actual_token, expected_token),
                    },
                    "ancestor_candidates": [],
                    "ancestor_lift_score": None,
                    "expected_source": "target_raw_residual_alias_quote",
                    "change_scope": "alias_or_wrapper",
                    "unit_completion": "synthetic_projection_alias_unit",
                    "cut_strategy": "residual_raw_quote_diff",
                    "dependency_merged": False,
                    "lift_reason": "residual_quote_diff_not_covered_by_canonical_split",
                    "variants": [],
                    "residual_raw_diff": {
                        "kind": "identifier_quote",
                        "actual_alias": actual_alias,
                        "expected_alias": expected_alias,
                        "source_alias": source_alias,
                    },
                })

            for idx, expected_idx, source_expr, expected_expr, match_score in match_expressions(source_exprs, expected_exprs):
                actual_idx = idx if idx < len(actual_exprs) else expected_idx
                actual_expr = actual_exprs[actual_idx] if actual_idx < len(actual_exprs) else None
                span = source_text_span(select_ordinal, "projection", idx)
                actual_span = actual_text_span(actual_select_ordinal, "projection", actual_idx)
                expected_span = expected_text_span(expected_select_ordinal, "projection", expected_idx)
                comparison_expr = actual_expr if actual_expr is not None else source_expr
                comparison_select = actual_select if actual_expr is not None else source_select
                canonical_changed = changed_from_expected(comparison_expr, expected_expr)
                if not canonical_changed:
                    source_projection = (span or {}).get("text") or ""
                    actual_projection = (actual_span or {}).get("text") or ""
                    expected_projection = (expected_span or {}).get("text") or ""

                    def projection_body(text):
                        return re.sub(
                            r"(?is)\s+AS\s+(?:`(?:``|[^`])+`|\"(?:\"\"|[^\"])+\"|[A-Za-z_][A-Za-z0-9_$]*)\s*$",
                            "",
                            text or "",
                        ).rstrip()

                    source_body = projection_body(source_projection)
                    actual_body = projection_body(actual_projection)
                    expected_body = projection_body(expected_projection)
                    if (
                        source_body
                        and actual_body
                        and expected_body
                        and normalized(source_body) == normalized(actual_body) == normalized(expected_body)
                    ):
                        preservation_candidates.append({
                            "source_body": source_body,
                            "actual_body": actual_body,
                            "expected_body": expected_body,
                        })
                    add_projection_quote_residual_split(
                        idx,
                        expected_idx,
                        actual_idx,
                        source_expr,
                        expected_expr,
                        actual_expr,
                        match_score,
                        span,
                        actual_span,
                        expected_span,
                    )
                    continue
                source_diff_node, actual_diff_node, expected_diff_node, target_diff_path, diff_depth = best_target_diff_node(
                    comparison_expr,
                    expected_expr,
                    source_expr,
                )
                body_changed = changed_from_expected(alias_body(comparison_expr), alias_body(expected_expr))
                lift_reason = "target_diff_node_lifted_to_projection_select_unit"
                source_canonical_unit = completed_select(source_select, src_dialect, root=source_ast, expression=source_expr)
                expected_unit_fallback = completed_select(expected_select, tgt_dialect, root=expected_ast, expression=expected_expr)
                expected_unit, expected_unit_source = raw_expected_unit(
                    expected_select_ordinal,
                    "projection",
                    expected_span,
                    expected_unit_fallback,
                )
                source_unit, source_unit_source = raw_source_unit(
                    select_ordinal,
                    "projection",
                    span,
                    source_canonical_unit,
                )
                if actual_expr is not None:
                    actual_canonical_unit = completed_select(actual_select, tgt_dialect, root=actual_ast, expression=actual_expr)
                else:
                    actual_canonical_unit = transpile_source_unit(source_canonical_unit)
                actual_unit, actual_unit_source = raw_actual_unit(
                    actual_select_ordinal,
                    "projection",
                    actual_span,
                    actual_canonical_unit,
                )
                candidates = projection_ancestor_candidates(
                    source_select,
                    expected_select,
                    comparison_select,
                    source_expr,
                    expected_expr,
                    comparison_expr,
                    target_diff_path,
                    select_ordinal=select_ordinal,
                    expected_select_ordinal=expected_select_ordinal,
                    actual_select_ordinal=actual_select_ordinal,
                    source_unit=source_unit,
                    expected_unit=expected_unit,
                    actual_unit=actual_unit,
                    source_canonical_unit=source_canonical_unit,
                    expected_canonical_unit=expected_unit_fallback,
                    actual_canonical_unit=actual_canonical_unit,
                    source_unit_source=source_unit_source,
                    actual_unit_source=actual_unit_source,
                    source_span=span,
                    actual_span=actual_span,
                    expected_span=expected_span,
                    expected_unit_source=expected_unit_source,
                    diff_depth=diff_depth,
                )
                selected = candidates[0]
                selected_source_node = selected["source_node"]
                selected_expected_node = selected["expected_node"]
                selected_actual_node = selected["actual_node"]
                source_unit = selected["source_sql"]
                expected_unit = selected["expected_sql"]
                actual_unit = selected["actual_sql"]
                source_canonical_unit = selected.get("source_canonical_sql") or source_unit
                expected_canonical_unit = selected.get("expected_canonical_sql") or expected_unit
                actual_canonical_unit = selected.get("actual_canonical_sql") or actual_unit
                source_unit_source = selected.get("source_sql_origin")
                expected_unit_source = selected["expected_unit_source"]
                actual_unit_source = selected.get("actual_sql_origin")
                span = selected["source_text_span"]
                actual_span = selected["actual_text_span"]
                expected_span = selected["expected_text_span"]
                lift_score = selected["score"]
                selected_candidate_kind = selected.get("candidate_kind")
                variants = combined_projection_variants(
                    source_select,
                    expected_select,
                    selected_source_node,
                    selected_expected_node,
                    select_ordinal,
                    idx,
                    selected_candidate_kind=selected_candidate_kind,
                )
                add_split({
                    "split_id": "select_%s_projection_%s" % (select_ordinal, idx),
                    "slice_type": "projection_slice",
                    "projection_index": idx,
                    "target_projection_index": expected_idx,
                    "actual_projection_index": actual_idx if actual_idx < len(actual_exprs) else None,
                    "projection_match_score": match_score,
                    "select_ordinal": select_ordinal,
                    "actual_select_ordinal": actual_select_ordinal,
                    "target_select_ordinal": expected_select_ordinal,
                    "source_sql": source_unit,
                    "expected_sql": expected_unit,
                    "source_sql_origin": source_unit_source,
                    "expected_unit_source": expected_unit_source,
                    "actual_sql_origin": actual_unit_source,
                    "actual_sql": actual_unit,
                    "source_canonical_sql": source_canonical_unit,
                    "expected_canonical_sql": expected_canonical_unit,
                    "actual_canonical_sql": actual_canonical_unit,
                    "source_expr_sql": node_sql(selected_source_node, src_dialect),
                    "expected_expr_sql": node_sql(selected_expected_node, tgt_dialect),
                    "actual_expr_sql": node_sql(selected_actual_node, tgt_dialect) if selected_actual_node is not None else actual_sql,
                    "source_ast_path": select_path(select_ordinal, "expressions[%s]" % idx),
                    "source_text_span": span,
                    "actual_text_span": actual_span,
                    "expected_text_span": expected_span,
                    "target_anchor": "%s.%s" % (
                        select_path(select_ordinal, "expressions[%s]" % expected_idx),
                        selected["candidate_path"],
                    ) if selected.get("candidate_path") else select_path(select_ordinal, "expressions[%s]" % expected_idx),
                    **four_route_metadata(
                        selected_source_node,
                        selected_actual_node,
                        selected_expected_node,
                        span,
                        actual_span,
                        expected_span,
                        source_unit,
                        actual_unit,
                        expected_unit,
                        cut_kind="projection",
                        source_canonical_unit=source_canonical_unit,
                        actual_canonical_unit=actual_canonical_unit,
                        expected_canonical_unit=expected_canonical_unit,
                    ),
                    "diff_anchor": diff_anchor(
                        selected_source_node,
                        selected_actual_node,
                        selected_expected_node,
                        span,
                        cut_kind="projection",
                        lift_reason=lift_reason,
                        actual_span=actual_span,
                        expected_span=expected_span,
                        target_diff_node=expected_diff_node,
                        source_diff_node=source_diff_node,
                        ancestor_lift_score=lift_score,
                        target_diff_path=format_arg_path(target_diff_path),
                        ancestor_coverage=selected.get("ancestor_coverage"),
                    ),
                    "cut_node_type": selected_source_node.__class__.__name__ if selected_source_node is not None else None,
                    "diff_node_type": source_diff_node.__class__.__name__ if source_diff_node is not None else None,
                    "target_diff_node_type": expected_diff_node.__class__.__name__ if expected_diff_node is not None else None,
                    "target_diff_node_path": format_arg_path(target_diff_path),
                    "selected_ancestor_kind": selected.get("candidate_kind"),
                    "selected_ancestor_path": selected.get("candidate_path"),
                    "selected_ancestor_coverage": selected.get("ancestor_coverage"),
                    "selected_target_diff": selected_target_diff_metadata(
                        selected_actual_node,
                        selected_expected_node,
                        selected_source_node,
                        actual_unit,
                        expected_unit,
                        source_unit,
                        candidate_kind=selected.get("candidate_kind"),
                        candidate_path=selected.get("candidate_path"),
                    ),
                    "ancestor_candidates": public_candidate_summary(candidates),
                    "ancestor_lift_score": lift_score,
                    "expected_source": "target_ast_aligned",
                    "change_scope": "expression" if body_changed else "alias_or_wrapper",
                    "unit_completion": selected.get("unit_completion"),
                    "cut_strategy": "target_diff_guided_ancestor_lift",
                    "dependency_merged": selected.get("dependency_merged"),
                    "lift_reason": lift_reason,
                    "variants": variants,
                })
                expected_alias = raw_projection_alias((expected_span or {}).get("text"))
                canonical_split_covers_quote = (
                    selected.get("candidate_kind") == "projection_boundary"
                    and expected_alias is not None
                    and expected_alias.get("quote") in {"`", '"'}
                    and expected_alias.get("token") in (expected_unit or "")
                )
                add_projection_quote_residual_split(
                    idx,
                    expected_idx,
                    actual_idx,
                    source_expr,
                    expected_expr,
                    actual_expr,
                    match_score,
                    span,
                    actual_span,
                    expected_span,
                    canonical_split_covers_quote=canonical_split_covers_quote,
                )

        def add_clause_split(source_select, expected_select, actual_select, select_ordinal, expected_select_ordinal, actual_select_ordinal, slice_type, arg_name, keep):
            source_node = source_select.args.get(arg_name)
            expected_node = expected_select.args.get(arg_name)
            actual_node = actual_select.args.get(arg_name) if actual_select is not None else None
            if source_node is None or expected_node is None:
                return
            comparison_node = actual_node if actual_node is not None else source_node
            comparison_select = actual_select if actual_node is not None else source_select
            if not changed_from_expected(comparison_node, expected_node):
                return
            source_diff_node, actual_diff_node, expected_diff_node, target_diff_path, diff_depth = best_target_diff_node(
                comparison_node,
                expected_node,
                source_node,
            )
            span = source_text_span(select_ordinal, arg_name)
            actual_span = actual_text_span(actual_select_ordinal, arg_name)
            expected_span = expected_text_span(expected_select_ordinal, arg_name)
            source_dependency_indexes = context_expression_indexes(
                source_select,
                source_node,
                context_kind=arg_name,
            )
            expected_dependency_indexes = context_expression_indexes(
                expected_select,
                expected_node,
                context_kind=arg_name,
            )
            actual_dependency_indexes = context_expression_indexes(
                actual_select,
                actual_node,
                context_kind=arg_name,
            ) if actual_select is not None and actual_node is not None else []
            raw_clause_spans = (span, actual_span, expected_span)
            context_free_clause = (
                arg_name in {"where", "group", "order"}
                and not source_dependency_indexes
                and not actual_dependency_indexes
                and not expected_dependency_indexes
                and all(isinstance(item, dict) and item.get("text") for item in raw_clause_spans)
                and not any(
                    re.search(
                        r"\bSELECT\b",
                        re.sub(r"'(?:''|[^'])*'", "'__STR__'", item.get("text") or ""),
                        re.IGNORECASE,
                    )
                    for item in raw_clause_spans
                )
            )
            lift_reason = "target_diff_node_lifted_to_clause_select_unit"
            source_canonical_unit = completed_select(source_select, src_dialect, root=source_ast, keep=keep, context_node=source_node, context_kind=arg_name)
            source_unit, source_unit_source = raw_source_unit(
                select_ordinal,
                arg_name,
                span,
                source_canonical_unit,
                source_dependency_indexes,
                context_free=context_free_clause,
            )
            expected_unit_fallback = completed_select(expected_select, tgt_dialect, root=expected_ast, keep=keep, context_node=expected_node, context_kind=arg_name)
            expected_unit, expected_unit_source = raw_expected_unit(
                expected_select_ordinal,
                arg_name,
                expected_span,
                expected_unit_fallback,
                expected_dependency_indexes,
                context_free=context_free_clause,
            )
            if actual_node is not None:
                actual_canonical_unit = completed_select(actual_select, tgt_dialect, root=actual_ast, keep=keep, context_node=actual_node, context_kind=arg_name)
            else:
                actual_canonical_unit = transpile_source_unit(source_canonical_unit)
            actual_dependency_projection_texts, context_normalizations = harmonized_actual_dependency_projections(
                actual_select_ordinal,
                actual_dependency_indexes,
                expected_select_ordinal,
                expected_dependency_indexes,
            )
            actual_unit, actual_unit_source = raw_actual_unit(
                actual_select_ordinal,
                arg_name,
                actual_span,
                actual_canonical_unit,
                actual_dependency_indexes,
                actual_dependency_projection_texts,
                context_free=context_free_clause,
            )
            candidates = clause_ancestor_candidates(
                source_select,
                expected_select,
                comparison_select,
                source_node,
                expected_node,
                comparison_node,
                target_diff_path,
                source_unit=source_unit,
                expected_unit=expected_unit,
                actual_unit=actual_unit,
                source_canonical_unit=source_canonical_unit,
                expected_canonical_unit=expected_unit_fallback,
                actual_canonical_unit=actual_canonical_unit,
                source_unit_source=source_unit_source,
                actual_unit_source=actual_unit_source,
                source_span=span,
                actual_span=actual_span,
                expected_span=expected_span,
                expected_unit_source=expected_unit_source,
                diff_depth=diff_depth,
                dependency_merged=True,
                unit_completion="dependency_context",
                allow_expression_candidates=arg_name == "where",
            )
            selected = candidates[0]
            if selected.get("candidate_kind") != "clause_boundary":
                context_normalizations = []
            selected_source_node = selected["source_node"]
            selected_expected_node = selected["expected_node"]
            selected_actual_node = selected["actual_node"]
            source_unit = selected["source_sql"]
            expected_unit = selected["expected_sql"]
            actual_unit = selected["actual_sql"]
            source_canonical_unit = selected.get("source_canonical_sql") or source_unit
            expected_canonical_unit = selected.get("expected_canonical_sql") or expected_unit
            actual_canonical_unit = selected.get("actual_canonical_sql") or actual_unit
            source_unit_source = selected.get("source_sql_origin")
            expected_unit_source = selected["expected_unit_source"]
            actual_unit_source = selected.get("actual_sql_origin")
            source_unit, actual_unit, expected_unit, source_unit_source, actual_unit_source, expected_unit_source = (
                align_select_one_projection_units(
                    source_unit,
                    actual_unit,
                    expected_unit,
                    source_origin=source_unit_source,
                    actual_origin=actual_unit_source,
                    expected_origin=expected_unit_source,
                )
            )
            span = selected["source_text_span"]
            actual_span = selected["actual_text_span"]
            expected_span = selected["expected_text_span"]
            lift_score = selected["score"]
            add_split({
                "split_id": "select_%s_%s" % (select_ordinal, arg_name),
                "slice_type": slice_type,
                "projection_index": None,
                "select_ordinal": select_ordinal,
                "actual_select_ordinal": actual_select_ordinal,
                "target_select_ordinal": expected_select_ordinal,
                "source_sql": source_unit,
                "expected_sql": expected_unit,
                "source_sql_origin": source_unit_source,
                "expected_unit_source": expected_unit_source,
                "actual_sql_origin": actual_unit_source,
                "actual_sql": actual_unit,
                "context_normalizations": context_normalizations,
                "source_canonical_sql": source_canonical_unit,
                "expected_canonical_sql": expected_canonical_unit,
                "actual_canonical_sql": actual_canonical_unit,
                "source_expr_sql": node_sql(selected_source_node, src_dialect),
                "expected_expr_sql": node_sql(selected_expected_node, tgt_dialect),
                "actual_expr_sql": node_sql(selected_actual_node, tgt_dialect) if selected_actual_node is not None else actual_sql,
                "source_ast_path": select_path(select_ordinal, arg_name),
                "source_text_span": span,
                "actual_text_span": actual_span,
                "expected_text_span": expected_span,
                "target_anchor": "%s.%s" % (
                    select_path(select_ordinal, arg_name),
                    selected["candidate_path"],
                ) if selected.get("candidate_path") else select_path(select_ordinal, arg_name),
                **four_route_metadata(
                    selected_source_node,
                    selected_actual_node,
                    selected_expected_node,
                    span,
                    actual_span,
                    expected_span,
                    source_unit,
                    actual_unit,
                    expected_unit,
                    cut_kind=arg_name,
                    source_canonical_unit=source_canonical_unit,
                    actual_canonical_unit=actual_canonical_unit,
                    expected_canonical_unit=expected_canonical_unit,
                ),
                "diff_anchor": diff_anchor(
                    selected_source_node,
                    selected_actual_node,
                    selected_expected_node,
                    span,
                    cut_kind=arg_name,
                    lift_reason=lift_reason,
                    actual_span=actual_span,
                    expected_span=expected_span,
                    target_diff_node=expected_diff_node,
                    source_diff_node=source_diff_node,
                    ancestor_lift_score=lift_score,
                    target_diff_path=format_arg_path(target_diff_path),
                    ancestor_coverage=selected.get("ancestor_coverage"),
                ),
                "cut_node_type": selected_source_node.__class__.__name__ if selected_source_node is not None else None,
                "diff_node_type": source_diff_node.__class__.__name__ if source_diff_node is not None else None,
                "target_diff_node_type": expected_diff_node.__class__.__name__ if expected_diff_node is not None else None,
                "target_diff_node_path": format_arg_path(target_diff_path),
                "selected_ancestor_kind": selected.get("candidate_kind"),
                "selected_ancestor_path": selected.get("candidate_path"),
                "selected_ancestor_coverage": selected.get("ancestor_coverage"),
                "selected_target_diff": selected_target_diff_metadata(
                    selected_actual_node,
                    selected_expected_node,
                    selected_source_node,
                    actual_unit,
                    expected_unit,
                    source_unit,
                    candidate_kind=selected.get("candidate_kind"),
                    candidate_path=selected.get("candidate_path"),
                ),
                "ancestor_candidates": public_candidate_summary(candidates),
                "ancestor_lift_score": lift_score,
                "expected_source": "target_ast_aligned",
                "unit_completion": selected.get("unit_completion"),
                "cut_strategy": "target_diff_guided_ancestor_lift",
                "dependency_merged": selected.get("dependency_merged"),
                "lift_reason": lift_reason,
                "variants": [],
            })

        def add_join_splits(source_select, expected_select, actual_select, select_ordinal, expected_select_ordinal, actual_select_ordinal):
            source_joins = list(source_select.args.get("joins") or [])
            expected_joins = list(expected_select.args.get("joins") or [])
            actual_joins = list(actual_select.args.get("joins") or []) if actual_select is not None else []
            for idx in range(min(len(source_joins), len(expected_joins))):
                source_join = source_joins[idx]
                expected_join = expected_joins[idx]
                actual_join = actual_joins[idx] if idx < len(actual_joins) else None
                comparison_join = actual_join if actual_join is not None else source_join
                if not changed_from_expected(comparison_join, expected_join):
                    continue
                source_diff_node, actual_diff_node, expected_diff_node, target_diff_path, diff_depth = best_target_diff_node(
                    comparison_join,
                    expected_join,
                    source_join,
                )
                span = source_text_span(select_ordinal, "join", idx)
                actual_span = actual_text_span(actual_select_ordinal, "join", idx)
                expected_span = expected_text_span(expected_select_ordinal, "join", idx)
                lift_reason = "target_diff_node_lifted_to_join_prefix_select_unit"
                source_unit = completed_select(source_select, src_dialect, root=source_ast, keep={"joins"}, join_index=idx, context_node=source_join)
                expected_unit_fallback = completed_select(expected_select, tgt_dialect, root=expected_ast, keep={"joins"}, join_index=idx, context_node=expected_join)
                expected_unit, expected_unit_source = raw_expected_unit(
                    expected_select_ordinal,
                    "join",
                    expected_span,
                    expected_unit_fallback,
                )
                if actual_join is not None:
                    actual_unit = completed_select(actual_select, tgt_dialect, root=actual_ast, keep={"joins"}, join_index=idx, context_node=actual_join)
                else:
                    actual_unit = transpile_source_unit(source_unit)
                source_unit, actual_unit, expected_unit, source_unit_source, actual_unit_source, expected_unit_source = (
                    align_select_one_projection_units(
                        source_unit,
                        actual_unit,
                        expected_unit,
                        source_origin="source_ast_dependency_context",
                        actual_origin="actual_ast_dependency_context" if actual_join is not None else "source_transpile_dependency_context",
                        expected_origin=expected_unit_source,
                    )
                )
                lift_score = ancestor_lift_score(source_unit, expected_unit, span, actual_span, expected_span, diff_depth, dependency_merged=True)
                add_split({
                    "split_id": "select_%s_join_%s" % (select_ordinal, idx),
                    "slice_type": "join_slice",
                    "projection_index": None,
                    "select_ordinal": select_ordinal,
                    "actual_select_ordinal": actual_select_ordinal,
                    "target_select_ordinal": expected_select_ordinal,
                    "source_sql": source_unit,
                    "expected_sql": expected_unit,
                    "source_sql_origin": source_unit_source,
                    "expected_unit_source": expected_unit_source,
                    "actual_sql_origin": actual_unit_source,
                    "actual_sql": actual_unit,
                    "source_expr_sql": node_sql(source_join, src_dialect),
                    "expected_expr_sql": node_sql(expected_join, tgt_dialect),
                    "actual_expr_sql": node_sql(comparison_join, tgt_dialect),
                    "source_ast_path": select_path(select_ordinal, "joins[%s]" % idx),
                    "source_text_span": span,
                    "actual_text_span": actual_span,
                    "expected_text_span": expected_span,
                    "target_anchor": select_path(select_ordinal, "joins[%s]" % idx),
                    **four_route_metadata(
                        source_join,
                        comparison_join,
                        expected_join,
                        span,
                        actual_span,
                        expected_span,
                        source_unit,
                        actual_unit,
                        expected_unit,
                        cut_kind="join",
                    ),
                    "diff_anchor": diff_anchor(
                        source_join,
                        comparison_join,
                        expected_join,
                        span,
                        cut_kind="join",
                        lift_reason=lift_reason,
                        actual_span=actual_span,
                        expected_span=expected_span,
                        target_diff_node=expected_diff_node,
                        source_diff_node=source_diff_node,
                        ancestor_lift_score=lift_score,
                        target_diff_path=format_arg_path(target_diff_path),
                    ),
                    "cut_node_type": source_join.__class__.__name__,
                    "diff_node_type": source_diff_node.__class__.__name__ if source_diff_node is not None else None,
                    "target_diff_node_type": expected_diff_node.__class__.__name__ if expected_diff_node is not None else None,
                    "target_diff_node_path": format_arg_path(target_diff_path),
                    "ancestor_lift_score": lift_score,
                    "expected_source": "target_ast_aligned",
                    "unit_completion": "join_prefix_context",
                    "cut_strategy": "target_diff_guided_ancestor_lift",
                    "dependency_merged": True,
                    "lift_reason": lift_reason,
                    "variants": [],
                })

        table_alias_as_re = re.compile(
            r"(?:\b(?:FROM|JOIN)\s+|,\s*)"
            r"(?P<table>(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w$]*))?)"
            r"\s+(?P<as>AS\s+)?"
            r"(?P<alias>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w$]*)"
            r"(?=\s+(?:ON|USING|WHERE|JOIN|INNER|LEFT|RIGHT|FULL|CROSS|GROUP|ORDER|HAVING|LIMIT|QUALIFY|UNION|$)|\s*,|\)|$)",
            re.IGNORECASE,
        )

        def normalized_alias_identifier(value):
            value = (value or "").strip()
            if len(value) >= 2 and (
                (value[0] == value[-1] and value[0] in {'"', "`"})
                or (value[0] == "[" and value[-1] == "]")
            ):
                value = value[1:-1]
            return value.lower()

        def raw_base_table_alias_styles(sql):
            masked = re.sub(r"'(?:''|[^'])*'", "'__STR__'", sql or "")
            styles = {}
            for match in table_alias_as_re.finditer(masked):
                table = ".".join(
                    normalized_alias_identifier(part)
                    for part in re.split(r"\s*\.\s*", match.group("table"))
                )
                alias = normalized_alias_identifier(match.group("alias"))
                if table and alias:
                    styles.setdefault((table, alias), []).append(bool(match.group("as")))
            return styles

        def synthetic_table_alias_unit(has_as):
            separator = " AS " if has_as else " "
            return "SELECT 1 FROM autodialect_table%sautodialect_alias" % separator

        def add_base_table_alias_as_residual_splits():
            source_styles = raw_base_table_alias_styles(src_sql)
            actual_styles = raw_base_table_alias_styles(actual_sql)
            expected_styles = raw_base_table_alias_styles(expected_sql)
            emitted = set()
            for pair in source_styles.keys() & actual_styles.keys() & expected_styles.keys():
                source_values = set(source_styles[pair])
                actual_values = set(actual_styles[pair])
                expected_values = set(expected_styles[pair])
                if len(source_values) != 1 or len(actual_values) != 1 or len(expected_values) != 1:
                    continue
                source_has_as = next(iter(source_values))
                actual_has_as = next(iter(actual_values))
                expected_has_as = next(iter(expected_values))
                # A synthetic hard unit is safe only for exact raw-style
                # preservation. Generic alias-style conversion remains parent-only.
                if source_has_as != expected_has_as or actual_has_as == expected_has_as:
                    continue
                style_key = (source_has_as, actual_has_as, expected_has_as)
                if style_key in emitted:
                    continue
                emitted.add(style_key)
                source_unit = synthetic_table_alias_unit(source_has_as)
                expected_unit = synthetic_table_alias_unit(expected_has_as)
                try:
                    actual_unit = sqlglot.transpile(
                        source_unit,
                        read=src_dialect,
                        write=tgt_dialect,
                    )[0]
                except Exception:
                    actual_unit = synthetic_table_alias_unit(actual_has_as)
                actual_synthetic_styles = raw_base_table_alias_styles(actual_unit)
                synthetic_pair = ("autodialect_table", "autodialect_alias")
                if set(actual_synthetic_styles.get(synthetic_pair) or []) != {actual_has_as}:
                    continue
                source_expr_sql = "FROM autodialect_table%sautodialect_alias" % (
                    " AS " if source_has_as else " "
                )
                actual_expr_sql = "FROM autodialect_table%sautodialect_alias" % (
                    " AS " if actual_has_as else " "
                )
                expected_expr_sql = "FROM autodialect_table%sautodialect_alias" % (
                    " AS " if expected_has_as else " "
                )
                token_summary = changed_token_summary(actual_expr_sql, expected_expr_sql)
                add_split({
                    "split_id": "base_table_alias_as_%s_%s_%s" % tuple(
                        "with" if value else "without"
                        for value in style_key
                    ),
                    "slice_type": "table_alias_as_surface_slice",
                    "source_sql": source_unit,
                    "actual_sql": actual_unit,
                    "expected_sql": expected_unit,
                    "source_expr_sql": source_expr_sql,
                    "actual_expr_sql": actual_expr_sql,
                    "expected_expr_sql": expected_expr_sql,
                    "source_rule_fragment_sql": source_expr_sql,
                    "actual_rule_fragment_sql": actual_expr_sql,
                    "expected_rule_fragment_sql": expected_expr_sql,
                    "source_sql_origin": "source_raw_residual_table_alias_as_synthetic_unit",
                    "actual_sql_origin": "actual_raw_residual_table_alias_as_synthetic_unit",
                    "expected_unit_source": "expected_raw_residual_table_alias_as_synthetic_unit",
                    "source_ast_path": "TableAlias",
                    "target_anchor": "TableAlias",
                    "cut_node_type": "TableAlias",
                    "target_diff_node_type": "TableAlias",
                    "target_diff_node_path": "from.alias",
                    "selected_ancestor_kind": "table_alias_as_surface",
                    "selected_ancestor_path": "from.alias",
                    "selected_target_diff": {
                        "basis": "residual_actual_expected_raw_table_alias_as_diff",
                        "candidate_kind": "table_alias_as_surface",
                        "candidate_path": "from.alias",
                        "source_node_type": "TableAlias",
                        "actual_node_type": "TableAlias",
                        "expected_node_type": "TableAlias",
                        "source_node_sql": source_expr_sql,
                        "actual_node_sql": actual_expr_sql,
                        "expected_node_sql": expected_expr_sql,
                        **token_summary,
                    },
                    "target_diff": {
                        "basis": "residual_actual_expected_raw_table_alias_as_diff",
                        "cut_kind": "table_alias_as_surface",
                        **token_summary,
                    },
                    "unit_completion": "synthetic_base_table_alias_unit",
                    "cut_strategy": "residual_raw_table_alias_as_diff",
                    "dependency_merged": False,
                    "lift_reason": "table_alias_as_diff_not_covered_by_clause_unit",
                    "variants": [],
                })

        def function_names(sql):
            return {
                token.upper()
                for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()", sql or "")
            }

        def add_surface_provenance_preservation_guards():
            canonical_collision_triggers = {}
            for split in splits:
                if normalized(split.get("actual_sql")) == normalized(split.get("expected_sql")):
                    continue
                canonical_functions = function_names(split.get("expected_expr_sql"))
                raw_functions = function_names(split.get("expected_rule_fragment_sql"))
                for function_name in canonical_functions - raw_functions:
                    canonical_collision_triggers.setdefault(function_name, set()).add(split.get("split_id"))
            canonical_collision_functions = set(canonical_collision_triggers)
            if not canonical_collision_functions:
                return

            emitted_functions = set()
            for candidate in preservation_candidates:
                guarded_functions = (
                    function_names(candidate.get("expected_body"))
                    & canonical_collision_functions
                ) - emitted_functions
                if not guarded_functions:
                    continue
                source_unit = "SELECT %s" % candidate["source_body"]
                expected_unit = "SELECT %s" % candidate["expected_body"]
                try:
                    actual_unit = sqlglot.transpile(
                        source_unit,
                        read=src_dialect,
                        write=tgt_dialect,
                    )[0]
                except Exception:
                    continue
                if normalized(actual_unit) != normalized(expected_unit):
                    continue
                guard_name = sorted(guarded_functions)[0]
                emitted_functions.update(guarded_functions)
                trigger_split_ids = sorted({
                    split_id
                    for function_name in guarded_functions
                    for split_id in canonical_collision_triggers.get(function_name, set())
                    if split_id
                })
                if not trigger_split_ids:
                    continue
                add_split({
                    "split_id": "preserve_surface_provenance_%s" % guard_name.lower(),
                    "slice_type": "surface_provenance_guard_slice",
                    "source_sql": source_unit,
                    "actual_sql": actual_unit,
                    "expected_sql": expected_unit,
                    "source_expr_sql": candidate["source_body"],
                    "actual_expr_sql": candidate["actual_body"],
                    # This guard is deliberately raw-surface based. Reusing the
                    # positionally aligned AST expression here can attach a
                    # different projection's identifiers to an otherwise exact
                    # raw match and pollute grouping/prompt metadata.
                    "expected_expr_sql": candidate["expected_body"],
                    "source_rule_fragment_sql": candidate["source_body"],
                    "actual_rule_fragment_sql": candidate["actual_body"],
                    "expected_rule_fragment_sql": candidate["expected_body"],
                    "source_sql_origin": "source_raw_surface_provenance_guard_unit",
                    "actual_sql_origin": "actual_raw_surface_provenance_guard_unit",
                    "expected_unit_source": "expected_raw_surface_provenance_guard_unit",
                    "source_ast_path": "SurfaceProvenanceGuard.%s" % guard_name,
                    "target_anchor": "SurfaceProvenanceGuard.%s" % guard_name,
                    "cut_node_type": "Func",
                    "target_diff_node_type": "Func",
                    "selected_ancestor_kind": "surface_provenance_guard",
                    "selected_ancestor_path": None,
                    "selected_target_diff": {
                        "basis": "already_correct_surface_provenance_guard",
                        "candidate_kind": "surface_provenance_guard",
                        "source_node_type": "Func",
                        "actual_node_type": "Func",
                        "expected_node_type": "Func",
                        "source_node_sql": candidate["source_body"],
                        "actual_node_sql": candidate["actual_body"],
                        "expected_node_sql": candidate["expected_body"],
                    },
                    "target_diff": {
                        "basis": "already_correct_surface_provenance_guard",
                        "cut_kind": "surface_provenance_guard",
                        "actual_only": [],
                        "expected_only": [],
                    },
                    "unit_completion": "synthetic_surface_provenance_guard_unit",
                    "cut_strategy": "already_correct_surface_provenance_guard",
                    "dependency_merged": False,
                    "lift_reason": "protect_raw_function_spelling_from_canonical_collision",
                    "preservation_trigger_split_ids": trigger_split_ids,
                    "variants": [],
                })
                if len(emitted_functions) >= 3:
                    break

        source_selects = select_nodes(source_ast)
        source_select_spans = align_select_spans(source_selects, build_select_spans(src_sql, src_dialect), src_dialect)
        expected_selects = select_nodes(expected_ast)
        actual_selects = select_nodes(actual_ast)
        actual_select_spans = align_select_spans(actual_selects, build_select_spans(actual_sql, tgt_dialect), tgt_dialect)
        expected_select_spans = align_select_spans(expected_selects, build_select_spans(expected_sql, tgt_dialect), tgt_dialect)
        expected_matches = match_selects(source_selects, expected_selects)
        actual_matches = match_selects(source_selects, actual_selects)
        processed_select_count = min(len(source_selects), 12)
        if len(source_selects) > processed_select_count:
            split_limits["truncated"] = True
        for select_ordinal, source_select in enumerate(source_selects[:12]):
            if select_ordinal not in expected_matches:
                continue
            expected_ordinal, expected_select, select_match_score = expected_matches[select_ordinal]
            actual_match = actual_matches.get(select_ordinal)
            actual_ordinal = actual_match[0] if actual_match else None
            actual_select = actual_match[1] if actual_match else None
            add_projection_splits(source_select, expected_select, actual_select, select_ordinal, expected_ordinal, actual_ordinal)
            add_clause_split(source_select, expected_select, actual_select, select_ordinal, expected_ordinal, actual_ordinal, "where_slice", "where", {"where", "joins"})
            add_join_splits(source_select, expected_select, actual_select, select_ordinal, expected_ordinal, actual_ordinal)
            add_clause_split(source_select, expected_select, actual_select, select_ordinal, expected_ordinal, actual_ordinal, "group_slice", "group", {"group", "joins"})
            add_clause_split(source_select, expected_select, actual_select, select_ordinal, expected_ordinal, actual_ordinal, "having_slice", "having", {"group", "having", "joins"})
            add_clause_split(source_select, expected_select, actual_select, select_ordinal, expected_ordinal, actual_ordinal, "order_slice", "order", {"order", "joins"})
            add_clause_split(source_select, expected_select, actual_select, select_ordinal, expected_ordinal, actual_ordinal, "qualify_slice", "qualify", {"qualify", "joins"})
            for split in splits:
                if split.get("select_ordinal") == select_ordinal and "target_select_ordinal" not in split:
                    split["target_select_ordinal"] = expected_ordinal
                    split["select_match_score"] = select_match_score

        add_surface_provenance_preservation_guards()
        add_base_table_alias_as_residual_splits()

        if not splits:
            expected_parent_sql = expected_sql
            expected_parent_canonical_sql = preserve_expected_identifier_quote(
                expected_ast.sql(dialect=tgt_dialect)
            ) if expected_ast is not None else expected_sql
            fallback_source_path = source_ast.__class__.__name__
            if source_selects:
                fallback_source_path = "%s.Select" % fallback_source_path
            parent_source_span = {
                "start": 0,
                "end": len(src_sql),
                "text": src_sql,
                "kind": "parent",
                "source": "source_raw_parent_sql_text",
            }
            parent_actual_span = {
                "start": 0,
                "end": len(actual_sql),
                "text": actual_sql,
                "kind": "parent",
                "source": "actual_raw_parent_sql_text",
            }
            parent_expected_span = {
                "start": 0,
                "end": len(expected_sql),
                "text": expected_sql,
                "kind": "parent",
                "source": "expected_raw_parent_sql_text",
            }
            parent_source_unit = src_sql
            parent_source_canonical_unit = source_ast.sql(dialect=src_dialect)
            splits.append({
                "split_id": "parent_0",
                "slice_type": "parent",
                "projection_index": None,
                "select_ordinal": None,
                "source_sql": parent_source_unit,
                "expected_sql": expected_parent_sql,
                "source_sql_origin": "source_raw_parent_sql_text",
                "expected_unit_source": "expected_raw_parent_sql_text",
                "actual_sql_origin": "actual_raw_parent_sql_text",
                "actual_sql": actual_sql,
                "source_canonical_sql": parent_source_canonical_unit,
                "expected_canonical_sql": expected_parent_canonical_sql,
                "actual_canonical_sql": actual_sql,
                "source_expr_sql": parent_source_unit,
                "expected_expr_sql": expected_parent_sql,
                "actual_expr_sql": actual_sql,
                "source_ast_path": fallback_source_path,
                "source_text_span": parent_source_span,
                "actual_text_span": parent_actual_span,
                "expected_text_span": parent_expected_span,
                **four_route_metadata(
                    source_ast,
                    actual_ast,
                    expected_ast,
                    parent_source_span,
                    parent_actual_span,
                    parent_expected_span,
                    parent_source_unit,
                    actual_sql,
                    expected_parent_sql,
                    cut_kind="parent",
                    source_canonical_unit=parent_source_canonical_unit,
                    actual_canonical_unit=actual_sql,
                    expected_canonical_unit=expected_parent_canonical_sql,
                ),
                "target_anchor": expected_ast.__class__.__name__ if expected_ast is not None else "ExpectedSQLText",
                "cut_node_type": source_ast.__class__.__name__,
                "hard_assertion": expected_ast is not None,
                "skip_reason": None if expected_ast is not None else "target_parse_unavailable",
                "unit_completion": "parent_fallback",
                "cut_strategy": "parent_fallback",
                "dependency_merged": True,
                "lift_reason": "no_changed_child_cut_found",
                "expected_source": "target_ast_aligned" if expected_ast is not None else "target_text_fallback",
                "variants": [],
            })

        emit({
            "ok": True,
            "status": "split_ok",
            "actual_sql": actual_sql,
            "expected_status": expected_status,
            "expected_error": expected_error,
            "source_root": source_ast.__class__.__name__,
            "expected_root": expected_ast.__class__.__name__ if expected_ast is not None else None,
            "select_count": len(source_selects),
            "processed_select_count": processed_select_count,
            "source_text_span_count": len(source_select_spans),
            "expected_select_count": len(expected_selects),
            "actual_select_count": len(actual_selects),
            "split_truncated": split_limits["truncated"],
            "splits": splits,
        })
    except Exception as exc:
        emit({
            "ok": False,
            "status": "split_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })


if __name__ == "__main__":
    main()
