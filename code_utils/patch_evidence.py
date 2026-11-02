"""Shared deterministic extraction of changed-file and patch evidence.

These helpers are neutral pipeline utilities used by stage handoff and runtime
memory summaries.
They intentionally have no dependency on ``code_utils.memory``.
"""

from __future__ import annotations

import re
from typing import Any

_SQLGLOT_PATCH_PREFIXES = (
    "sqlglot/dialects/",
    "sqlglot/expressions.py",
    "sqlglot/parser.py",
    "sqlglot/tokens.py",
    "sqlglot/transforms.py",
    "sqlglot/generator.py",
)


def _clean_path(path: Any) -> str | None:
    if not path:
        return None
    if isinstance(path, dict):
        path = path.get("path") or path.get("file") or path.get("file_path") or path.get("name")
    if not isinstance(path, str):
        return None
    path = path.strip().replace("\\", "/")
    return path or None


def extract_changed_files(record: dict[str, Any]) -> list[str]:
    candidates: list[Any] = []
    for key in ("file_changes", "backfilled_file_changes", "changed_files"):
        value = record.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, str):
            candidates.extend(re.findall(r"[\w./-]+\.py", value))

    gen_code = record.get("gen_code") or record.get("backfilled_gen_code") or ""
    if isinstance(gen_code, str):
        candidates.extend(re.findall(r"(?:^|\s)(sqlglot/[\w./-]+\.py)", gen_code))
        candidates.extend(re.findall(r"diff --git a/(.*?) b/", gen_code))
        candidates.extend(re.findall(r"\+\+\+ b/(.*?\.py)", gen_code))

    output: list[str] = []
    for item in candidates:
        cleaned = _clean_path(item)
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output


def split_changed_files(changed_files: list[str]) -> tuple[list[str], list[str]]:
    sqlglot_files: list[str] = []
    auxiliary_files: list[str] = []
    for path in changed_files:
        if path.startswith(_SQLGLOT_PATCH_PREFIXES):
            sqlglot_files.append(path)
        else:
            auxiliary_files.append(path)
    return sqlglot_files, auxiliary_files


def _record_patch_text(record: dict[str, Any]) -> str:
    parts = []
    for key in ("gen_code", "backfilled_gen_code"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def _symbol_from_line(line: str) -> str | None:
    match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Z][A-Z0-9_]{2,})\b", line)
    return match.group(1) if match else None


def _truncate_patch_line(line: str, limit: int = 180) -> str:
    line = re.sub(r"\s+", " ", line.strip())
    return line if len(line) <= limit else line[:limit] + "...[truncated]"


def extract_patch_evidence(record: dict[str, Any], changed_files: list[str]) -> dict[str, Any]:
    sqlglot_files, auxiliary_files = split_changed_files(changed_files)
    patch_text = _record_patch_text(record)
    snippets: list[dict[str, Any]] = []
    symbols: list[str] = []
    current_file: str | None = None
    current: dict[str, Any] | None = None

    def finish_current() -> None:
        nonlocal current
        if current and (
            current.get("added_lines")
            or current.get("removed_lines")
            or current.get("symbols")
        ):
            snippets.append(current)
        current = None

    for raw_line in patch_text.splitlines():
        if raw_line.startswith("diff --git "):
            finish_current()
            match = re.search(r" b/(\S+)$", raw_line)
            current_file = match.group(1) if match else None
            continue
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[len("+++ b/"):].strip()
            continue
        if raw_line.startswith("@@"):
            finish_current()
            if current_file and current_file.startswith(_SQLGLOT_PATCH_PREFIXES):
                current = {
                    "file": current_file,
                    "hunk_header": raw_line[:180],
                    "added_lines": [],
                    "removed_lines": [],
                    "symbols": [],
                }
            continue
        if not current or raw_line.startswith(("+++", "---")):
            continue
        symbol = None
        if raw_line.startswith("+"):
            line = raw_line[1:]
            if line.strip() and len(current["added_lines"]) < 8:
                current["added_lines"].append(_truncate_patch_line(line))
            symbol = _symbol_from_line(line)
        elif raw_line.startswith("-"):
            line = raw_line[1:]
            if line.strip() and len(current["removed_lines"]) < 4:
                current["removed_lines"].append(_truncate_patch_line(line))
            symbol = _symbol_from_line(line)
        else:
            symbol = _symbol_from_line(raw_line)
        if symbol:
            if symbol not in symbols:
                symbols.append(symbol)
            if symbol not in current["symbols"]:
                current["symbols"].append(symbol)
    finish_current()

    if not symbols:
        for symbol in re.findall(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", patch_text):
            if symbol not in symbols:
                symbols.append(symbol)

    return {
        "sqlglot_changed_files": sqlglot_files,
        "auxiliary_changed_files": auxiliary_files,
        "patch_symbols": symbols[:12],
        "patch_snippets": snippets[:4],
        "patch_snippet_count": len(snippets),
        "patch_text_chars": len(patch_text),
    }


def infer_patch_tags(changed_files: list[str], record: dict[str, Any] | None = None) -> list[str]:
    text = " ".join(changed_files)
    gen_code = ""
    if record:
        gen_code = " ".join(
            str(record.get(key) or "")
            for key in ("gen_code", "backfilled_gen_code")
        )
    joined = f"{text}\n{gen_code}".lower()
    tags: list[str] = []
    if "tokenizer" in joined or "tokens" in joined or "keywords" in joined:
        tags.append("patch:tokenizer")
    if "parser" in joined or "_parse" in joined or "functions" in joined:
        tags.append("patch:parser")
    if "generator" in joined or "transforms" in joined or "generator.transforms" in joined:
        tags.append("patch:generator")
    if "sqlglot/transforms.py" in joined:
        tags.append("patch:transform")
    if "sqlglot/expressions.py" in joined:
        tags.append("patch:expression")
    if not tags and any(path.startswith("sqlglot/dialects/") for path in changed_files):
        tags.append("patch:dialect")
    if not tags:
        tags.append("patch:unknown")
    return sorted(dict.fromkeys(tags))
