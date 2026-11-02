"""Small SQL identity helpers shared by memory extraction and retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_sql_for_memory(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"--[^\n\r]*", " ", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = text.strip().rstrip(";").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([(),;=+\-*/<>])\s*", r"\1", text)
    return text.lower()


def sql_pair_identity(
    *,
    src_dialect: Any,
    tgt_dialect: Any,
    source_sql: Any,
    expected_sql: Any,
) -> str:
    source = normalize_sql_for_memory(source_sql)
    expected = normalize_sql_for_memory(expected_sql)
    if not source or not expected:
        return ""
    return json.dumps(
        {
            "src": str(src_dialect or "").lower(),
            "tgt": str(tgt_dialect or "").lower(),
            "source_sql": source,
            "expected_sql": expected,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sql_pair_hash(
    *,
    src_dialect: Any,
    tgt_dialect: Any,
    source_sql: Any,
    expected_sql: Any,
) -> str:
    identity = sql_pair_identity(
        src_dialect=src_dialect,
        tgt_dialect=tgt_dialect,
        source_sql=source_sql,
        expected_sql=expected_sql,
    )
    if not identity:
        return ""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
