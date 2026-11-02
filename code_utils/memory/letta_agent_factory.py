#!/usr/bin/env python3
"""Runtime-only Letta client and response normalization helpers."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

try:
    from letta_client import Letta
except ModuleNotFoundError as exc:
    if exc.name != "letta_client":
        raise
    Letta = None

LETTA_OBJECT_DICT_KEYS = (
    "id",
    "name",
    "agent_type",
    "message_buffer_autoclear",
    "created_at",
    "llm_config",
    "embedding_config",
    "passage_id",
    "label",
    "value",
    "tags",
    "content",
    "text",
    "timestamp",
    "rrf_score",
    "vector_rank",
    "fts_rank",
    "score",
    "passage",
    "results",
    "items",
    "data",
    "agents",
    "passages",
    "messages",
    "message_type",
    "type",
)


def letta_object_to_dict(
    value: Any,
    *,
    attribute_keys: tuple[str, ...] = LETTA_OBJECT_DICT_KEYS,
) -> dict[str, Any]:
    """Convert common Letta SDK result objects into plain dictionaries.

    Native dictionaries and SDK ``model_dump``/``to_dict`` methods are
    preferred. Attribute fallback is limited to known response fields and
    skips callables so methods are never mistaken for response data.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    for method_name in ("to_dict", "model_dump"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                data = method()
            except (TypeError, ValueError, AttributeError):
                data = None
            if isinstance(data, dict):
                return data
    try:
        data = dict(value)
    except (TypeError, ValueError, AttributeError):
        data = {}
    if isinstance(data, dict) and data:
        return data

    data = {}
    for key in attribute_keys:
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        data[key] = item
    return data


def letta_collection_items(value: Any) -> list[Any]:
    """Normalize Letta SDK pages, wrappers, and iterables into response items.

    Supported wrappers cover current and legacy ``items``, ``results``,
    ``data``, ``agents``, ``passages``, and ``messages`` shapes, including
    nested page data. Unrecognized non-collection objects produce an empty
    list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        for key in ("items", "results", "data", "agents", "passages", "messages"):
            items = value.get(key)
            if isinstance(items, list):
                return items
            if isinstance(items, dict):
                nested_items = letta_collection_items(items)
                if nested_items:
                    return nested_items
        return []
    items = getattr(value, "items", None)
    if isinstance(items, list):
        return items
    data = letta_object_to_dict(value)
    if data:
        return letta_collection_items(data)
    try:
        return list(value)
    except TypeError:
        return []


# Client setup and agent id resolution.
def _configured_env_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("__SET_ME"):
        return None
    return value


def letta_client_kwargs(base_url: str | None = None) -> dict[str, Any]:
    """Return consistent Letta SDK kwargs for AutoDialect memory clients."""
    resolved_base_url = base_url or os.getenv("LETTA_BASE_URL", "http://localhost:8284")
    raw_timeout = os.getenv("LETTA_TIMEOUT", "120").strip()
    try:
        timeout = int(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"LETTA_TIMEOUT must be an integer; got {raw_timeout!r}.") from exc
    if timeout <= 0:
        raise ValueError(f"LETTA_TIMEOUT must be greater than zero; got {timeout}.")
    # Runtime retrieval owns its retry policy through LETTA_MEMORY_RETRIES.
    # Disable the generated SDK's hidden retry layer to avoid multiplicative
    # attempts and unexpectedly long stage startup delays.
    kwargs: dict[str, Any] = {
        "base_url": resolved_base_url,
        "timeout": timeout,
        "max_retries": 0,
    }
    api_key = _configured_env_value("LETTA_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def ensure_no_proxy_for_url(base_url: str | None) -> None:
    """Route direct Letta server calls around HTTP proxies when a host is configured."""
    host = urlparse(base_url or "").hostname
    if not host:
        return

    entries: list[str] = []
    seen: set[str] = set()
    for key in ("NO_PROXY", "no_proxy"):
        for raw_item in (os.environ.get(key) or "").split(","):
            item = raw_item.strip()
            if item and item not in seen:
                seen.add(item)
                entries.append(item)
    if host not in seen:
        entries.append(host)

    value = ",".join(entries)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def resolve_agent_id(
    client: Letta,
    *,
    agent_id: str | None = None,
    agent_name: str | None = None,
) -> str:
    """Resolve an explicit id or an exact unique Letta agent name.

    Explicit ids take precedence and avoid a list request. Name lookup rejects
    zero matches, ambiguous matches, and malformed matches without ids.
    """
    if agent_id:
        normalized_agent_id = str(agent_id).strip()
        if normalized_agent_id:
            return normalized_agent_id
    if not agent_name:
        raise ValueError("Either LETTA_AGENT_ID or LETTA_AGENT_NAME must be configured.")

    response = client.agents.list(name=agent_name, limit=10000)
    agents = letta_collection_items(response)
    matches = [
        agent
        for agent in agents
        if letta_object_to_dict(agent).get("name") == agent_name
    ]
    if len(matches) != 1:
        ids = [
            letta_object_to_dict(agent).get("id")
            for agent in matches
        ]
        raise ValueError(
            f"LETTA_AGENT_NAME={agent_name!r} resolved to {len(matches)} agents; "
            f"expected exactly one. Matching ids: {ids}"
        )

    match = matches[0]
    match_dict = letta_object_to_dict(match)
    resolved_agent_id = str(match_dict.get("id") or "").strip()
    if not resolved_agent_id:
        raise ValueError(f"LETTA_AGENT_NAME={agent_name!r} resolved to an agent without an id.")
    return resolved_agent_id
