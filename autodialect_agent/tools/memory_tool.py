# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""Read-only Letta archival memory search wrapper for AutoDialect.

1. Resolve and connect to the configured AutoDialect Letta agent.
2. Provide tag-filtered archival passage search for runtime memory retrieval.
3. Normalize Letta SDK search response shapes and support tag-filtered compact
   passage output for prompt injection.
4. Keep the optional letta_client dependency lazy so non-memory runs can start
   without a local Letta installation.
"""

import sys
from pathlib import Path
from typing import Any, override

# ruff: noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path[:] = [str(ROOT_DIR), *[path for path in sys.path if path != str(ROOT_DIR)]]

from code_utils.constants import DEFAULT_AGENT_ID, DEFAULT_AGENT_NAME, LETTA_BASE_URL
from code_utils.memory.letta_agent_factory import (
    ensure_no_proxy_for_url,
    letta_client_kwargs,
    letta_collection_items,
    letta_object_to_dict,
    resolve_agent_id,
)

try:
    from letta_client import APITimeoutError, Letta
except ModuleNotFoundError as exc:
    if exc.name != "letta_client":
        raise
    Letta = None

    class APITimeoutError(Exception):
        pass

from autodialect_agent.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter

MemoryToolCommands = [
    "search_passages",
]

DEFAULT_LETTA_BASE_URL = LETTA_BASE_URL
DEFAULT_LETTA_AGENT_ID = DEFAULT_AGENT_ID
DEFAULT_LETTA_AGENT_NAME = DEFAULT_AGENT_NAME


class MemoryTool(Tool):
    """Read-only tool used by AutoDialect runtime to search published passages."""

    def __init__(
        self,
        model_provider: str | None = None,
        agent_id: str | None = None,
        agent_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model_provider)
        self._agent_id: str = agent_id or DEFAULT_LETTA_AGENT_ID
        self._agent_name: str = agent_name or DEFAULT_LETTA_AGENT_NAME
        self._client: Letta | None = None
        self._base_url: str = base_url or DEFAULT_LETTA_BASE_URL

    def _require_agent_id(self) -> str:
        if self._agent_id:
            return self._agent_id
        try:
            self._agent_id = resolve_agent_id(
                self._get_client(),
                agent_name=self._agent_name,
            )
        except Exception as exc:
            raise ToolError(
                "Could not resolve Letta agent. Set LETTA_AGENT_ID, or set a unique LETTA_AGENT_NAME."
            ) from exc
        return self._agent_id

    def _get_client(self) -> Letta:
        if self._client is None:
            if Letta is None:
                raise ToolError("The optional `letta_client` package is required for memory operations.")
            ensure_no_proxy_for_url(self._base_url)
            kwargs = letta_client_kwargs(self._base_url)
            self._client = Letta(**kwargs)
        return self._client

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "memory"

    @override
    def get_description(self) -> str:
        return """Search configured long-term memory passages in Letta.
Runtime memory is read-only; AutoDialect does not create or modify passages.
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description=f"The command to run. Allowed options are {', '.join(MemoryToolCommands)}.",
                required=True,
                enum=MemoryToolCommands,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Required for `search_passages`. The semantic query used to find relevant passages.",
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Optional for `search_passages` command. Number of top results to return. Default is 3.",
            ),
            ToolParameter(
                name="tags",
                type="string",
                description=(
                    "Optional for `search_passages` command. Comma-separated Letta passage tags "
                    "used to filter archival search results."
                ),
            ),
            ToolParameter(
                name="tag_match_mode",
                type="string",
                description="Optional for `search_passages` command. Either `all` or `any`. Default is `all` when tags are provided.",
                enum=["all", "any"],
            ),
            ToolParameter(
                name="compact",
                type="boolean",
                description="Optional for `search_passages` command. Return a compact passage summary suitable for prompt injection.",
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        command = str(arguments.get("command")) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )

        try:
            match command:
                case "search_passages":
                    return self._search_passages(arguments)
                case _:
                    return ToolExecResult(error=f"Invalid command: {command}", error_code=-1)
        except APITimeoutError:
            return ToolExecResult(
                error=f"Request timed out while executing {command}. Please check if the Letta server is accessible.",
                error_code=-1,
            )
        except Exception as e:
            return ToolExecResult(error=f"Error executing {command}: {str(e)}", error_code=-1)

    def _search_passages(self, arguments: ToolCallArguments) -> ToolExecResult:
        query = str(arguments.get("query")) if "query" in arguments else None
        if query is None:
            return ToolExecResult(
                error="No query provided for search_passages command",
                error_code=-1,
            )

        top_k = int(arguments.get("top_k", 3)) if "top_k" in arguments else 3
        tags = self._parse_tags(arguments.get("tags"))
        tag_match_mode = str(arguments.get("tag_match_mode") or ("all" if tags else "")).strip() or None
        compact = self._parse_bool(arguments.get("compact"))
        if tag_match_mode and tag_match_mode not in {"all", "any"}:
            return ToolExecResult(
                error=f"Invalid tag_match_mode: {tag_match_mode}. Expected `all` or `any`.",
                error_code=-1,
            )

        client = self._get_client()
        agent_id = self._require_agent_id()
        search_kwargs = {
            "agent_id": agent_id,
            "query": query,
            "top_k": top_k,
        }
        if tags:
            search_kwargs["tags"] = tags
            search_kwargs["tag_match_mode"] = tag_match_mode or "all"
        results = client.agents.passages.search(**search_kwargs)
        items = self._normalize_passage_search_results(results)

        output = f"Found {len(items)} archival memory passage(s) for query: '{query}'\n"
        if tags:
            output += f"Tag filter: {tags} (mode={tag_match_mode or 'all'})\n"
        output += "=" * 80 + "\n"

        for idx, item in enumerate(items, 1):
            passage_id = item.get('id') or 'N/A'
            content = item.get('content') or item.get('text') or ''
            timestamp = item.get('timestamp', '')
            tags = item.get('tags', [])
            relevance = {
                key: item.get(key)
                for key in ("rrf_score", "vector_rank", "fts_rank", "score")
                if item.get(key) is not None
            }

            output += f"\n{idx}. Passage ID: {passage_id}\n"
            output += f"   Timestamp: {timestamp}\n"
            if tags:
                output += f"   Tags: {tags}\n"
            if relevance:
                output += f"   Relevance: {relevance}\n"
            output += "   Content:\n"
            output += f"   {'-' * 76}\n"
            content_lines = (
                self._compact_passage_lines(content)
                if compact else content.split('\n')
            )
            for line in content_lines:
                output += f"   {line}\n"
            output += f"   {'-' * 76}\n"
            output += "=" * 80 + "\n"

        return ToolExecResult(output=output)

    @staticmethod
    def _first_present(*values: Any) -> Any:
        for value in values:
            if value not in (None, "", [], {}):
                return value
        return None

    @classmethod
    def _normalize_passage_search_results(cls, results: Any) -> list[dict[str, Any]]:
        """Normalize supported Letta search pages and nested passage records."""
        normalized: list[dict[str, Any]] = []
        for raw_item in letta_collection_items(results):
            item = letta_object_to_dict(raw_item)
            passage = letta_object_to_dict(item.get("passage"))
            if not passage:
                try:
                    passage = letta_object_to_dict(raw_item.passage)
                except Exception:
                    passage = {}
            normalized.append(
                {
                    "id": cls._first_present(
                        item.get("id"),
                        item.get("passage_id"),
                        passage.get("id"),
                    ),
                    "content": cls._first_present(
                        item.get("content"),
                        item.get("text"),
                        passage.get("content"),
                        passage.get("text"),
                    )
                    or "",
                    "timestamp": cls._first_present(
                        item.get("timestamp"),
                        passage.get("timestamp"),
                    )
                    or "",
                    "tags": cls._first_present(item.get("tags"), passage.get("tags")) or [],
                    "rrf_score": cls._first_present(item.get("rrf_score"), passage.get("rrf_score")),
                    "vector_rank": cls._first_present(item.get("vector_rank"), passage.get("vector_rank")),
                    "fts_rank": cls._first_present(item.get("fts_rank"), passage.get("fts_rank")),
                    "score": cls._first_present(item.get("score"), passage.get("score")),
                }
            )
        return normalized

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _compact_passage_lines(content: str, limit: int = 18) -> list[str]:
        keep_prefixes = (
            "schema:",
            "id:",
            "kind:",
            "labels:",
            "pair:",
            "scope:",
            "feature:",
            "lesson:",
            "tags:",
            "evidence:",
            "- case:",
            "- outcome:",
            "- rule_group_id:",
            "- split_ids:",
            "- files:",
            "- sql_pair_hash:",
        )
        kept: list[str] = []
        memory_lines_remaining = 4
        in_memory = False
        for raw_line in (content or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line in {"AutoDialect Long-term Memory", "Task:", "Memory:", "Evidence:"}:
                kept.append(line)
                in_memory = line == "Memory:"
            elif line.startswith(keep_prefixes):
                kept.append(line)
            elif in_memory and memory_lines_remaining > 0:
                kept.append(line)
                memory_lines_remaining -= 1
            if len(kept) >= limit:
                break
        return kept or (content or "").splitlines()[:limit]

    @staticmethod
    def _parse_tags(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    import json

                    loaded = json.loads(text)
                    if isinstance(loaded, list):
                        return [str(item).strip() for item in loaded if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []
