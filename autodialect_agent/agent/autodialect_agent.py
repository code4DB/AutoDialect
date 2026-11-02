# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""AutoDialectAgent implementation used by AutoDialect case execution.

1. Build the task prompt from project, issue, memory, and completion-validation
   context before entering the base agent loop.
2. Retrieve compact Letta memory passages when memory is enabled and record
   retrieval audit details for experiment records. Runtime long-term memory is
   read-only; runtime execution does not construct memory content.
3. Enforce AutoDialect completion validation after task_done so invalid or
   empty patches are rejected before the outer pipeline accepts success.
4. Keep the tool surface limited to the SQLGlot-focused main path.
"""

import json
import os
import re
import subprocess
from dataclasses import replace
from typing import Any, override

from code_utils.common_tool import (
    is_legacy_family_key as _is_legacy_family_key,
)
from code_utils.common_tool import (
    scrub_legacy_family_text as _scrub_legacy_family_text,
)
from code_utils.common_tool import (
    strip_legacy_family_metadata as _strip_legacy_family_metadata,
)
from code_utils.completion_validation_runtime import (
    build_completion_validation_audit_section,
    format_completion_validation_message,
    load_completion_validation_spec,
    run_completion_validation,
)
from autodialect_agent.agent.agent_basics import AgentError, AgentExecution
from autodialect_agent.agent.base_agent import BaseAgent, available_configured_tool_names
from autodialect_agent.prompt.agent_prompt import (
    EMPTY_PATCH_RETRY_PROMPT,
    AUTODIALECT_AGENT_SYSTEM_PROMPT,
    AUTODIALECT_ISSUE_CONTEXT_PROMPT,
    AUTODIALECT_MEMORY_CONTEXT_PROMPT,
    AUTODIALECT_PROJECT_CONTEXT_PROMPT,
)
from autodialect_agent.tools import tools_registry
from autodialect_agent.tools.base import Tool, ToolExecutor, ToolResult
from autodialect_agent.tools.memory_tool import MemoryTool
from autodialect_agent.utils.config import AutoDialectAgentConfig
from autodialect_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from autodialect_agent.utils.llm_clients.retry_utils import async_retry_with

AutoDialectAgentToolNames = [
    "str_replace_based_edit_tool",
    "sequentialthinking",
    "task_done",
    "bash",
]


def _item_get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _history_item_has_task_done(item: Any) -> bool:
    if _item_get(item, "name") == "task_done":
        return True
    tool_calls = _item_get(item, "tool_calls") or []
    for tool_call in tool_calls:
        if _item_get(tool_call, "name") == "task_done":
            return True
        function = _item_get(tool_call, "function")
        if _item_get(function, "name") == "task_done":
            return True
    return False


# 1. Task prompt setup, memory retrieval, and completion validation.
class AutoDialectAgent(BaseAgent):
    """AutoDialect Agent specialized for software engineering tasks."""

    def __init__(self, autodialect_agent_config: AutoDialectAgentConfig):
        """Initialize AutoDialectAgent.

        Args:
            config: Configuration object containing model parameters and other settings.
                   Required if llm_client is not provided.
            llm_client: Optional pre-configured LLMClient instance.
                       If provided, it will be used instead of creating a new one from config.
        """
        self.project_path: str = ""
        self.base_commit: str | None = None
        self.must_patch: str = "false"
        self.patch_path: str | None = None
        self.completion_validation_path: str | None = None
        self._completion_validation_failed_task_done_count = 0
        self._completion_validation_message: str | None = None
        self._completion_validation_spec_cache: dict[str, Any] | None = None
        self._completion_validation_last_result: dict[str, Any] | None = None
        self.enable_memory = autodialect_agent_config.enable_memory
        self.memory_audit: dict[str, object] = {
            "enabled": self.enable_memory,
            "tool_initialized": False,
            "retrieval": {"attempted": False},
            "completion_validation": {
                "enabled": False,
                "attempts": 0,
                "failed_task_done_count": 0,
            },
        }

        # Memory is opt-in because it talks to an external Letta service.
        self.memory_tool: MemoryTool | None = None
        if self.enable_memory:
            try:
                self.memory_tool = MemoryTool()
                self.memory_audit["tool_initialized"] = True
            except Exception as e:
                self.memory_audit["init_error"] = str(e)
                print(f"⚠️  Failed to initialize MemoryTool: {e}")

        base_agent_config = autodialect_agent_config
        # Memory is an internal read-only retrieval channel for AutoDialect.
        if "memory" in autodialect_agent_config.tools:
            base_agent_config = replace(
                autodialect_agent_config,
                tools=[tool_name for tool_name in autodialect_agent_config.tools if tool_name != "memory"],
            )

        super().__init__(agent_config=base_agent_config)

    # 1. Task prompt setup.
    @override
    def new_task(
            self,
            task: str,
            extra_args: dict[str, str] | None = None,
            tool_names: list[str] | None = None,
    ):
        """Create a new task."""
        if not extra_args:
            raise AgentError("Project path and issue information are required.")

        self._task: str = task
        self.memory_audit = {
            "enabled": self.enable_memory,
            "tool_initialized": self.memory_tool is not None,
            "retrieval": {
                "attempted": False,
                "method": "search_passages",
                "injected": False,
            },
            "completion_validation": {
                "enabled": bool(extra_args.get("completion_validation_path")),
                "spec_path": extra_args.get("completion_validation_path"),
                "attempts": 0,
                "failed_task_done_count": 0,
            },
        }

        if tool_names is None and len(self._tools) == 0:
            tool_names = AutoDialectAgentToolNames

        if tool_names is not None:
            tool_names = [tool_name for tool_name in tool_names if tool_name != "memory"]
        if tool_names is not None:
            tool_names = available_configured_tool_names(tool_names)

        if tool_names is not None and len(self._tools) == 0:
            # Get the model provider from the LLM client
            provider = self._model_config.model_provider.provider
            self._tools: list[Tool] = [
                tools_registry[tool_name](model_provider=provider) for tool_name in tool_names
            ]
        self._tool_caller: ToolExecutor = ToolExecutor(self._tools)

        self._initial_messages: list[LLMMessage] = []
        self._initial_messages.append(LLMMessage(role="system", content=self.get_system_prompt()))

        user_message = ""
        if "project_path" not in extra_args:
            raise AgentError("Project path is required")

        self.project_path = extra_args.get("project_path", "")
        if not self.project_path or not os.path.isabs(self.project_path):
            raise AgentError(
                f"Project path must be a non-empty absolute path: {self.project_path!r}"
            )
        for tool in self._tools:
            tool.set_project_path(self.project_path)
        user_message += AUTODIALECT_PROJECT_CONTEXT_PROMPT.format(project_path=self.project_path)

        if "issue" in extra_args:
            user_message += AUTODIALECT_ISSUE_CONTEXT_PROMPT.format(issue=extra_args["issue"])

        self._initial_messages.append(LLMMessage(role="user", content=user_message))

        self.completion_validation_path = None
        self._completion_validation_failed_task_done_count = 0
        self._completion_validation_message = None
        self._completion_validation_spec_cache = None
        self._completion_validation_last_result = None

        optional_attrs_to_set = [
            "base_commit",
            "must_patch",
            "patch_path",
            "completion_validation_path",
        ]
        for attr in optional_attrs_to_set:
            if attr in extra_args:
                setattr(self, attr, extra_args[attr])

        # If trajectory recorder is set, start recording
        if self._trajectory_recorder:
            self._trajectory_recorder.start_recording(
                task=task,
                provider=self._llm_client.provider.value,
                model=self._model_config.model,
                max_steps=self._max_steps,
            )

    # 2. Memory retrieval.
    @override
    async def execute_task(self) -> AgentExecution:
        """Execute the task and finalize trajectory recording."""
        if self.enable_memory and self.memory_tool:
            memory_context = await self._retrieve_memory_context(self._task)
            if memory_context and self._initial_messages:
                user_message = self._initial_messages[-1]
                user_message.content = AUTODIALECT_MEMORY_CONTEXT_PROMPT.format(
                    task_context=user_message.content or "",
                    memory_context=memory_context,
                )
                retrieval = self.memory_audit.setdefault("retrieval", {})
                if isinstance(retrieval, dict):
                    retrieval["injected"] = True

        execution = await super().execute_task()

        # Finalize trajectory recording if recorder is available
        if self._trajectory_recorder:
            self._trajectory_recorder.finalize_recording(
                success=execution.success, final_result=execution.final_result
            )

        if self.patch_path is not None:
            if not os.path.exists(os.path.dirname(self.patch_path)):
                os.makedirs(os.path.dirname(self.patch_path))
            with open(self.patch_path, "w") as patch_f:
                _ = patch_f.write(self.get_git_diff())

        return execution

    def get_system_prompt(self) -> str:
        """Get the system prompt for AutoDialectAgent."""
        return AUTODIALECT_AGENT_SYSTEM_PROMPT

    async def _retrieve_memory_context(self, task: str) -> str | None:
        """Retrieve relevant memory context through Letta archival passage search."""
        if not self.memory_tool:
            self.memory_audit["retrieval"] = {
                "attempted": False,
                "method": "search_passages",
                "injected": False,
                "error": "MemoryTool is not initialized",
            }
            return None

        memory_requests = self._build_memory_search_requests(task)
        self.memory_audit["retrieval"] = {
            "attempted": bool(memory_requests),
            "method": "search_passages",
            "requests": [
                {
                    "label": request["label"],
                    "tags": request["tags"],
                    "top_k": request["top_k"],
                    "query_length": len(str(request["query"])),
                    "exclude_case_ids": request.get("exclude_case_ids", []),
                    "exclude_sql_pair_hashes": request.get("exclude_sql_pair_hashes", []),
                }
                for request in memory_requests
            ],
            "injected": False,
        }
        if not memory_requests:
            self.memory_audit["retrieval"]["skipped"] = True
            self.memory_audit["retrieval"]["reason"] = "structured_memory_query_unavailable"
            return None

        try:
            search_errors: list[str] = []
            search_call_count = 0
            raw_attempts = os.getenv("LETTA_MEMORY_RETRIES", "3").strip()
            try:
                configured_attempts = int(raw_attempts)
            except ValueError as exc:
                raise ValueError(
                    f"LETTA_MEMORY_RETRIES must be an integer; got {raw_attempts!r}."
                ) from exc
            max_retries = max(0, configured_attempts - 1)

            async def _search_once(request):
                nonlocal search_call_count
                search_call_count += 1
                result = await self.memory_tool.execute({
                    "command": "search_passages",
                    "query": request["query"],
                    "top_k": request["top_k"],
                    "tags": ",".join(request["tags"]),
                    "tag_match_mode": "all",
                    "compact": True,
                })
                if result.error:
                    label = str(request.get("label") or "unknown")
                    search_errors.append(f"{label}: {result.error}")
                    retrieval = self.memory_audit.setdefault("retrieval", {})
                    if isinstance(retrieval, dict):
                        retrieval["attempts"] = search_call_count
                        retrieval["errors"] = search_errors
                    raise RuntimeError(result.error)
                return result

            outputs: list[tuple[dict[str, object], str]] = []
            for request in memory_requests:
                retry_search = async_retry_with(
                    lambda request=request: _search_once(request),
                    provider_name=f"Letta memory passage search ({request['label']})",
                    max_retries=max_retries,
                )
                try:
                    result = await retry_search()
                    if result.output:
                        outputs.append((request, result.output))
                except Exception as e:
                    retrieval = self.memory_audit.setdefault("retrieval", {})
                    if isinstance(retrieval, dict):
                        retrieval["partial_failure"] = True
                        retrieval["attempts"] = search_call_count
                        retrieval["errors"] = search_errors
                        retrieval["error"] = str(e)
                    print(f"⚠️  Memory search error for {request['label']}: {e}")

            if not outputs:
                retrieval = self.memory_audit.setdefault("retrieval", {})
                if isinstance(retrieval, dict):
                    retrieval["success"] = False
                    retrieval["attempts"] = search_call_count
                    retrieval["errors"] = search_errors
                    retrieval["error"] = "all memory passage searches failed or returned empty output"
                return None

            retrieval = self.memory_audit.setdefault("retrieval", {})
            if isinstance(retrieval, dict):
                retrieval["success"] = True
                retrieval["attempts"] = search_call_count
                if search_errors:
                    retrieval["errors"] = search_errors
                retrieval["raw_output_length"] = sum(len(output) for _, output in outputs)

            budgets = {"relevant": 2400}
            accepted_sections: list[str] = []
            filter_audits: list[dict[str, object]] = []
            for request, output in outputs:
                required_tags = self._memory_required_context_tags(request)
                filtered_section = self._filter_memory_context(
                    output,
                    required_tags=required_tags,
                    exclude_case_ids=[
                        str(item)
                        for item in request.get("exclude_case_ids", [])
                        if item
                    ],
                    exclude_sql_pair_hashes=[
                        str(item)
                        for item in request.get("exclude_sql_pair_hashes", [])
                        if item
                    ],
                )
                current_filter = retrieval.get("filter") if isinstance(retrieval, dict) else None
                filter_audits.append({
                    "label": request["label"],
                    **(dict(current_filter) if isinstance(current_filter, dict) else {}),
                    "relevance": self._memory_relevance_audit(
                        str(request.get("query") or ""),
                        filtered_section or "",
                        required_tags=required_tags,
                    ),
                })
                if filtered_section:
                    label = str(request["label"])
                    budget = budgets.get(label, 500)
                    accepted_sections.append(
                        f"[{label}]\n{filtered_section[:budget].rstrip()}"
                    )

            filtered = "\n\n".join(accepted_sections) or None
            retrieval = self.memory_audit.setdefault("retrieval", {})
            if isinstance(retrieval, dict):
                excluded_case_ids = sorted({
                    case_id
                    for item in filter_audits
                    for case_id in item.get("excluded_case_ids", [])
                })
                excluded_sql_pair_hashes = sorted({
                    hash_value
                    for item in filter_audits
                    for hash_value in item.get("excluded_sql_pair_hashes", [])
                })
                self_exclusion_case_ids = sorted({
                    case_id
                    for item in filter_audits
                    for case_id in item.get("self_exclusion_case_ids", [])
                })
                self_exclusion_sql_pair_hashes = sorted({
                    hash_value
                    for item in filter_audits
                    for hash_value in item.get("self_exclusion_sql_pair_hashes", [])
                })
                retrieval["filters"] = filter_audits
                retrieval["filter"] = {
                    "accepted_labels": [
                        item["label"] for item in filter_audits if item.get("accepted")
                    ],
                    "rejected_labels": [
                        item["label"] for item in filter_audits if not item.get("accepted")
                    ],
                    "self_exclusion_case_ids": self_exclusion_case_ids,
                    "self_exclusion_sql_pair_hashes": self_exclusion_sql_pair_hashes,
                    "excluded_case_ids": excluded_case_ids,
                    "excluded_sql_pair_hashes": excluded_sql_pair_hashes,
                    "excluded_passage_count": sum(
                        int(item.get("excluded_passage_count") or 0)
                        for item in filter_audits
                    ),
                    "accepted_passage_count": sum(
                        int(item.get("accepted_passage_count") or 0)
                        for item in filter_audits
                    ),
                    "accepted": bool(filtered),
                }
                retrieval["output_length"] = len(filtered or "")
                retrieval["accepted"] = bool(filtered)
            return filtered
        except Exception as e:
            retrieval = self.memory_audit.setdefault("retrieval", {})
            if isinstance(retrieval, dict):
                retrieval["success"] = False
                retrieval["error"] = str(e)
            print(f"⚠️  Failed to retrieve memory context: {e}")
            return None

    @staticmethod
    def _memory_required_context_tags(memory_request: dict[str, object]) -> list[str]:
        tags: list[str] = ["autodialect_memory_asset", "schema:autodialect_ltm_v3"]
        for tag in memory_request.get("tags") or []:
            if isinstance(tag, str) and tag.startswith(("src:", "tgt:", "kind:")):
                tags.append(tag)
        return sorted(dict.fromkeys(tags))

    def _build_memory_search_requests(self, task: str) -> list[dict[str, object]]:
        query = self._build_memory_retrieval_query(task)
        try:
            data = json.loads(query)
        except Exception:
            data = None

        if not isinstance(data, dict):
            return []

        repair_target = data.get("repair_target") if isinstance(data.get("repair_target"), dict) else {}
        repair_residual = data.get("repair_residual") if isinstance(data.get("repair_residual"), dict) else {}
        is_repair_query = bool(repair_target or repair_residual)
        current_case_id = str(data.get("current_case_id") or "").strip()
        current_sql_pair_hash = str(data.get("current_sql_pair_hash") or "").strip()
        terms = ["AutoDialect Long-term Memory", "schema:autodialect_ltm_v3"]
        tags = ["autodialect_memory_asset", "schema:autodialect_ltm_v3"]
        if is_repair_query:
            terms.extend([
                "kind:experience",
                "kind:risk",
                "kind:guard",
                "label:correction",
                "repair_then_passed",
                "validated location",
                "process guard",
            ])
        else:
            terms.extend([
                "kind:experience",
                "kind:risk",
                "kind:guard",
                "label:navigation",
                "label:implementation",
                "validated SQLGlot change",
                "variant composition process guard",
            ])
        src = data.get("src_dialect") or data.get("source_dialect")
        tgt = data.get("tgt_dialect") or data.get("target_dialect")
        if src:
            terms.extend([f"src:{src}", str(src)])
            tags.append(f"src:{src}")
        if tgt:
            terms.extend([f"tgt:{tgt}", str(tgt)])
            tags.append(f"tgt:{tgt}")

        if not is_repair_query:
            terms.extend(["focused_validation_passed", "implementation", "navigation"])

        skipped_evidence_keys = {"family", "rule_family", "family_hint", "confidence"}

        def add_value(value):
            if value is None:
                return
            if isinstance(value, str):
                scrubbed = _scrub_legacy_family_text(value)
                if scrubbed and len(" ".join(terms)) < 3500:
                    terms.append(scrubbed)
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in skipped_evidence_keys or _is_legacy_family_key(key):
                        continue
                    add_value(item)
                return
            if isinstance(value, list):
                for item in value[:8]:
                    add_value(item)

        for key in (
            "group_signature",
            "source_sql",
            "expected_sql",
            "source_expr_sql",
            "actual_expr_sql",
            "expected_expr_sql",
            "selected_target_diff",
            "transition_summary",
            "groups",
            "units",
            "memory_units",
            "repair_target",
            "repair_residual",
        ):
            add_value(data.get(key))
        base_query = "\n".join(str(term) for term in terms if term)[:5000]
        return [{
            "label": "relevant",
            "query": base_query,
            "tags": sorted(dict.fromkeys(tags)),
            "top_k": 5,
            "exclude_case_ids": [current_case_id] if current_case_id else [],
            "exclude_sql_pair_hashes": [current_sql_pair_hash] if current_sql_pair_hash else [],
        }]

    def _build_memory_retrieval_query(self, task: str) -> str:
        """Extract the structured AutoDialect memory query JSON from the prompt."""
        marker = "[AutoDialect Memory Query]"
        marker_index = (task or "").find(marker)
        if marker_index >= 0:
            marker_text = (task or "")[marker_index + len(marker):]
            fenced = re.search(r"```json\s*(.*?)\s*```", marker_text, flags=re.DOTALL)
            if fenced:
                return fenced.group(1).strip()[:5000]
            return marker_text.strip()[:5000]
        return ""

    def _filter_memory_context(
        self,
        output: str,
        required_tags: list[str],
        *,
        exclude_case_ids: list[str] | None = None,
        exclude_sql_pair_hashes: list[str] | None = None,
    ) -> str | None:
        text = (output or "").strip()
        if not text:
            return None
        exclude_case_set = {str(item).strip().lower() for item in exclude_case_ids or [] if str(item).strip()}
        exclude_hash_set = {str(item).strip().lower() for item in exclude_sql_pair_hashes or [] if str(item).strip()}
        blocks = self._memory_output_blocks(text)
        accepted_passages: list[str] = []
        audits: list[dict[str, object]] = []
        for block in blocks:
            passage_text, audit = self._filter_memory_context_block(
                block,
                required_tags,
                exclude_case_set=exclude_case_set,
                exclude_hash_set=exclude_hash_set,
            )
            audits.append(audit)
            if passage_text:
                accepted_passages.append(passage_text)

        accepted = bool(accepted_passages)
        missing_required_tags = sorted({
            tag
            for audit in audits
            for tag in audit.get("missing_required_tags", [])
        })
        tag_metadata_tokens = sorted({
            token
            for audit in audits
            for token in audit.get("tag_metadata_tokens", [])
        })[:20]
        missing_evidence_markers = sorted({
            marker
            for audit in audits
            for marker in audit.get("missing_evidence_markers", [])
        })
        retrieval = self.memory_audit.setdefault("retrieval", {})
        if isinstance(retrieval, dict):
            retrieval["filter"] = {
                "required_tags": required_tags,
                "missing_required_tags": [] if accepted else missing_required_tags,
                "tag_metadata_present": any(audit.get("tag_metadata_present") for audit in audits),
                "tag_metadata_tokens": tag_metadata_tokens,
                "required_tags_satisfied": accepted or any(
                    audit.get("required_tags_satisfied") for audit in audits
                ),
                "evidence_required_satisfied": accepted or any(
                    audit.get("evidence_required_satisfied") for audit in audits
                ),
                "missing_evidence_markers": [] if accepted else missing_evidence_markers,
                "has_asset_marker": any(audit.get("has_asset_marker") for audit in audits),
                "evidence_count": max(
                    [int(audit.get("evidence_count") or 0) for audit in audits] or [0]
                ),
                "self_exclusion_case_ids": sorted(exclude_case_set),
                "self_exclusion_sql_pair_hashes": sorted(exclude_hash_set),
                "excluded_case_ids": sorted({
                    case_id
                    for audit in audits
                    for case_id in audit.get("excluded_cases", [])
                }),
                "excluded_sql_pair_hashes": sorted({
                    hash_value
                    for audit in audits
                    for hash_value in audit.get("excluded_sql_pair_hashes", [])
                }),
                "excluded_passage_count": sum(1 for audit in audits if audit.get("excluded")),
                "accepted_passage_count": len(accepted_passages),
                "accepted": accepted,
            }
        if not accepted:
            return None
        return _scrub_legacy_family_text("\n\n".join(accepted_passages))[:4000]

    @staticmethod
    def _memory_output_blocks(text: str) -> list[str]:
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in (text or "").splitlines():
            if re.match(r"^\s*\d+\.\s+Passage ID:", line) and current:
                blocks.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append(current)
        return ["\n".join(block).strip() for block in blocks if "\n".join(block).strip()]

    def _filter_memory_context_block(
        self,
        text: str,
        required_tags: list[str],
        *,
        exclude_case_set: set[str],
        exclude_hash_set: set[str],
    ) -> tuple[str | None, dict[str, object]]:
        passage_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith(("Found ", "Tag filter:", "Timestamp:", "Relevance:"))
                or " for query: " in stripped
                or stripped == "Content:"
                or (stripped and set(stripped) <= {"=", "-"})
                or re.match(r"^\d+\. Passage ID:", stripped)
            ):
                continue
            passage_lines.append(line)
        passage_text = "\n".join(passage_lines)
        lower_text = passage_text.lower()
        tag_metadata_text = "\n".join(
            line
            for line in text.splitlines()
            if re.match(r"^\s*tags\s*:", line, flags=re.IGNORECASE)
        ).lower()
        tag_metadata_tokens = {
            token.lower()
            for token in re.findall(
                r"\bautodialect_memory_asset\b|\b[A-Za-z_][A-Za-z0-9_]*:[A-Za-z0-9_.-]+\b",
                tag_metadata_text,
            )
        }
        content_case_ids = {
            match.strip().lower()
            for match in re.findall(r"(?im)^\s*-?\s*case\s*:\s*([A-Za-z0-9_.-]+)\s*$", passage_text)
        }
        content_hashes = {
            match.strip().lower()
            for match in re.findall(r"(?im)^\s*-?\s*sql_pair_hash\s*:\s*([A-Za-z0-9_.-]+)\s*$", passage_text)
        }
        tag_cases = {
            token.split(":", 1)[1]
            for token in tag_metadata_tokens
            if token.startswith("case:")
        }
        tag_hashes = {
            token.split(":", 1)[1]
            for token in tag_metadata_tokens
            if token.startswith("sql_pair_hash:")
        }
        excluded_cases = sorted(exclude_case_set & (content_case_ids | tag_cases))
        excluded_hashes = sorted(exclude_hash_set & (content_hashes | tag_hashes))
        excluded = bool(excluded_cases or excluded_hashes)
        missing_required_tags = [
            tag
            for tag in required_tags
            if tag.lower() not in tag_metadata_tokens
        ]
        has_asset_marker = "autodialect memory asset" in lower_text or "autodialect_memory_asset" in lower_text
        required_evidence_markers = ("feature:", "lesson:", "evidence:")
        missing_evidence_markers = [
            marker for marker in required_evidence_markers if marker not in lower_text
        ]
        evidence_count = len(required_evidence_markers) - len(missing_evidence_markers)
        required_tags_satisfied = not missing_required_tags
        evidence_required_satisfied = not missing_evidence_markers
        accepted = (
            has_asset_marker
            and required_tags_satisfied
            and evidence_required_satisfied
            and not excluded
        )
        audit = {
            "required_tags": required_tags,
            "missing_required_tags": missing_required_tags,
            "tag_metadata_present": bool(tag_metadata_text),
            "tag_metadata_tokens": sorted(tag_metadata_tokens)[:20],
            "required_tags_satisfied": required_tags_satisfied,
            "evidence_required_satisfied": evidence_required_satisfied,
            "missing_evidence_markers": missing_evidence_markers,
            "has_asset_marker": has_asset_marker,
            "evidence_count": evidence_count,
            "excluded": excluded,
            "excluded_cases": excluded_cases,
            "excluded_sql_pair_hashes": excluded_hashes,
            "accepted": accepted,
        }
        if not accepted:
            return None, audit
        return _scrub_legacy_family_text(passage_text), audit

    @staticmethod
    def _memory_relevance_audit(
        query: str,
        filtered_section: str,
        *,
        required_tags: list[str],
    ) -> dict[str, object]:
        tag_tokens = {
            token.lower()
            for token in re.findall(
                r"\b[A-Za-z_][A-Za-z0-9_]*:[A-Za-z0-9_.-]+\b",
                filtered_section or "",
            )
        }
        query_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query or "")
            if token.lower()
            not in {
                "autodialect",
                "long",
                "term",
                "memory",
                "schema",
                "source",
                "expected",
                "actual",
                "select",
                "from",
                "where",
                "group",
                "order",
                "rule",
            }
        }
        feature_tokens = {
            token.split(":", 1)[1]
            for token in tag_tokens
            if token.startswith("feature:")
        }
        file_tags = sorted(token for token in tag_tokens if token.startswith("file:"))
        asset_ids = sorted(
            token.split(":", 1)[1] for token in tag_tokens if token.startswith("asset_id:")
        )
        overlap = sorted(query_tokens & feature_tokens)
        return {
            "asset_ids": asset_ids[:10],
            "file_tags": file_tags[:10],
            "required_tags": required_tags,
            "query_feature_overlap": overlap[:20],
            "query_feature_overlap_count": len(overlap),
            "feature_tag_count": len(feature_tokens),
        }

    @override
    def reflect_on_result(self, tool_results: list[ToolResult]) -> str | None:
        return None

    def get_git_diff(self) -> str:
        """Get the git diff of the project."""
        if not self.project_path or not os.path.isdir(self.project_path):
            return ""
        pwd = os.getcwd()
        try:
            os.chdir(self.project_path)
            if not self.base_commit:
                stdout = subprocess.check_output(["git", "--no-pager", "diff"]).decode()
            else:
                stdout = subprocess.check_output(
                    ["git", "--no-pager", "diff", self.base_commit, "HEAD"]
                ).decode()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            stdout = ""
        finally:
            os.chdir(pwd)
        return stdout

    # Copyright (c) 2024 paul-gauthier
    # SPDX-License-Identifier: Apache-2.0
    # Original remove_patches_to_tests function was released under Apache-2.0 License, with the full license text
    # available at https://github.com/Aider-AI/aider-swe-bench/blob/6e98cd6c3b2cbcba12976d6ae1b07f847480cb74/LICENSE.txt
    # Original function is at https://github.com/Aider-AI/aider-swe-bench/blob/6e98cd6c3b2cbcba12976d6ae1b07f847480cb74/tests.py#L45

    # 3. Completion validation.
    def remove_patches_to_tests(self, model_patch: str) -> str:
        """
        Remove any changes to the tests directory from the provided patch.
        This is to ensure that the model_patch does not disturb the repo's
        tests when doing acceptance testing with the `test_patch`.
        """
        lines = model_patch.splitlines(keepends=True)
        filtered_lines: list[str] = []
        test_patterns = ["/test/", "/tests/", "/testing/", "test_", "tox.ini"]
        is_tests = False

        for line in lines:
            if line.startswith("diff --git a/"):
                target_path = line.split()[-1]
                is_tests = target_path.startswith("b/") and any(
                    p in target_path for p in test_patterns
                )

            if not is_tests:
                filtered_lines.append(line)

        return "".join(filtered_lines)

    def _load_completion_validation_spec(self) -> dict[str, Any] | None:
        if not self.completion_validation_path:
            return None
        if self._completion_validation_spec_cache is not None:
            return self._completion_validation_spec_cache
        payload = load_completion_validation_spec(self.completion_validation_path)
        self._completion_validation_spec_cache = payload
        return payload

    def _run_completion_validation(self) -> dict[str, Any] | None:
        spec = self._load_completion_validation_spec()
        return run_completion_validation(
            project_path=getattr(self, "project_path", ""),
            completion_validation_path=self.completion_validation_path,
            spec=spec,
        )

    def _format_completion_validation_message(self, validation: dict[str, Any]) -> str:
        return format_completion_validation_message(validation)

    def _empty_patch_validation_result(self) -> dict[str, Any]:
        spec = self._load_completion_validation_spec()
        if not isinstance(spec, dict):
            spec = {}
        return {
            "passed": False,
            "failed_stage": "empty_patch",
            "validation_scope": spec.get("validation_scope"),
            "focus_rule_group_id": spec.get("focus_rule_group_id"),
            "focus_rule_group_ids": spec.get("focus_rule_group_ids"),
            "repair_hint": "The task_done call was rejected because no non-test patch was produced.",
            "repair_target": {
                "stage": "empty_patch",
                "reason": "no_non_test_patch",
                "instruction": "Modify production code, not only tests, before calling task_done.",
            },
        }

    def _record_completion_validation_audit(
        self,
        validation: dict[str, Any] | None,
        *,
        rejected: bool = False,
        completed: bool = False,
        trigger: str = "task_done",
    ) -> None:
        audit = getattr(self, "memory_audit", None)
        if not isinstance(audit, dict):
            return
        previous_section = audit.get("completion_validation")
        if not isinstance(previous_section, dict):
            previous_section = {}
        spec = self._completion_validation_spec_cache if isinstance(self._completion_validation_spec_cache, dict) else {}
        section = build_completion_validation_audit_section(
            validation,
            previous_section=previous_section,
            spec=spec,
            completion_validation_path=self.completion_validation_path,
            failed_task_done_count=self._completion_validation_failed_task_done_count,
            rejected=rejected,
            completed=completed,
        )
        section["completion_trigger"] = trigger
        if trigger != "task_done":
            if section.pop("rejected_current_task_done", None):
                section["rejected_current_attempt"] = True
            if section.pop("completed_current_task_done", None):
                section["completed_current_attempt"] = True
        audit["completion_validation"] = section

    def _discard_last_task_done_from_history(self) -> None:
        llm_client = getattr(self, "_llm_client", None)
        provider_client = getattr(llm_client, "client", llm_client)
        history = getattr(provider_client, "message_history", None)
        if isinstance(history, list) and history and _history_item_has_task_done(history[-1]):
            history.pop()

    @override
    def llm_indicates_task_completed(self, llm_response: LLMResponse) -> bool:
        """Check if the LLM indicates that the task is completed."""
        if llm_response.tool_calls is None:
            return False
        return any(tool_call.name == "task_done" for tool_call in llm_response.tool_calls)

    @override
    def _is_task_completed(self, llm_response: LLMResponse) -> bool:
        """Enhanced task completion detection."""
        if self.must_patch == "true":
            model_patch = self.get_git_diff()
            patch = self.remove_patches_to_tests(model_patch)
            if not patch.strip():
                validation = self._empty_patch_validation_result()
                self._completion_validation_last_result = validation
                self._completion_validation_failed_task_done_count += 1
                self._record_completion_validation_audit(validation, rejected=True)
                self._completion_validation_message = None
                self._discard_last_task_done_from_history()
                return False

        validation = _strip_legacy_family_metadata(self._run_completion_validation())
        self._completion_validation_last_result = validation
        validation_audit_recorded = False
        if validation is not None and validation.get("failed_stage") == "completion_validation_spec":
            self._record_completion_validation_audit(validation, completed=True)
            self._completion_validation_message = None
            return True
        if validation is not None and not validation.get("passed"):
            self._completion_validation_failed_task_done_count += 1
            self._record_completion_validation_audit(validation, rejected=True)
            validation_audit_recorded = True
            self._completion_validation_message = self._format_completion_validation_message(validation)
            self._discard_last_task_done_from_history()
            return False

        if validation is not None and not validation_audit_recorded:
            self._record_completion_validation_audit(validation, completed=True)
        self._completion_validation_message = None
        return True

    @override
    def _finalize_after_step_limit(self) -> str | None:
        """Validate the current worktree before treating max_steps as failure."""
        if not self.completion_validation_path:
            return None

        if self.must_patch == "true":
            patch = self.remove_patches_to_tests(self.get_git_diff())
            if not patch.strip():
                validation = self._empty_patch_validation_result()
                self._completion_validation_last_result = validation
                self._record_completion_validation_audit(
                    validation,
                    rejected=True,
                    trigger="max_steps",
                )
                return None

        validation = _strip_legacy_family_metadata(self._run_completion_validation())
        self._completion_validation_last_result = validation
        if not isinstance(validation, dict):
            return None
        if validation.get("failed_stage") == "completion_validation_spec":
            self._record_completion_validation_audit(
                validation,
                rejected=True,
                trigger="max_steps",
            )
            return None
        if not validation.get("passed"):
            self._record_completion_validation_audit(
                validation,
                rejected=True,
                trigger="max_steps",
            )
            return None

        self._record_completion_validation_audit(
            validation,
            completed=True,
            trigger="max_steps",
        )
        self._completion_validation_message = None
        return "Task completed after final validation at the maximum step limit."

    @override
    def task_incomplete_message(self) -> str:
        """Return a message indicating that the task is incomplete."""
        completion_validation_message = getattr(self, "_completion_validation_message", None)
        if completion_validation_message:
            return completion_validation_message
        return EMPTY_PATCH_RETRY_PROMPT
