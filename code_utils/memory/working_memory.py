"""Deterministic, current-task Working Memory for AutoDialect.

The file rendered by this module is deliberately a view over existing facts:
pipeline state, parser gates, validation results, worktree changes, and raw
trajectory pointers.  It does not ask an LLM to author notes or recommendations.

The caller must construct this manager only when the existing global AutoDialect
``enable_memory`` flag is enabled.  ``create_if_enabled`` makes that contract
explicit and returns ``None`` while memory is disabled.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from code_utils.common_tool import model_facing_diagnostics
from code_utils.sql_ut.sql_ut_splitter import split_is_reliable_rule_evidence


WORKING_MEMORY_RELATIVE_PATH = os.path.join(".autodialect", "WORKING_MEMORY.md")
WORKING_MEMORY_PROMPT = """

[AutoDialect Working Memory]
The runner-maintained current-task memory is at `.autodialect/WORKING_MEMORY.md`.
Read it before acting. Treat it as a stage-start snapshot: it is refreshed by
the runner between AutoDialect stages/groups/repairs, not after every tool call inside
this stage. Use its evidence pointers to inspect raw trajectory steps when
needed. Do not edit this generated file.

Useful checks:
- `sed -n '1,220p' .autodialect/WORKING_MEMORY.md`
- If it points to `trajectory:<stage>#step=N`, search that stage in the
  trajectory manifest and inspect the matching raw trajectory step.
""".rstrip()

_FUNCTION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ERROR_TYPE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b")
_ABSOLUTE_PYTHON_PATH_RE = re.compile(r"(/[A-Za-z0-9_.\-/]+\.py)")
_STAGE_INDEX_RE = re.compile(r"(?:^|_)(\d+)$")
_RAW_TOOL_ARG_RE_TEMPLATE = r'"{key}"\s*:\s*"([^"]+)"'
_MAX_FAILED_OUTCOMES = 8
_WRITE_EDIT_COMMANDS = {"create", "insert", "str_replace"}


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", [], ()):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _stable_text_list(value: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in _as_list(value):
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _inline_code(value: Any) -> str:
    return f"`{str(value).replace('`', '')}`"


def _inline_list(values: Any) -> str:
    items = _stable_text_list(values)
    return "[" + ", ".join(_inline_code(item) for item in items) + "]"


def _relative_project_path(path: str, project_path: str) -> str:
    if not path:
        return ""
    absolute = os.path.abspath(path)
    project = os.path.abspath(project_path)
    try:
        if os.path.commonpath([absolute, project]) == project:
            return os.path.relpath(absolute, project)
    except ValueError:
        pass
    return absolute


def _first_function(sql: Any) -> str:
    match = _FUNCTION_RE.search(str(sql or ""))
    return match.group(1).upper() if match else ""


def _representative_validation_item(validation: dict[str, Any]) -> dict[str, Any]:
    split_results = validation.get("split_results") or []
    for item in split_results:
        if isinstance(item, dict) and item.get("passed") is False:
            return item
    for item in split_results:
        if isinstance(item, dict):
            return item
    repair_target = validation.get("repair_target")
    if isinstance(repair_target, dict):
        for key in ("failed_splits", "failed_variants"):
            for item in repair_target.get(key) or []:
                if isinstance(item, dict):
                    return item
    if any(
        validation.get(key) not in (None, "", [], {})
        for key in ("actual_sql", "expected_sql", "diff_summary", "error")
    ):
        return validation
    return {}


def _selected_diff(item: dict[str, Any]) -> dict[str, Any]:
    selected = item.get("selected_target_diff")
    return selected if isinstance(selected, dict) else {}


def _validation_diff(item: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    selected = _selected_diff(item)
    if selected:
        return (
            "Selected-expression diff",
            _stable_text_list(selected.get("actual_only")),
            _stable_text_list(selected.get("expected_only")),
        )
    diff_summary = _stable_text_list(item.get("diff_summary"))
    if diff_summary:
        return "Split diff", diff_summary, []
    return "", [], []


def _validation_actual_expected(item: dict[str, Any]) -> tuple[str, str]:
    selected = _selected_diff(item)
    actual = selected.get("actual_node_sql") if selected else None
    expected = selected.get("expected_node_sql") if selected else None
    return (
        str(actual or item.get("actual_sql") or "").strip(),
        str(expected or item.get("expected_sql") or "").strip(),
    )


def _should_render_actual_expected(item: dict[str, Any]) -> bool:
    """Show full actual/expected only when the compact diff is insufficient.

    The selected-expression diff is usually the most useful and shortest signal
    for repair.  Reprinting full SQL beside that diff makes WORKING_MEMORY.md
    longer without adding much.  Keep actual/expected for cases that only have
    raw SQL evidence or where both sides are short enough to scan quickly.
    """
    diff_label, actual_only, expected_only = _validation_diff(item)
    actual, expected = _validation_actual_expected(item)
    if not actual and not expected:
        return False
    if not diff_label:
        return True
    if actual_only or expected_only:
        return False
    return len(actual) + len(expected) <= 140


def _compact_failure_text(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _safe_step_number(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tool_argument(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if value not in (None, ""):
        return str(value)
    raw_arguments = str(arguments.get("_raw_arguments") or "")
    if not raw_arguments:
        return ""
    pattern = _RAW_TOOL_ARG_RE_TEMPLATE.format(key=re.escape(key))
    match = re.search(pattern, raw_arguments)
    return match.group(1) if match else ""


def _is_write_edit_call(tool_name: str, arguments: dict[str, Any]) -> bool:
    if tool_name == "json_edit_tool":
        return True
    if tool_name != "str_replace_based_edit_tool":
        return False
    command = _tool_argument(arguments, "command").strip()
    return command in _WRITE_EDIT_COMMANDS


@dataclass
class WorkingMemoryManager:
    """Maintain one generated ``WORKING_MEMORY.md`` for the current task."""

    project_path: str
    rule_name: str
    src_dialect: str
    tgt_dialect: str
    trajectory_manifest_path: str
    base_ref: str | None = None
    current_stage: str = "case_initialization"
    focus_group_ids: list[str] = field(default_factory=list)
    parser_gate: dict[str, Any] = field(default_factory=dict)
    parser_gate_stage: str = ""
    target_parser_gate: dict[str, Any] = field(default_factory=dict)
    target_parser_gate_stage: str = ""
    split_manifest: dict[str, Any] = field(default_factory=dict)
    group_execution: dict[str, Any] = field(default_factory=dict)
    latest_validation: dict[str, Any] = field(default_factory=dict)
    latest_validation_stage: str = ""
    failed_outcomes: list[dict[str, Any]] = field(default_factory=list)
    trajectory_stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create_if_enabled(
        cls,
        *,
        enabled: bool,
        project_path: str,
        rule_name: str,
        src_dialect: str,
        tgt_dialect: str,
        trajectory_manifest_path: str,
        base_ref: str | None = None,
    ) -> WorkingMemoryManager | None:
        """Return an initialized manager only for the global memory-on path."""
        if not enabled:
            return None
        manager = cls(
            project_path=os.path.abspath(project_path),
            rule_name=rule_name,
            src_dialect=src_dialect,
            tgt_dialect=tgt_dialect,
            trajectory_manifest_path=os.path.abspath(trajectory_manifest_path),
            base_ref=base_ref,
        )
        try:
            manager._initialize_storage()
            manager.render()
        except Exception:
            # This auxiliary layer must never make a memory-enabled run fail.
            pass
        return manager

    @property
    def path(self) -> str:
        return os.path.join(self.project_path, WORKING_MEMORY_RELATIVE_PATH)

    def attach_to_prompt(self, prompt: str) -> str:
        """Expose the generated file without copying its content into the prompt."""
        if not os.path.isfile(self.path):
            return prompt
        return f"{prompt.rstrip()}\n{WORKING_MEMORY_PROMPT}\n"

    def record_parser_gate(
        self,
        *,
        role: str,
        result: dict[str, Any],
        stage: str,
    ) -> None:
        if role == "target":
            self.target_parser_gate = dict(result or {})
            self.target_parser_gate_stage = stage
        else:
            self.parser_gate = dict(result or {})
            self.parser_gate_stage = stage
        self.current_stage = stage
        if result.get("status") != "parse_ok":
            check = "target_parser_gate" if role == "target" else "source_parser_gate"
            self._upsert_failed_outcome(
                {
                    "kind": "runner",
                    "check": check,
                    "error_type": (
                        result.get("error_type") or result.get("status") or "parse_failed"
                    ),
                    "failure_signature": result.get("status") or "parse_failed",
                    "evidence": [f"pipeline:{stage}"],
                    "signature": (
                        f"gate|{check}|{result.get('status')}|{result.get('error_type')}"
                    ),
                }
            )
        self.render()

    def record_split_manifest(self, manifest: dict[str, Any]) -> None:
        self.split_manifest = manifest or {}
        self.current_stage = "ut_split"
        self.render()

    def record_group_plan(
        self,
        group_execution: dict[str, Any],
        *,
        stage: str = "group_queue",
    ) -> None:
        self.group_execution = group_execution or {}
        self.current_stage = stage
        self.render()

    def refresh_group_execution(self, group_execution: dict[str, Any]) -> None:
        """Refresh group counters/statuses without changing the current event."""
        self.group_execution = group_execution or {}
        self.render()

    def set_stage(
        self,
        stage: str,
        *,
        focus_group_ids: Any = None,
    ) -> None:
        self.current_stage = stage
        if focus_group_ids is not None:
            self.focus_group_ids = _stable_text_list(focus_group_ids)
        self.render()

    def record_validation(
        self,
        *,
        stage: str,
        validation: dict[str, Any],
        focus_group_ids: Any = None,
        trajectory_stage: str | None = None,
    ) -> None:
        self.current_stage = stage
        self.latest_validation = model_facing_diagnostics(validation or {})
        self.latest_validation_stage = stage
        passed = self.latest_validation.get("passed")
        if focus_group_ids is not None:
            self.focus_group_ids = _stable_text_list(focus_group_ids)
        if passed is False:
            self._record_failed_validation(stage, trajectory_stage=trajectory_stage)
        self.render()

    def record_stage_trajectory(self, stage: str, trajectory_file: str) -> None:
        """Index deterministic edit and tool-failure facts from one raw trajectory."""
        record: dict[str, Any] = {
            "groups": list(self.focus_group_ids),
            "edit_steps": [],
            "edited_files": [],
        }
        try:
            with open(trajectory_file, "r", encoding="utf-8") as rf:
                payload = json.load(rf)
        except (OSError, ValueError):
            # Do not advertise a trajectory stage as indexed when the source
            # could not be loaded. The raw case manifest remains authoritative.
            return

        edited_files: list[str] = []
        edit_steps: list[int] = []
        for step in payload.get("agent_steps") or []:
            if not isinstance(step, dict):
                continue
            step_number = _safe_step_number(step.get("step_number"))
            calls = {
                str(call.get("call_id") or ""): call
                for call in step.get("tool_calls") or []
                if isinstance(call, dict)
            }
            results = {
                str(result.get("call_id") or ""): result
                for result in step.get("tool_results") or []
                if isinstance(result, dict)
            }
            for call_id, call in calls.items():
                name = str(call.get("name") or "")
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                path = _tool_argument(arguments, "path")
                result = results.get(call_id, {})
                if (
                    _is_write_edit_call(name, arguments)
                    and path
                    and result.get("success") is True
                ):
                    relative_path = _relative_project_path(path, self.project_path)
                    # Temporary probes and any external paths are not production
                    # worktree changes. Keep them available in the raw trajectory,
                    # but do not advertise them as Changed evidence.
                    if not os.path.isabs(relative_path):
                        edited_files.append(relative_path)
                        if step_number:
                            edit_steps.append(step_number)

            for tool_result in results.values():
                if tool_result.get("success") is not False:
                    continue
                call = calls.get(str(tool_result.get("call_id") or ""), {})
                self._record_tool_failure(
                    stage=stage,
                    step_number=step_number,
                    tool_name=str(call.get("name") or "tool"),
                    error=str(tool_result.get("error") or tool_result.get("result") or ""),
                )

        record["edit_steps"] = sorted(set(edit_steps))
        record["edited_files"] = list(dict.fromkeys(edited_files))
        self.trajectory_stages[stage] = record
        self.render()

    def finalize(self) -> None:
        self.current_stage = "case_complete"
        self.render()

    def render(self) -> None:
        """Atomically refresh the Markdown view; failures never break the run."""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            content = self._render_markdown()
            temporary = f"{self.path}.tmp"
            with open(temporary, "w", encoding="utf-8") as wf:
                wf.write(content)
            os.replace(temporary, self.path)
        except Exception:  # best-effort auxiliary memory must not break generation
            pass

    def _initialize_storage(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            git_path = subprocess.check_output(
                ["git", "rev-parse", "--git-path", "info/exclude"],
                cwd=self.project_path,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if not os.path.isabs(git_path):
                git_path = os.path.join(self.project_path, git_path)
            os.makedirs(os.path.dirname(git_path), exist_ok=True)
            existing = ""
            if os.path.isfile(git_path):
                with open(git_path, "r", encoding="utf-8") as rf:
                    existing = rf.read()
            if ".autodialect/" not in {line.strip() for line in existing.splitlines()}:
                with open(git_path, "a", encoding="utf-8") as wf:
                    if existing and not existing.endswith("\n"):
                        wf.write("\n")
                    wf.write(".autodialect/\n")
        except (OSError, subprocess.CalledProcessError):
            # Git exclusion is a cleanliness optimization; rendering still works without it.
            pass

    def _changed_files(self) -> list[str]:
        commands = [
            ["git", "diff", "--name-only"],
            ["git", "diff", "--cached", "--name-only"],
            ["git", "ls-files", "--others", "--exclude-standard"],
        ]
        if self.base_ref:
            commands.append(["git", "diff", "--name-only", f"{self.base_ref}..HEAD"])
        files: list[str] = []
        for command in commands:
            try:
                output = subprocess.check_output(
                    command,
                    cwd=self.project_path,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            for line in output.splitlines():
                path = line.strip()
                if path and not path.startswith(".autodialect/") and path not in files:
                    files.append(path)
        return files

    def _focused_splits(self) -> list[dict[str, Any]]:
        splits = [
            item
            for item in self.split_manifest.get("splits") or []
            if isinstance(item, dict) and split_is_reliable_rule_evidence(item)
        ]
        if self.focus_group_ids:
            focused = [item for item in splits if item.get("rule_group_id") in self.focus_group_ids]
            return focused
        return splits[:1]

    def _expected_transitions(self) -> list[str]:
        transitions: list[str] = []
        for split in self._focused_splits():
            selected = split.get("selected_target_diff")
            if not isinstance(selected, dict):
                selected = {}
            source_type = str(
                selected.get("source_node_type")
                or split.get("diff_node_type")
                or split.get("cut_node_type")
                or ""
            )
            expected_type = str(
                selected.get("expected_node_type")
                or split.get("target_diff_node_type")
                or ""
            )
            source_sql = selected.get("source_node_sql") or split.get("source_expr_sql")
            expected_sql = selected.get("expected_node_sql") or split.get("expected_expr_sql")
            source_function = _first_function(source_sql)
            expected_function = _first_function(expected_sql)
            source_label = source_type
            expected_label = expected_type
            if source_function:
                source_label = (
                    f"{source_label} ({source_function})" if source_label else source_function
                )
            if expected_function:
                expected_label = (
                    f"{expected_label} ({expected_function})"
                    if expected_label
                    else expected_function
                )
            if source_label and expected_label:
                transition = f"{source_label} -> {expected_label}"
                if transition not in transitions:
                    transitions.append(transition)
        return transitions

    def _focus_split_ids(self) -> list[str]:
        # ``_stable_text_list`` intentionally treats unknown scalar-like values as
        # one item.  Materialize the comprehension here so a generator is not
        # rendered as ``<generator object ...>`` in WORKING_MEMORY.md.
        return _stable_text_list(
            [split.get("split_id") for split in self._focused_splits()]
        )

    def _record_failed_validation(self, stage: str, *, trajectory_stage: str | None) -> None:
        item = _representative_validation_item(self.latest_validation)
        diff_label, actual_only, expected_only = _validation_diff(item)
        actual, expected = _validation_actual_expected(item)
        group_ids = _stable_text_list(
            self.latest_validation.get("focus_rule_group_ids")
            or self.latest_validation.get("focus_rule_group_id")
            or self.latest_validation.get("pending_group_ids")
            or item.get("rule_group_id")
            or self.focus_group_ids
        )
        split_ids = _stable_text_list(item.get("split_id"))
        signature_parts = [
            stage,
            ",".join(group_ids),
            ",".join(split_ids),
            ",".join(actual_only),
            ",".join(expected_only),
        ]
        signature = "|".join(signature_parts)
        evidence = []
        trajectory = self.trajectory_stages.get(trajectory_stage or "")
        if trajectory_stage and trajectory:
            edit_steps = trajectory.get("edit_steps") or []
            if edit_steps:
                rendered_steps = ",".join(str(step) for step in edit_steps)
                evidence.append(f"trajectory:{trajectory_stage}#step={rendered_steps}")
        evidence.append(f"pipeline:{stage}")
        repair_target = self.latest_validation.get("repair_target")
        repair_target = repair_target if isinstance(repair_target, dict) else {}
        validation_scope = self.latest_validation.get("validation_scope")
        check = (
            "group_execution_validation"
            if validation_scope == "group_queue"
            else "focused_group_validation"
            if validation_scope == "focused_group" or group_ids
            else "full_manifest_validation"
        )
        outcome = {
            "kind": "runner",
            "check": check,
            "scope_groups": group_ids,
            "scope_splits": split_ids,
            "diff_label": diff_label,
            "actual_only": actual_only,
            "expected_only": expected_only,
            "actual_sql": actual,
            "expected_sql": expected,
            "why_failed": self._failure_summary(item, repair_target),
            "failure_signature": (
                repair_target.get("reason")
                or self.latest_validation.get("failed_stage")
                or "validation_failed"
            ),
            "changed_files": list((trajectory or {}).get("edited_files") or []),
            "evidence": evidence,
            "signature": signature,
        }
        self._upsert_failed_outcome(outcome)

    def _failure_summary(
        self,
        item: dict[str, Any],
        repair_target: dict[str, Any] | None = None,
    ) -> str:
        repair_target = repair_target if isinstance(repair_target, dict) else {}
        diff_label, actual_only, expected_only = _validation_diff(item)
        if diff_label == "Selected-expression diff":
            parts = []
            if actual_only:
                parts.append("remove/replace actual-only " + ", ".join(actual_only[:4]))
            if expected_only:
                parts.append("emit expected-only " + ", ".join(expected_only[:4]))
            if parts:
                return "; ".join(parts)
        if diff_label == "Split diff" and actual_only:
            return "SQL mismatch tokens: " + ", ".join(actual_only[:6])
        actual, expected = _validation_actual_expected(item)
        if actual or expected:
            return (
                f"actual `{_compact_failure_text(actual, 90)}` does not match "
                f"expected `{_compact_failure_text(expected, 90)}`"
            )
        return str(
            repair_target.get("reason")
            or self.latest_validation.get("failed_stage")
            or "validation_failed"
        )

    def _remaining_failure_summary(self) -> str:
        if not self.latest_validation or self.latest_validation.get("passed") is not False:
            return ""
        item = _representative_validation_item(self.latest_validation)
        repair_target = self.latest_validation.get("repair_target")
        repair_target = repair_target if isinstance(repair_target, dict) else {}
        return self._failure_summary(item, repair_target)

    def _record_tool_failure(
        self,
        *,
        stage: str,
        step_number: int,
        tool_name: str,
        error: str,
    ) -> None:
        error_type_match = _ERROR_TYPE_RE.search(error)
        error_type = error_type_match.group(1) if error_type_match else "ToolError"
        signature = error_type
        if "outside the current project root" in error.lower():
            signature = "write_path_outside_current_worktree"
        else:
            for path in _ABSOLUTE_PYTHON_PATH_RE.findall(error):
                absolute = os.path.abspath(path)
                try:
                    outside = (
                        os.path.commonpath([absolute, self.project_path])
                        != self.project_path
                    )
                except ValueError:
                    outside = True
                if outside and "/sqlglot/" in absolute:
                    signature = "module_path_outside_current_worktree"
                    break
        outcome = {
            "kind": "tool",
            "check": tool_name or "tool_call",
            "scope_stage": stage,
            "error_type": error_type,
            "failure_signature": signature,
            "evidence": [f"trajectory:{stage}#step={step_number}"],
            "signature": f"tool|{stage}|{step_number}|{tool_name}|{signature}",
        }
        self._upsert_failed_outcome(outcome)

    def _upsert_failed_outcome(self, outcome: dict[str, Any]) -> None:
        signature = outcome.get("signature")
        self.failed_outcomes = [
            item for item in self.failed_outcomes if item.get("signature") != signature
        ]
        self.failed_outcomes.append(outcome)
        self.failed_outcomes = self.failed_outcomes[-_MAX_FAILED_OUTCOMES:]

    def _render_markdown(self) -> str:
        lines = [
            "# Working Memory",
            "",
            "<!-- AUTODIALECT_AUTO_START -->",
            "",
            "## Current State",
            "",
            "### Scope",
            "",
            (
                f"- Task: {_inline_code(self.rule_name)} "
                f"({_inline_code(self.src_dialect)} -> {_inline_code(self.tgt_dialect)})"
            ),
            f"- Stage: {_inline_code(self.current_stage)}",
        ]
        focus_parts = [*self.focus_group_ids, *self._focus_split_ids()]
        if focus_parts:
            lines.append("- Focus: " + " / ".join(_inline_code(item) for item in focus_parts))

        transitions = self._expected_transitions()
        if transitions:
            lines.extend(["", "### Expected Transition", ""])
            lines.extend(f"- {_inline_code(transition)}" for transition in transitions)

        lines.extend(["", "### Latest Validation", ""])
        if self.latest_validation:
            passed = self.latest_validation.get("passed")
            result = "passed" if passed is True else "failed" if passed is False else "unknown"
            validation_stage = self.latest_validation_stage or self.current_stage
            lines.append(f"- Result: {_inline_code(result)} at {_inline_code(validation_stage)}")
            pending_group_ids = _stable_text_list(self.latest_validation.get("pending_group_ids"))
            if pending_group_ids:
                lines.append(f"- Pending groups: {_inline_list(pending_group_ids)}")
            repair_target = self.latest_validation.get("repair_target")
            repair_target = repair_target if isinstance(repair_target, dict) else {}
            if repair_target.get("reason"):
                lines.append(f"- Repair reason: {_inline_code(repair_target['reason'])}")
            item = _representative_validation_item(self.latest_validation)
            remaining_failure = self._remaining_failure_summary()
            if remaining_failure:
                lines.append(f"- Remaining failure: {_inline_code(remaining_failure)}")
            if _should_render_actual_expected(item):
                actual, expected = _validation_actual_expected(item)
                if actual:
                    lines.extend(["- Actual:", f"  {_inline_code(actual)}"])
                if expected:
                    lines.extend(["- Expected:", f"  {_inline_code(expected)}"])
            diff_label, actual_only, expected_only = _validation_diff(item)
            if diff_label:
                lines.append(f"- {diff_label}:")
                if diff_label == "Split diff":
                    lines.append(f"  - items: {_inline_list(actual_only)}")
                else:
                    lines.extend(
                        [
                            f"  - actual_only: {_inline_list(actual_only)}",
                            f"  - expected_only: {_inline_list(expected_only)}",
                        ]
                    )
        else:
            lines.append(f"- Result: {_inline_code('not_run')}")

        changed_files = self._changed_files()
        lines.extend(["", "### Worktree", ""])
        if changed_files:
            lines.append("- Changed:")
            lines.extend(f"  - {_inline_code(path)}" for path in changed_files)
        else:
            lines.append("- Changed: none")

        lines.extend(["", "## Verified Outcomes", "", "### Passed", ""])
        passed_lines = self._render_passed_outcomes()
        lines.extend(passed_lines or ["- None recorded."])

        lines.extend(["", "### Failed", ""])
        failed_lines = self._render_failed_outcomes()
        lines.extend(failed_lines or ["- None recorded."])

        lines.extend(["", "## Task Progress", ""])
        lines.append(f"- Trajectory manifest: {_inline_code(self.trajectory_manifest_path)}")
        lines.extend(
            [
                "",
                "| Rule group / split | State | Attempts | Trajectory stages |",
                "|---|---|---:|---|",
            ]
        )
        rows = self._progress_rows()
        lines.extend(rows or ["| none | `pending` | 0 | none |"])
        lines.extend(["", "<!-- AUTODIALECT_AUTO_END -->", ""])
        return "\n".join(lines)

    def _render_passed_outcomes(self) -> list[str]:
        lines: list[str] = []
        for label, gate, gate_stage in (
            ("source_parser_gate", self.parser_gate, self.parser_gate_stage),
            ("target_parser_gate", self.target_parser_gate, self.target_parser_gate_stage),
        ):
            if gate.get("status") != "parse_ok":
                continue
            nodes = _stable_text_list(gate.get("node_types"))
            transition_types: list[str] = []
            for split in self._focused_splits():
                selected = split.get("selected_target_diff")
                selected = selected if isinstance(selected, dict) else {}
                key = "expected_node_type" if label.startswith("target") else "source_node_type"
                value = str(selected.get(key) or "")
                if value:
                    transition_types.append(value)
            focus_node = next(
                (node for node in transition_types if node in nodes),
                gate.get("root") or "parse_ok",
            )
            lines.extend(
                [
                    f"- [runner] {_inline_code(label)}",
                    f"  - Observed node: {_inline_code(focus_node)}",
                    f"  - Evidence: {_inline_code(f'pipeline:{gate_stage}')}",
                ]
            )

        if self.latest_validation:
            item = _representative_validation_item(self.latest_validation)
            selected = _selected_diff(item)
            actual_type = str(selected.get("actual_node_type") or "")
            expected_type = str(selected.get("expected_node_type") or "")
            if actual_type and actual_type == expected_type:
                scope = [*self.focus_group_ids, *_stable_text_list(item.get("split_id"))]
                lines.extend(
                    [
                        f"- [derived] {_inline_code('focused_node_type_match')}",
                        "  - Scope: " + " / ".join(_inline_code(value) for value in scope),
                        f"  - Actual node type: {_inline_code(actual_type)}",
                        f"  - Expected node type: {_inline_code(expected_type)}",
                        (
                            "  - Evidence: "
                            + _inline_code(
                                f"pipeline:{self.latest_validation_stage}.selected_target_diff"
                            )
                        ),
                    ]
                )
        return lines

    def _render_failed_outcomes(self) -> list[str]:
        lines: list[str] = []
        for outcome in self.failed_outcomes:
            lines.append(
                f"- [{outcome.get('kind')}] {_inline_code(outcome.get('check') or 'failure')}"
            )
            scope = [
                *_stable_text_list(outcome.get("scope_groups")),
                *_stable_text_list(outcome.get("scope_splits")),
            ]
            if outcome.get("scope_stage"):
                scope.append(str(outcome["scope_stage"]))
            if scope:
                lines.append("  - Scope: " + " / ".join(_inline_code(value) for value in scope))
            if outcome.get("diff_label"):
                lines.append(f"  - {outcome['diff_label']}:")
                if outcome["diff_label"] == "Split diff":
                    lines.append(f"    - items: {_inline_list(outcome.get('actual_only'))}")
                else:
                    lines.extend(
                        [
                            f"    - actual_only: {_inline_list(outcome.get('actual_only'))}",
                            f"    - expected_only: {_inline_list(outcome.get('expected_only'))}",
                        ]
                    )
            if outcome.get("error_type"):
                lines.append(f"  - Error: {_inline_code(outcome['error_type'])}")
            if outcome.get("failure_signature"):
                lines.append(
                    f"  - Failure signature: {_inline_code(outcome['failure_signature'])}"
                )
            if outcome.get("why_failed"):
                lines.append(f"  - Why failed: {_inline_code(outcome['why_failed'])}")
            evidence = _stable_text_list(outcome.get("evidence"))
            if len(evidence) == 1:
                lines.append(f"  - Evidence: {_inline_code(evidence[0])}")
            elif evidence:
                lines.append("  - Evidence:")
                lines.extend(f"    - {_inline_code(item)}" for item in evidence)
        return lines

    def _progress_rows(self) -> list[str]:
        hard_split_ids_by_group: dict[str, list[str]] = {}
        for split in self.split_manifest.get("splits") or []:
            if not isinstance(split, dict) or not split_is_reliable_rule_evidence(split):
                continue
            group_id = str(split.get("rule_group_id") or "")
            split_id = str(split.get("split_id") or "")
            if group_id and split_id:
                hard_split_ids_by_group.setdefault(group_id, []).append(split_id)
        groups = [
            item
            for item in self.split_manifest.get("groups") or []
            if isinstance(item, dict)
            and str(item.get("rule_group_id") or "") in hard_split_ids_by_group
        ]
        statuses = self.group_execution.get("group_statuses") or {}
        rows: list[str] = []
        for group in groups:
            group_id = str(group.get("rule_group_id") or "")
            if not group_id:
                continue
            split_ids = hard_split_ids_by_group.get(group_id, [])
            focus_label = " / ".join(_inline_code(value) for value in [group_id, *split_ids])
            status = statuses.get(group_id) if isinstance(statuses, dict) else {}
            status = status if isinstance(status, dict) else {}
            state = str(status.get("status") or "pending")
            attempts = int(status.get("attempts") or 0)
            if group_id in self.focus_group_ids and self.current_stage.startswith("rule_group"):
                if "repair" in self.current_stage:
                    state = "repair_active"
                elif state == "pending":
                    state = "active"
                if _STAGE_INDEX_RE.search(self.current_stage):
                    attempts += 1
            trajectory_stages = [
                stage
                for stage, item in self.trajectory_stages.items()
                if group_id in _stable_text_list(item.get("groups"))
            ]
            stages_text = ", ".join(_inline_code(stage) for stage in trajectory_stages) or "none"
            rows.append(
                f"| {focus_label} | {_inline_code(state)} | {attempts} | {stages_text} |"
            )
        return rows
