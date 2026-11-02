#!/usr/bin/env python3
"""Run selected AutoDialect cases and persist experiment records.

Case selection: build_parser()/resolve_experiment_run_config()/main() choose
the experiment and case_indices_file/limit, load_online_cases() loads
cases, and run_gen() iterates selected cases while preserving original result keys.

AUTODIALECT patch pipeline summary for each case:
1. Parser gates: _run_autodialect_case_pipeline() calls run_parser_cycle() for source
   and target parser validation; parser repair uses run_autodialect_stage().
2. UT split: _run_autodialect_case_pipeline() calls build_split_manifest() and records
   split_manifest_stage_record().
3. Queue planning: prepare_group_execution_plan() creates focused group batches.
4. Focused patching: run_focused_group_batch() calls run_outer_validation_stage(),
   run_autodialect_stage(), reuse_completion_validation_or_run_outer_validation(),
   record_group_prevalidation(), record_group_attempt(), and _build_stage_handoff().
5. Final validation/result: run_outer_validation_stage() proves the full
   manifest, all_queued_groups_passed() guards group completion,
   build_pipeline_result() returns pipeline state, and run_gen() writes records.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent_runners import (
    ensure_agent_available,
    get_agent_cli_version,
    prepare_agent_runtime,
    run_agent,
)
from code_utils.execution_runtime import (
    _build_stage_handoff,
    _record_stage_memory_audit,
    run_parser_gate,
)
from code_utils.common_tool import (
    compact_top_level_fields,
    prepare_case_worktree,
    get_git_diff,
    get_git_head,
)
from code_utils.completion_validation_runtime import (
    _completion_validation_result_from_audit,
    _normalize_group_ids,
    _source_parser_failure_validation,
    _target_parser_failure_validation,
    _validation_stage_record,
    _write_completion_validation_spec,
    _write_parser_completion_validation_spec,
)
from code_utils.constants import (
    AUTODIALECT_MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT,
    AUTODIALECT_AGENT,
    AUTODIALECT_CONFIG_FILE,
)
from code_utils.experiment_utils import (
    DEFAULT_RUN_CONFIG,
    EXPERIMENT_RUN_CONFIGS,
    CaseTrajectoryCapture,
    ExperimentSpec,
    ValidationError,
    _case_metadata,
    _load_case_indices_file,
    _should_skip_existing_record,
    _stage_autodialect_args,
    add_experiment_selection_args,
    build_case_info,
    build_default_run_configs,
    build_autodialect_extra_args,
    case_expected_sql,
    case_source_sql,
    case_src_dialect,
    case_tgt_dialect,
    experiment_spec_from_run_config,
    load_online_cases,
    load_result_data,
    namespace_from_run_config,
    resolve_experiment_run_config,
    save_json,
    scrub_result_deprecated_success_fields,
    take_first_n_items,
    validate_experiment_spec,
)
from code_utils.memory.working_memory import WorkingMemoryManager
from code_utils.model_runtime import (
    ModelRuntime,
    resolve_agent_memory_enabled,
    resolve_model_runtime,
)
from code_utils.prompt_utils import build_prompt
from code_utils.sql_ut.group_execution import (
    all_queued_groups_passed,
    describe_group_batch,
    focus_validation_on_group,
    group_execution_pending_validation,
    prepare_group_execution_plan,
    prepare_group_repair_plan,
    record_group_attempt,
    record_group_prevalidation,
    split_manifest_stage_record,
)
from code_utils.sql_ut.split_validator import validate_manifest
from code_utils.sql_ut.sql_ut_splitter import build_split_manifest


@dataclass
class _AutoDialectCasePipelineResult:
    task_prompt: str
    response: str
    execution_time: float
    is_success: bool
    memory_audit: dict[str, Any]
    parser_gate_result: dict[str, Any]
    target_parser_gate_result: dict[str, Any]
    split_manifest: dict[str, Any]
    autodialect_validation: dict[str, Any]
    group_execution: dict[str, Any]
    stage_handoff: dict[str, Any] | None
    pipeline_stages: list[dict[str, Any]]


def _run_autodialect_case_pipeline(
    spec: ExperimentSpec,
    case: dict[str, Any],
    rule_name: str,
    project_path: str,
    info: dict[str, Any],
    runner_args: dict[str, Any],
    include_memory_query: bool,
    base_ref: str | None,
    trajectory_capture: CaseTrajectoryCapture,
    memory_enabled: bool = False,
) -> _AutoDialectCasePipelineResult:
    """Run one selected case through the AUTODIALECT AutoDialect pipeline."""
    src_dialect = case_src_dialect(spec, case)
    tgt_dialect = case_tgt_dialect(spec, case)
    source_sql = case_source_sql(spec, case)
    expected_sql = case_expected_sql(spec, case)
    task_prompt = ""
    response = ""
    execution_time = 0.0
    is_success = False
    memory_audit: dict[str, Any] = {}
    parser_gate_result: dict[str, Any] = {}
    target_parser_gate_result: dict[str, Any] = {}
    split_manifest: dict[str, Any] = {}
    autodialect_validation: dict[str, Any] = {}
    group_execution: dict[str, Any] = {}
    group_queue: list[dict[str, Any]] = []
    stage_handoff: dict[str, Any] | None = None
    pipeline_stages: list[dict[str, Any]] = []
    working_memory = WorkingMemoryManager.create_if_enabled(
        enabled=memory_enabled,
        project_path=project_path,
        rule_name=rule_name,
        src_dialect=src_dialect,
        tgt_dialect=tgt_dialect,
        trajectory_manifest_path=os.path.join(
            spec.result_folder,
            "trajectories",
            rule_name,
            "trajectory_manifest.json",
        ),
        base_ref=base_ref,
    )

    def with_working_memory(prompt: str) -> str:
        return working_memory.attach_to_prompt(prompt) if working_memory else prompt

    def append_pipeline_stage(record: dict[str, Any]) -> None:
        pipeline_stages.append(compact_top_level_fields(record))

    def build_pipeline_result() -> _AutoDialectCasePipelineResult:
        """Package current pipeline state."""
        if working_memory:
            working_memory.finalize()
        return _AutoDialectCasePipelineResult(
            task_prompt=task_prompt,
            response=response,
            execution_time=execution_time,
            is_success=is_success,
            memory_audit=memory_audit,
            parser_gate_result=parser_gate_result,
            target_parser_gate_result=target_parser_gate_result,
            split_manifest=split_manifest,
            autodialect_validation=autodialect_validation,
            group_execution=group_execution,
            stage_handoff=stage_handoff,
            pipeline_stages=pipeline_stages,
        )

    def run_autodialect_stage(
        stage: str,
        prompt: str,
        completion_validation_path: str | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Run one AUTODIALECT agent stage and record per-stage runtime data."""
        nonlocal execution_time, memory_audit
        stage_args = _stage_autodialect_args(runner_args, stage)
        if completion_validation_path:
            stage_args["completion_validation_path"] = completion_validation_path
        stage_started = time.monotonic()
        ok, stage_response, attempt_execution_time, attempt_memory_audit = run_agent(
            spec.agent_id,
            project_path,
            prompt,
            stage_args,
            is_summary=False,
        )
        stage_args["_duration_s"] = round(time.monotonic() - stage_started, 4)
        stage_args["_agent_execution_time_s"] = attempt_execution_time
        stage_args["_completion_validation_audit"] = (
            (attempt_memory_audit or {}).get("completion_validation")
            if isinstance(attempt_memory_audit, dict)
            else None
        )
        execution_time += attempt_execution_time
        memory_audit = _record_stage_memory_audit(
            memory_audit,
            stage=stage,
            attempt_memory_audit=attempt_memory_audit,
        )
        trajectory_copied = trajectory_capture.copy_autodialect_stage(
            stage_args["trajectory_file"],
            stage=stage,
        )
        if working_memory and trajectory_copied:
            # Make trajectory:<stage>#step=N pointers resolvable by the next
            # stage instead of waiting until the whole case has completed.
            trajectory_capture.write_manifest(pipeline_stages)
            copied_path = trajectory_capture.trajectory_manifest_entries[-1]["copied_path"]
            working_memory.record_stage_trajectory(stage, copied_path)
        return ok, stage_response, stage_args

    # 1. Parser gates: define and run parser validation/repair before rule generation.
    def run_parser_cycle(*, role: str) -> dict[str, Any]:
        """Validate and repair one parser role."""
        nonlocal autodialect_validation, is_success, response, task_prompt
        is_target = role == "target"
        sql = expected_sql if is_target else source_sql
        dialect = tgt_dialect if is_target else src_dialect
        gate_role = "target_expected" if is_target else "source"
        stage_prefix = "target_parser" if is_target else "parser"
        repair_prompt_kind = "target_parser_repair" if is_target else "parser_repair"
        repair_stage = "target_parser_repair" if is_target else "parser_repair"
        repair_label = "Target parser repair" if is_target else "Parser repair"
        failure_validation = (
            _target_parser_failure_validation if is_target else _source_parser_failure_validation
        )

        # Record each parser gate attempt in pipeline_stages.
        def append_gate_stage(stage: str, gate: dict[str, Any], started: float) -> None:
            """Record one parser gate attempt."""
            item = {
                "stage": stage,
                "parser_role": role,
                "status": gate.get("status"),
                "duration_s": round(time.monotonic() - started, 4),
            }
            if gate.get("error_type"):
                item["error_type"] = gate.get("error_type")
            append_pipeline_stage(item)
            if working_memory:
                working_memory.record_parser_gate(role=role, result=gate, stage=stage)

        stage_started = time.monotonic()
        gate_result = run_parser_gate(
            project_path=project_path,
            sql=sql,
            dialect=dialect,
            role=gate_role,
        )
        append_gate_stage(f"{stage_prefix}_gate", gate_result, stage_started)
        if gate_result.get("status") == "parse_ok":
            return gate_result

        prompt_kwargs = (
            {"target_parser_gate": gate_result}
            if is_target
            else {"parser_gate": gate_result}
        )
        parser_prompt = build_prompt(
            info,
            is_summary=False,
            prompt_kind=repair_prompt_kind,
            include_memory_query=include_memory_query,
            **prompt_kwargs,
        )
        parser_prompt = with_working_memory(parser_prompt)
        if not task_prompt:
            task_prompt = parser_prompt
        completion_validation_path = _write_parser_completion_validation_spec(
            spec=spec,
            rule_name=rule_name,
            stage=repair_stage,
            sql=sql,
            dialect=dialect,
            parser_role=gate_role,
        )
        ok, stage_response, stage_args = run_autodialect_stage(
            repair_stage,
            parser_prompt,
            completion_validation_path=completion_validation_path,
        )
        response = (
            f"{response}\n\n[{repair_label}]\n{stage_response}"
            if response else f"[{repair_label}]\n{stage_response}"
        )
        append_pipeline_stage(
            {
                "stage": repair_stage,
                "parser_role": role,
                "agent_success": ok,
                "duration_s": stage_args.get("_duration_s"),
                "agent_execution_time_s": stage_args.get("_agent_execution_time_s"),
                "completion_validation_audit": stage_args.get("_completion_validation_audit"),
            }
        )

        stage_started = time.monotonic()
        gate_result = run_parser_gate(
            project_path=project_path,
            sql=sql,
            dialect=dialect,
            role=gate_role,
        )
        append_gate_stage(f"{stage_prefix}_gate_after_repair", gate_result, stage_started)
        if gate_result.get("status") == "parse_ok":
            return gate_result

        autodialect_validation = failure_validation(gate_result)
        is_success = False
        response = (
            f"{response}\nParser repair could not finish {stage_prefix}: "
            f"{gate_result.get('error_type')}: {gate_result.get('error')}"
        )
        append_pipeline_stage(
            {
                "stage": f"{stage_prefix}_gate_failed",
                "parser_role": role,
                "status": gate_result.get("status"),
                "error_type": gate_result.get("error_type"),
                "repair_target": autodialect_validation.get("repair_target"),
            }
        )
        return gate_result

    # Run parser gates before rule generation.
    parser_gate_result = run_parser_cycle(role="source")
    if parser_gate_result.get("status") != "parse_ok":
        return build_pipeline_result()

    target_parser_gate_result = run_parser_cycle(role="target")
    if target_parser_gate_result.get("status") != "parse_ok":
        return build_pipeline_result()

    # 2. UT split: build the split manifest and record its pipeline stage.
    stage_started = time.monotonic()
    split_manifest = build_split_manifest(
        project_path=project_path,
        src_sql=source_sql,
        expected_sql=expected_sql,
        src_dialect=src_dialect,
        tgt_dialect=tgt_dialect,
        parent_case_id=rule_name,
    )
    append_pipeline_stage(
        split_manifest_stage_record(
            split_manifest,
            duration_s=round(time.monotonic() - stage_started, 4),
        )
    )
    if working_memory:
        working_memory.record_split_manifest(split_manifest)

    # 3. Queue planning: build focused group batches and initialize execution state.
    autodialect_validation = {}
    group_plan = prepare_group_execution_plan(split_manifest)
    group_queue = group_plan.group_queue
    group_execution = group_plan.group_execution
    group_batches = group_plan.group_batches
    append_pipeline_stage(group_plan.stage_record)
    if working_memory:
        working_memory.record_group_plan(group_execution)

    agent_calls = 0
    parent_repair_attempted = False

    # 4. Focused patching: define validation/patch helpers, patch queued groups,
    # then repair remaining failures.
    def run_outer_validation_stage(
        stage_name: str,
        focus_rule_group_id: Any | None = None,
        validation_scope: str | None = None,
        fallback_reason: str | None = None,
        trajectory_stage: str | None = None,
    ) -> dict[str, Any]:
        """Run validate_manifest() and append a validation stage record.

        Focused group checks and final validation both use this helper.
        """
        focus_group_ids = _normalize_group_ids(focus_rule_group_id)
        single_focus_group_id = focus_group_ids[0] if len(focus_group_ids) == 1 else None
        scope = validation_scope or ("focused_group" if focus_group_ids else "full_manifest")
        stage_started = time.monotonic()
        try:
            validation = validate_manifest(
                package_root=project_path,
                sys_path_backup=list(sys.path),
                manifest=split_manifest,
                parent_source_sql=source_sql,
                parent_expected_sql=expected_sql,
                src_dialect=src_dialect,
                tgt_dialect=tgt_dialect,
                validate_variants=True,
                validation_scope=scope,
                focus_rule_group_id=single_focus_group_id,
                focus_rule_group_ids=focus_group_ids,
            )
        except Exception as exc:
            validation = {
                "passed": False,
                "failed_stage": "validation_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "duration_s": round(time.monotonic() - stage_started, 4),
            }
        if scope == "full_manifest":
            validation = focus_validation_on_group(validation, single_focus_group_id)
        validation.setdefault("duration_s", round(time.monotonic() - stage_started, 4))
        append_pipeline_stage(
            _validation_stage_record(
                stage_name=stage_name,
                validation=validation,
                scope=scope,
                focus_rule_group_id=focus_group_ids,
                validation_source="outer_validate_manifest",
                fallback_reason=fallback_reason,
            )
        )
        if working_memory:
            working_memory.record_validation(
                stage=stage_name,
                validation=validation,
                focus_group_ids=focus_group_ids,
                trajectory_stage=trajectory_stage,
            )
        return validation

    def reuse_completion_validation_or_run_outer_validation(
        stage_name: str,
        stage_args: dict[str, Any],
        *,
        focus_rule_group_id: Any | None = None,
        validation_scope: str | None = None,
        trajectory_stage: str | None = None,
    ) -> dict[str, Any]:
        """Resolve validation from in-agent audit or outer validation.

        It prefers in-agent completion validation, then falls back to run_outer_validation_stage().
        """
        focus_group_ids = _normalize_group_ids(focus_rule_group_id)
        scope = validation_scope or ("focused_group" if focus_group_ids else "full_manifest")
        validation, fallback_reason = _completion_validation_result_from_audit(
            stage_args.get("_completion_validation_audit"),
            expected_scope=scope,
            expected_spec_path=stage_args.get("completion_validation_path"),
            expected_focus_rule_group_ids=focus_group_ids,
        )
        if validation is not None:
            append_pipeline_stage(
                _validation_stage_record(
                    stage_name=stage_name,
                    validation=validation,
                    scope=scope,
                    focus_rule_group_id=focus_group_ids,
                    validation_source="completion_validation_audit",
                )
            )
            if working_memory:
                working_memory.record_validation(
                    stage=stage_name,
                    validation=validation,
                    focus_group_ids=focus_group_ids,
                    trajectory_stage=trajectory_stage,
                )
            return validation
        return run_outer_validation_stage(
            stage_name,
            focus_rule_group_id=focus_rule_group_id,
            validation_scope=scope,
            fallback_reason=fallback_reason,
            trajectory_stage=trajectory_stage,
        )

    def should_run_parent_repair(validation: dict[str, Any]) -> bool:
        """Return whether final validation exposed a parent-only mismatch."""
        if not isinstance(validation, dict):
            return False
        repair_target = validation.get("repair_target")
        if not isinstance(repair_target, dict):
            repair_target = {}
        return (
            not parent_repair_attempted
            and parser_gate_result.get("status") == "parse_ok"
            and target_parser_gate_result.get("status") == "parse_ok"
            and validation.get("passed") is False
            and validation.get("failed_stage") == "parent"
            and repair_target.get("reason") == "parent_failed_after_split_groups"
            and all_queued_groups_passed(group_execution, group_queue)
        )

    def run_parent_repair(previous_stage: str) -> None:
        """Run the single parent-level fallback and refresh full validation."""
        nonlocal agent_calls, autodialect_validation, parent_repair_attempted
        nonlocal response, stage_handoff, task_prompt
        parent_repair_attempted = True
        repair_target = autodialect_validation.get("repair_target")
        stage_handoff = _build_stage_handoff(
            project_path=project_path,
            source_root=spec.src_dir,
            base_ref=base_ref,
            group_execution=group_execution,
            stage=previous_stage,
            validation=autodialect_validation,
            agent_success=False,
            agent_calls=agent_calls,
        )
        parent_repair_prompt = build_prompt(
            info,
            is_summary=False,
            parser_gate=parser_gate_result,
            target_parser_gate=target_parser_gate_result,
            split_manifest=split_manifest,
            prompt_kind="rule",
            max_diagnostic_hints_per_category=AUTODIALECT_MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT,
            stage_handoff=stage_handoff,
            include_memory_query=include_memory_query,
        )
        if working_memory:
            working_memory.set_stage("parent_repair")
        parent_repair_prompt = with_working_memory(parent_repair_prompt)
        if not task_prompt:
            task_prompt = parent_repair_prompt
        completion_validation_path = _write_completion_validation_spec(
            spec=spec,
            rule_name=rule_name,
            stage="parent_repair",
            split_manifest=split_manifest,
            case=case,
            validation_scope="full_manifest",
        )
        parent_agent_success, parent_response, stage_args = run_autodialect_stage(
            "parent_repair",
            parent_repair_prompt,
            completion_validation_path=completion_validation_path,
        )
        agent_calls += 1
        response = (
            f"{response}\n\n[Parent repair]\n{parent_response}"
            if response else parent_response
        )
        append_pipeline_stage(
            {
                "stage": "parent_repair_patch",
                "repair_target": repair_target,
                "agent_success": parent_agent_success,
                "agent_call_index": agent_calls,
                "duration_s": stage_args.get("_duration_s"),
                "agent_execution_time_s": stage_args.get("_agent_execution_time_s"),
                "completion_validation_audit": stage_args.get("_completion_validation_audit"),
            }
        )
        autodialect_validation = reuse_completion_validation_or_run_outer_validation(
            "validation_after_parent_repair",
            stage_args,
            validation_scope="full_manifest",
            trajectory_stage="parent_repair",
        )
        stage_handoff = _build_stage_handoff(
            project_path=project_path,
            source_root=spec.src_dir,
            base_ref=base_ref,
            group_execution=group_execution,
            stage="parent_repair_patch",
            validation=autodialect_validation,
            agent_success=parent_agent_success,
            agent_calls=agent_calls,
        )

    def run_focused_group_batch(
        group_batch: list[dict[str, Any]],
        batch_index: int,
        *,
        stage_prefix: str,
        patch_stage: str,
        validation_stage: str,
        response_label: str,
        terminal_failure: bool,
    ) -> dict[str, Any]:
        """Patch, validate, and record one focused group batch."""
        nonlocal agent_calls, autodialect_validation, response, stage_handoff, task_prompt

        batch_context = describe_group_batch(group_batch, split_manifest)
        group_ids = batch_context["rule_group_ids"]
        focus_rule_group_arg = batch_context["focus_rule_group_arg"]
        autodialect_stage = f"{stage_prefix}_{batch_index}"

        prevalidation_stage = f"pre_validation_{stage_prefix}_{batch_index}"
        prevalidation = run_outer_validation_stage(
            prevalidation_stage,
            focus_rule_group_id=focus_rule_group_arg,
            validation_scope="focused_group",
        )
        autodialect_validation = prevalidation
        passed_by_prevalidation = {
            group.get("rule_group_id")
            for group in group_batch
            if record_group_prevalidation(
                group_execution,
                group,
                stage=prevalidation_stage,
                validation=prevalidation,
            )
        }
        pending_group_batch = [
            group
            for group in group_batch
            if group.get("rule_group_id") not in passed_by_prevalidation
        ]
        if working_memory:
            working_memory.refresh_group_execution(group_execution)
        stage_handoff = _build_stage_handoff(
            project_path=project_path,
            source_root=spec.src_dir,
            base_ref=base_ref,
            group_execution=group_execution,
            stage=prevalidation_stage,
            group=describe_group_batch(
                pending_group_batch or group_batch,
                split_manifest,
            )["handoff_group"],
            validation=prevalidation,
            agent_success=prevalidation.get("passed"),
            agent_calls=agent_calls,
        )
        if not pending_group_batch:
            append_pipeline_stage(
                {
                    "stage": f"{patch_stage}_skipped",
                    "reason": "focused_prevalidation_passed",
                    "rule_group_ids": group_ids,
                    "batch_index": batch_index,
                    "batch_size": len(group_batch),
                    "merge_key": batch_context["merge_key"],
                    "refinement_key": batch_context["refinement_key"],
                }
            )
            return prevalidation
        group_batch = pending_group_batch
        batch_context = describe_group_batch(group_batch, split_manifest)
        group_ids = batch_context["rule_group_ids"]
        focus_rule_group_arg = batch_context["focus_rule_group_arg"]
        rule_prompt = build_prompt(
            info,
            is_summary=False,
            parser_gate=parser_gate_result,
            target_parser_gate=target_parser_gate_result,
            split_manifest=split_manifest,
            prompt_kind="rule",
            focus_rule_group_id=focus_rule_group_arg,
            stage_handoff=stage_handoff,
            include_memory_query=include_memory_query,
        )
        if working_memory:
            working_memory.set_stage(autodialect_stage, focus_group_ids=group_ids)
        rule_prompt = with_working_memory(rule_prompt)
        if not task_prompt:
            task_prompt = rule_prompt
        completion_validation_path = _write_completion_validation_spec(
            spec=spec,
            rule_name=rule_name,
            stage=autodialect_stage,
            split_manifest=split_manifest,
            case=case,
            focus_rule_group_id=batch_context["rule_group_id"],
            focus_rule_group_ids=group_ids,
            validation_scope="focused_group",
        )
        is_success, rule_response, stage_args = run_autodialect_stage(
            autodialect_stage,
            rule_prompt,
            completion_validation_path=completion_validation_path,
        )
        agent_calls += 1
        response = (
            f"{response}\n\n[{response_label} {batch_index}: {batch_context['label']}]\n{rule_response}"
            if response else rule_response
        )
        append_pipeline_stage(
            {
                "stage": patch_stage,
                "rule_group_id": batch_context["rule_group_id"],
                "rule_group_ids": group_ids,
                "batch_index": batch_index,
                "batch_size": len(group_batch),
                "merge_key": batch_context["merge_key"],
                "refinement_key": batch_context["refinement_key"],
                "terminal_failure": terminal_failure,
                "agent_success": is_success,
                "agent_call_index": agent_calls,
                "duration_s": stage_args.get("_duration_s"),
                "agent_execution_time_s": stage_args.get("_agent_execution_time_s"),
                "completion_validation_audit": stage_args.get("_completion_validation_audit"),
            }
        )
        autodialect_validation = reuse_completion_validation_or_run_outer_validation(
            validation_stage,
            stage_args,
            focus_rule_group_id=focus_rule_group_arg,
            validation_scope="focused_group",
            trajectory_stage=autodialect_stage,
        )
        for group in group_batch:
            record_group_attempt(
                group_execution,
                group,
                stage=patch_stage,
                validation=autodialect_validation,
                terminal_failure=terminal_failure,
            )
        if working_memory:
            working_memory.refresh_group_execution(group_execution)
        stage_handoff = _build_stage_handoff(
            project_path=project_path,
            source_root=spec.src_dir,
            base_ref=base_ref,
            group_execution=group_execution,
            stage=patch_stage,
            group=batch_context["handoff_group"],
            validation=autodialect_validation,
            agent_success=is_success,
            agent_calls=agent_calls,
        )
        return autodialect_validation

    # Patch queued groups, then repair remaining failures.
    if group_batches:
        for group_index, group_batch in enumerate(group_batches):
            run_focused_group_batch(
                group_batch,
                group_index,
                stage_prefix="rule_group",
                patch_stage="rule_group_patch",
                validation_stage="validation_after_group",
                response_label="Rule group batch",
                terminal_failure=False,
            )

        repair_plan = prepare_group_repair_plan(group_queue, group_execution, split_manifest)
        repair_group_batches = repair_plan.repair_group_batches
        if repair_plan.stage_record:
            append_pipeline_stage(repair_plan.stage_record)
        for repair_index, repair_batch in enumerate(repair_group_batches):
            run_focused_group_batch(
                repair_batch,
                repair_index,
                stage_prefix="rule_group_repair",
                patch_stage="rule_group_repair_patch",
                validation_stage="validation_after_group_repair",
                response_label="Rule group repair batch",
                terminal_failure=True,
            )
    else:
        rule_prompt = build_prompt(
            info,
            is_summary=False,
            parser_gate=parser_gate_result,
            target_parser_gate=target_parser_gate_result,
            split_manifest=split_manifest,
            prompt_kind="rule",
            max_diagnostic_hints_per_category=AUTODIALECT_MAX_DIAGNOSTIC_HINTS_PER_CATEGORY_PROMPT,
            stage_handoff=stage_handoff,
            include_memory_query=include_memory_query,
        )
        if working_memory:
            working_memory.set_stage("rule")
        rule_prompt = with_working_memory(rule_prompt)
        task_prompt = rule_prompt
        completion_validation_path = _write_completion_validation_spec(
            spec=spec,
            rule_name=rule_name,
            stage="rule",
            split_manifest=split_manifest,
            case=case,
        )
        is_success, rule_response, stage_args = run_autodialect_stage(
            "rule",
            rule_prompt,
            completion_validation_path=completion_validation_path,
        )
        agent_calls += 1
        response = f"{response}\n\n[Rule patch]\n{rule_response}" if response else rule_response
        append_pipeline_stage(
            {
                "stage": "rule_patch",
                "agent_success": is_success,
                "agent_call_index": agent_calls,
                "duration_s": stage_args.get("_duration_s"),
                "agent_execution_time_s": stage_args.get("_agent_execution_time_s"),
                "completion_validation_audit": stage_args.get("_completion_validation_audit"),
            }
        )
        autodialect_validation = reuse_completion_validation_or_run_outer_validation(
            "validation",
            stage_args,
            validation_scope="full_manifest",
            trajectory_stage="rule",
        )
        stage_handoff = _build_stage_handoff(
            project_path=project_path,
            source_root=spec.src_dir,
            base_ref=base_ref,
            group_execution=group_execution,
            stage="rule_patch",
            validation=autodialect_validation,
            agent_success=is_success,
            agent_calls=agent_calls,
        )

    # 5. Final validation/result: require full-manifest validation and package state.
    if not autodialect_validation:
        autodialect_validation = run_outer_validation_stage("validation")

    if (
        group_queue
        and autodialect_validation.get("passed")
        and not all_queued_groups_passed(group_execution, group_queue)
    ):
        autodialect_validation = group_execution_pending_validation(
            group_execution,
            group_queue,
        )
        append_pipeline_stage(
            {
                "stage": "group_execution_pending",
                "validation_scope": "group_queue",
                "passed": False,
                "pending_group_ids": autodialect_validation.get("pending_group_ids", []),
                "failed_groups": autodialect_validation.get("failed_groups", []),
                "repair_target": autodialect_validation.get("repair_target"),
            }
        )
        if working_memory:
            working_memory.refresh_group_execution(group_execution)
            working_memory.record_validation(
                stage="group_execution_pending",
                validation=autodialect_validation,
                focus_group_ids=autodialect_validation.get("pending_group_ids", []),
            )

    # A direct full-manifest parent failure previously bypassed the repair block
    # below because that block was only reachable from a passing validation.
    if should_run_parent_repair(autodialect_validation):
        run_parent_repair("validation")

    if (
        parser_gate_result.get("status") == "parse_ok"
        and autodialect_validation.get("passed")
        and all_queued_groups_passed(group_execution, group_queue)
    ):
        if autodialect_validation.get("validation_scope") == "full_manifest":
            append_pipeline_stage(
                _validation_stage_record(
                    stage_name="final_full_validation",
                    validation=autodialect_validation,
                    scope="full_manifest",
                    focus_rule_group_id=None,
                    validation_source="prior_full_manifest_validation",
                )
            )
        else:
            autodialect_validation = run_outer_validation_stage(
                "final_full_validation",
                validation_scope="full_manifest",
            )
        if should_run_parent_repair(autodialect_validation):
            run_parent_repair("final_full_validation")

    is_success = bool(autodialect_validation.get("passed"))

    # Return the completed pipeline state to run_gen().
    return build_pipeline_result()


def run_gen(
    spec: ExperimentSpec,
    test_cases: list[dict[str, Any]],
    *,
    limit: int | None = None,
    case_indices: list[int] | None = None,
    is_summary: bool = False,
    must_patch: bool = True,
    console_type: str | None = None,
    autodialect_config_file: str = AUTODIALECT_CONFIG_FILE,
    model_runtime: ModelRuntime | None = None,
) -> dict[str, Any]:
    """Run generation for selected cases and persist incremental outputs."""
    if model_runtime is None:
        model_runtime = resolve_model_runtime(autodialect_config_file)
    ensure_agent_available(spec.agent_id, model_runtime)
    validate_experiment_spec(spec)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if case_indices is None:
        indexed_test_cases = list(enumerate(test_cases))
    else:
        negative_indices = [index for index in case_indices if index < 0]
        if negative_indices:
            raise ValidationError(
                f"Case indices must be >= 0 for {spec.testcase_file}: {negative_indices[:10]}"
            )
        out_of_range = [index for index in case_indices if index >= len(test_cases)]
        if out_of_range:
            raise ValidationError(
                f"Case index out of range for {spec.testcase_file}: {out_of_range[:10]}"
            )
        indexed_test_cases = [(index, test_cases[index]) for index in case_indices]
    # Apply --limit after optional case-index filtering; tuple[0] stays the original testcase index.
    cases_to_run = take_first_n_items(indexed_test_cases, limit)
    result_data = load_result_data(spec.result_file)
    if scrub_result_deprecated_success_fields(result_data):
        save_json(spec.result_file, result_data)
    if not cases_to_run:
        return result_data

    prepare_agent_runtime(spec.agent_id, model_runtime)
    cli_version = get_agent_cli_version(spec.agent_id, model_runtime)
    autodialect_memory_enabled = (
        spec.agent_id == AUTODIALECT_AGENT
        and not is_summary
        and resolve_agent_memory_enabled(autodialect_config_file)
    )

    for no, case in tqdm(cases_to_run, total=len(cases_to_run)):
        src_dialect = case_src_dialect(spec, case)
        tgt_dialect = case_tgt_dialect(spec, case)
        source_sql = case_source_sql(spec, case)
        expected_sql = case_expected_sql(spec, case)
        rule_name = f"{src_dialect}_{tgt_dialect}_{no}"
        existing_record = result_data.get(rule_name)
        if _should_skip_existing_record(
            existing_record,
            agent_id=spec.agent_id,
            is_summary=is_summary,
        ):
            continue

        project_path = f"{spec.dst_dir}_{rule_name}"
        info = build_case_info(
            project_path=project_path,
            src_db=src_dialect,
            tgt_db=tgt_dialect,
            src_sql=source_sql,
            tgt_sql=expected_sql,
        )
        info["rule_name"] = rule_name
        runner_args = {"model_runtime": model_runtime}
        include_memory_query = autodialect_memory_enabled
        if spec.agent_id == AUTODIALECT_AGENT:
            runner_args.update(
                build_autodialect_extra_args(
                    spec.result_folder,
                    spec.agent_id,
                    spec.model_name,
                    rule_name,
                    config_file=autodialect_config_file,
                    must_patch=must_patch,
                    console_type=console_type,
                    timestamp=timestamp,
                )
            )

        # Build patch prompts only after the working copy is ready for the current case.
        task_prompt = build_prompt(info, is_summary=is_summary) if is_summary else ""
        base_ref = None
        response = ""
        execution_time = 0.0
        is_success = False
        memory_audit: dict[str, Any] = {}
        parser_gate_result: dict[str, Any] = {}
        target_parser_gate_result: dict[str, Any] = {}
        split_manifest: dict[str, Any] = {}
        autodialect_validation: dict[str, Any] = {}
        group_execution: dict[str, Any] = {}
        stage_handoff: dict[str, Any] | None = None
        pipeline_stages: list[dict[str, Any]] = []
        trajectory_capture = CaseTrajectoryCapture(spec, rule_name)
        if spec.agent_id == AUTODIALECT_AGENT and not is_summary:
            trajectory_capture.clear_case_dir()

        if is_summary:
            is_success = True
        else:
            is_success = prepare_case_worktree(spec.src_dir, project_path)

        if not is_success:
            response = f"Failed to prepare clean case worktree: {project_path}"
        else:
            base_ref = get_git_head(project_path)
            if not is_summary:
                autodialect_pipeline = _run_autodialect_case_pipeline(
                    spec=spec,
                    case=case,
                    rule_name=rule_name,
                    project_path=project_path,
                    info=info,
                    runner_args=runner_args,
                    include_memory_query=include_memory_query,
                    base_ref=base_ref,
                    trajectory_capture=trajectory_capture,
                    memory_enabled=autodialect_memory_enabled,
                )
                task_prompt = autodialect_pipeline.task_prompt
                response = autodialect_pipeline.response
                execution_time = autodialect_pipeline.execution_time
                is_success = autodialect_pipeline.is_success
                memory_audit = autodialect_pipeline.memory_audit
                parser_gate_result = autodialect_pipeline.parser_gate_result
                target_parser_gate_result = autodialect_pipeline.target_parser_gate_result
                split_manifest = autodialect_pipeline.split_manifest
                autodialect_validation = autodialect_pipeline.autodialect_validation
                group_execution = autodialect_pipeline.group_execution
                stage_handoff = autodialect_pipeline.stage_handoff
                pipeline_stages = autodialect_pipeline.pipeline_stages
            else:
                task_prompt = build_prompt(info, is_summary=True)
                is_success, response, execution_time, memory_audit = run_agent(
                    spec.agent_id,
                    project_path,
                    task_prompt,
                    runner_args,
                    is_summary=True,
                )

        if spec.agent_id == AUTODIALECT_AGENT and not trajectory_capture.autodialect_trajectory_copied_during_run:
            trajectory_file = runner_args["trajectory_file"]
            if os.path.isfile(trajectory_file):
                trajectory_capture.clear_case_dir()
                trajectory_capture.copy_autodialect_stage(trajectory_file, stage="single_run")

        trajectory_manifest_file = trajectory_capture.write_manifest(pipeline_stages)

        record = result_data.setdefault(rule_name, {})
        case_metadata = _case_metadata(spec, rule_name, case)
        if is_summary:
            record["case_metadata"] = case_metadata
            record["test_case"] = dict(case)
            record["summary"] = response
            record["agent_id"] = spec.agent_id
            record["model_runtime"] = model_runtime.metadata()
            record["cli_version"] = cli_version
            record["native_trajectory_files"] = trajectory_capture.native_trajectory_files
            record["trajectory_copy_files"] = trajectory_capture.copied_trajectory_files
            record["trajectory_manifest_file"] = trajectory_manifest_file
        else:
            # Persist the generated patch delta alongside the raw agent response.
            # Collect this best-effort even when the agent exits unsuccessfully,
            # because the CLI may have written useful files before returning a
            # timeout/context/provider error. This does not change agent success.
            gen_code, file_changes = get_git_diff(
                project_path,
                os.path.join(spec.result_folder, rule_name),
                base_ref=base_ref,
            )
            if is_success:
                if must_patch and not file_changes:
                    is_success = False

            result_data[rule_name] = compact_top_level_fields({
                "case_metadata": case_metadata,
                "test_case": dict(case),
                "prompt": task_prompt,
                "is_success": is_success,
                "execution_time": execution_time,
                "agent_id": spec.agent_id,
                "model_runtime": model_runtime.metadata(),
                "cli_version": cli_version,
                "response": response,
                "gen_code": gen_code,
                "file_changes": file_changes,
                "memory_audit": memory_audit,
                "native_trajectory_files": trajectory_capture.native_trajectory_files,
                "trajectory_copy_files": trajectory_capture.copied_trajectory_files,
                "trajectory_manifest_file": trajectory_manifest_file,
                "pipeline_stages": pipeline_stages,
                "group_execution": group_execution,
                "stage_handoff": stage_handoff,
                "parser_gate": parser_gate_result,
                "target_parser_gate": target_parser_gate_result,
                "split_manifest": split_manifest,
                "autodialect_validation": autodialect_validation,
            })

        save_json(spec.result_file, result_data)

    return result_data


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for generation entrypoint options."""
    parser = argparse.ArgumentParser(description="Run AutoDialect generation.")
    add_experiment_selection_args(
        parser,
        exp_choices=EXPERIMENT_RUN_CONFIGS.keys(),
        require_exp=False,
    )
    runtime_group = parser.add_argument_group("runtime controls")
    runtime_group.add_argument("--summary", action="store_true", help="Run summary mode instead of patch generation.")
    runtime_group.add_argument(
        "--console-type",
        choices=("simple",),
        help="AutoDialect console type.",
    )
    runtime_group.add_argument(
        "--autodialect-config-file",
        help="AutoDialect config YAML path used by each testcase subprocess.",
    )
    runtime_group.add_argument(
        "--must-patch",
        dest="must_patch",
        action="store_true",
        default=True,
        help="Require AutoDialect to generate a patch when applicable.",
    )
    runtime_group.add_argument(
        "--no-must-patch",
        dest="must_patch",
        action="store_false",
        help="Run AutoDialect without requiring a patch.",
    )
    runtime_group.add_argument("--limit", type=int, default=None, help="Only process the first N cases.")
    runtime_group.add_argument(
        "--case-indices-file",
        help="JSON list of zero-based testcase indices to run.",
    )
    parser.set_defaults(
        exp=DEFAULT_RUN_CONFIG["exp"],
        agent=None,
        model_alias=None,
        variant=None,
        input_id=None,
        testcase_file=None,
        src_dir=None,
        dst_dir=None,
        result_folder=None,
        result_file=None,
        summary=DEFAULT_RUN_CONFIG["summary"],
        console_type=DEFAULT_RUN_CONFIG["console_type"],
        autodialect_config_file=None,
        must_patch=DEFAULT_RUN_CONFIG["must_patch"],
        limit=DEFAULT_RUN_CONFIG["limit"],
        case_indices_file=None,
    )
    return parser


def main(args) -> dict[str, Any]:
    """Execute one generation run from resolved CLI or in-file configuration."""
    run_config = resolve_experiment_run_config(
        args,
        override_fields=(
            "input_id",
            "testcase_file",
            "src_dir",
            "dst_dir",
            "result_folder",
            "result_file",
            "summary",
            "console_type",
            "autodialect_config_file",
            "must_patch",
            "limit",
            "case_indices_file",
        ),
    )
    spec = experiment_spec_from_run_config(run_config)

    model_runtime = None
    preflight_errors = []
    try:
        validate_experiment_spec(spec)
    except ValidationError as exc:
        preflight_errors.append(str(exc))
    try:
        model_runtime = resolve_model_runtime(run_config["autodialect_config_file"])
    except ValidationError as exc:
        preflight_errors.append(str(exc))
    try:
        ensure_agent_available(spec.agent_id, model_runtime)
    except (RuntimeError, ValidationError) as exc:
        preflight_errors.append(str(exc))
    if preflight_errors:
        # Aggregate preflight failures so the caller sees the full validation state.
        formatted_errors = ["Preflight validation failed:"]
        for error in preflight_errors:
            for line in error.splitlines():
                if line == "Preflight validation failed:":
                    continue
                formatted_errors.append(line if line.startswith("- ") else f"- {line}")
        raise ValidationError("\n".join(formatted_errors))

    test_cases = load_online_cases(spec.testcase_file)
    case_indices = _load_case_indices_file(run_config.get("case_indices_file"))
    return run_gen(
        spec,
        test_cases,
        limit=run_config["limit"],
        case_indices=case_indices,
        is_summary=run_config["summary"],
        must_patch=run_config["must_patch"],
        console_type=run_config["console_type"],
        autodialect_config_file=run_config["autodialect_config_file"],
        model_runtime=model_runtime,
    )


def run_default_configs() -> dict[str, Any]:
    """Run every in-file default configuration when no CLI args are provided."""
    results = {}
    for config_name, run_config in build_default_run_configs():
        results[config_name] = main(namespace_from_run_config(DEFAULT_RUN_CONFIG, run_config))
    if len(results) == 1:
        return next(iter(results.values()))
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(build_parser().parse_args())
    else:
        run_default_configs()
