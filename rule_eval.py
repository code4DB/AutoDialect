#!/usr/bin/env python3
"""Validation entrypoints for AutoDialect generation results.

1. Rebuild validation worktrees from generation records and replay generated
   rule files into a clean translation-engine copy.
2. Re-run SQLGlot transpilation against each recorded case and persist
   AutoDialect validation status.
3. Build and validate SQL unit-test split manifests so generation and
   evaluation records share the same focused evidence contract.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Any

from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from code_utils.execution_runtime import safe_relative_path
from code_utils.common_tool import _SqlglotWorker, sql_key
from code_utils.experiment_utils import (
    DEFAULT_RUN_CONFIG,
    EXPERIMENT_RUN_CONFIGS,
    ExperimentSpec,
    ValidationError,
    _load_case_indices_file,
    add_experiment_selection_args,
    build_default_run_configs,
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
    scrub_deprecated_success_fields,
    scrub_result_deprecated_success_fields,
    take_first_n_items,
    validate_experiment_spec,
    validated_result_file,
)
from code_utils.sql_ut.split_validator import validate_manifest
from code_utils.sql_ut.sql_ut_splitter import build_split_manifest


# 1. Validation worktree rebuild helpers.
def _copy_case_files(src_dir: str, dst_dir: str, project_path: str, file_changes: list[str]) -> None:
    """Rebuild the validation worktree and best-effort replay generated files."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    for file in file_changes:
        try:
            safe_file = safe_relative_path(file)
            if safe_file is None:
                continue

            src_file = os.path.join(project_path, safe_file)
            dst_file = os.path.join(dst_dir, safe_file)
            if not os.path.exists(src_file):
                continue

            dst_parent = os.path.dirname(dst_file)
            if dst_parent:
                os.makedirs(dst_parent, exist_ok=True)
            shutil.copyfile(src_file, dst_file)
        except (OSError, ValueError):
            continue


# 2. Validation status and SQLGlot worker helpers.
def clear_autodialect_validation(record: dict[str, Any]) -> None:
    """Remove strict AutoDialect validation fields when strict validation is not refreshed."""
    scrub_deprecated_success_fields(record)
    record.pop("autodialect_validation", None)


def _record_autodialect_validation_error(
    record: dict[str, Any],
    *,
    error_type: str,
    error: str,
) -> None:
    scrub_deprecated_success_fields(record)
    record["autodialect_validation"] = {
        "passed": False,
        "failed_stage": "validation_error",
        "error_type": error_type,
        "error": error,
    }


def _transpile_with_worker(package_root: str, sys_path_backup: list[str], sql: str, read: str, write: str) -> str:
    """Run one transpilation in an isolated sqlglot worker process."""
    worker = _SqlglotWorker(package_root, sys_path_backup)
    try:
        return worker.transpile(sql=sql, read=read, write=write)
    finally:
        worker.close()


def _eval_rule_name(spec: ExperimentSpec, case: dict[str, Any], index: int) -> str:
    src_dialect = case_src_dialect(spec, case)
    tgt_dialect = case_tgt_dialect(spec, case)
    return f"{src_dialect}_{tgt_dialect}_{index}"


def _select_eval_rule_artifacts(
    *,
    spec: ExperimentSpec,
    case: dict[str, Any],
    index: int,
) -> tuple[str, str]:
    rule_name = _eval_rule_name(spec, case, index)
    return rule_name, os.path.join(spec.result_folder, rule_name)


# 3. Evaluation entrypoints.
def run_eval(
    spec: ExperimentSpec,
    test_cases: list[dict[str, Any]],
    *,
    limit: int | None = None,
    case_indices: list[int] | None = None,
    validate_splits: bool = False,
    validate_variants: bool = False,
) -> dict[str, Any]:
    """Validate generated case outputs against the expected target SQL."""
    validate_experiment_spec(spec)
    if not os.path.isfile(spec.result_file):
        raise ValidationError(f"Result file does not exist: {spec.result_file}")

    result_data = load_result_data(spec.result_file)
    deprecated_fields_removed = scrub_result_deprecated_success_fields(result_data)
    validated_file = validated_result_file(spec.result_file)
    sys_path_backup = list(sys.path)

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

    cases_to_run = take_first_n_items(indexed_test_cases, limit)
    if not cases_to_run:
        if deprecated_fields_removed:
            save_json(validated_file, result_data)
        return result_data

    for no, case in tqdm(cases_to_run, total=len(cases_to_run)):
        src_dialect = case_src_dialect(spec, case)
        tgt_dialect = case_tgt_dialect(spec, case)
        source_sql = case_source_sql(spec, case)
        expected_sql = case_expected_sql(spec, case)
        rule_name, project_path = _select_eval_rule_artifacts(
            spec=spec,
            case=case,
            index=no,
        )
        record = result_data.setdefault(rule_name, {})
        scrub_deprecated_success_fields(record)
        file_changes = list(record.get("file_changes", []))
        record["test_case"] = case
        if not os.path.isdir(project_path):
            error = f"Missing project path: {project_path}"
            record["is_success"] = False
            record["test_result"] = error
            if validate_splits or validate_variants:
                _record_autodialect_validation_error(
                    record,
                    error_type="MissingProjectPath",
                    error=error,
                )
            else:
                clear_autodialect_validation(record)
            record.pop("group_variant_validation", None)
            save_json(validated_file, result_data)
            continue

        try:
            # Recreate a clean validation tree before replaying generated file changes.
            _copy_case_files(
                spec.src_dir,
                spec.dst_dir,
                project_path,
                file_changes,
            )

            # Compare normalized SQL text rather than raw formatting differences.
            translated_sql = _transpile_with_worker(
                spec.dst_dir,
                sys_path_backup,
                sql=source_sql,
                read=src_dialect,
                write=tgt_dialect,
            )

            parent_is_success = sql_key(translated_sql) == sql_key(expected_sql)
            record["is_success"] = parent_is_success
            record["test_result"] = translated_sql

            if validate_splits or validate_variants:
                # A schema-current generation manifest can still be stale after
                # replaying parser changes into this validation worktree.
                manifest = build_split_manifest(
                    project_path=spec.dst_dir,
                    src_sql=source_sql,
                    expected_sql=expected_sql,
                    src_dialect=src_dialect,
                    tgt_dialect=tgt_dialect,
                    parent_case_id=rule_name,
                )
                record["split_manifest"] = manifest

                record["autodialect_validation"] = validate_manifest(
                    package_root=spec.dst_dir,
                    sys_path_backup=sys_path_backup,
                    manifest=manifest,
                    parent_source_sql=source_sql,
                    parent_expected_sql=expected_sql,
                    src_dialect=src_dialect,
                    tgt_dialect=tgt_dialect,
                    validate_variants=validate_variants,
                )
                record["is_success"] = bool(record["autodialect_validation"].get("passed"))
            else:
                clear_autodialect_validation(record)

            record.pop("group_variant_validation", None)
        except Exception as exc:
            record["is_success"] = False
            record["test_result"] = str(exc)
            if validate_splits or validate_variants:
                _record_autodialect_validation_error(
                    record,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            else:
                clear_autodialect_validation(record)
            record.pop("group_variant_validation", None)

        save_json(validated_file, result_data)

    return result_data


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for validation entrypoint options."""
    parser = argparse.ArgumentParser(description="Validate AutoDialect generation results.")
    add_experiment_selection_args(
        parser,
        exp_choices=EXPERIMENT_RUN_CONFIGS.keys(),
        require_exp=False,
    )
    parser.add_argument("--limit", type=int, default=None, help="Only validate the first N cases.")
    parser.add_argument(
        "--validate-splits",
        action="store_true",
        help="Also validate AutoDialect split/group UTs into appended result fields.",
    )
    parser.add_argument(
        "--validate-variants",
        action="store_true",
        help="Validate generated AutoDialect SQL variants in addition to parent and split/group UTs; implies split validation.",
    )
    parser.add_argument(
        "--case-indices-file",
        help=(
            "JSON list of testcase indices. Use the same file for generation "
            "and evaluation subset runs."
        ),
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
        limit=DEFAULT_RUN_CONFIG["limit"],
        case_indices_file=None,
        validate_splits=False,
        validate_variants=False,
    )
    return parser


def main(args) -> dict[str, Any]:
    """Execute one validation run from resolved CLI or in-file configuration."""
    run_config = resolve_experiment_run_config(
        args,
        override_fields=(
            "input_id",
            "testcase_file",
            "src_dir",
            "dst_dir",
            "result_folder",
            "result_file",
            "limit",
            "case_indices_file",
        ),
    )
    spec = experiment_spec_from_run_config(run_config)
    validate_experiment_spec(spec)
    test_cases = load_online_cases(spec.testcase_file)
    case_indices = _load_case_indices_file(run_config.get("case_indices_file"))
    return run_eval(
        spec,
        test_cases,
        limit=run_config["limit"],
        case_indices=case_indices,
        validate_splits=getattr(args, "validate_splits", False),
        validate_variants=getattr(args, "validate_variants", False),
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
