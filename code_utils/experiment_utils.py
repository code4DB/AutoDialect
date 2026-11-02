#!/usr/bin/env python3
"""Shared testcase, result, configuration, and AutoDialect trajectory helpers."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Sequence

from code_utils.constants import (
    CODE_OUTPUT_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SQLGLOT_WORKTREE_PREFIX,
    AUTODIALECT_AGENT,
    AUTODIALECT_CONFIG_FILE,
)


# 1. Shared types and agent ids.
class ValidationError(RuntimeError):
    """Raised when required local inputs are missing or inconsistent."""


SUPPORTED_AGENT_IDS = (AUTODIALECT_AGENT,)


@dataclass(frozen=True)
class ExperimentSpec:
    preset_name: str
    input_id: str
    variant: str
    agent_id: str
    model_name: str
    testcase_file: str
    src_dir: str
    dst_dir: str
    result_folder: str
    result_file: str
    missing_testcase_hint: str | None = None


def experiment_spec_from_run_config(config: dict[str, Any]) -> ExperimentSpec:
    """Build the execution spec from a fully resolved run config."""
    return ExperimentSpec(
        preset_name=config["exp"],
        input_id=config["input_id"],
        variant=config["variant"],
        agent_id=config["agent"],
        model_name=config["model_alias"],
        testcase_file=config["testcase_file"],
        src_dir=config["src_dir"],
        dst_dir=config["dst_dir"],
        result_folder=config["result_folder"],
        result_file=config["result_file"],
        missing_testcase_hint=config.get("missing_testcase_hint"),
    )


DEFAULT_RUN_CONFIG = {
    "exp": "online",
    "input_id": "online",
    "agent": AUTODIALECT_AGENT,
    "model_alias": "online",
    "variant": "default",
    "testcase_file": os.getenv("AUTODIALECT_TESTCASE_FILE", ""),
    "src_dir": os.getenv("AUTODIALECT_ENGINE_SOURCE", ""),
    "dst_dir": os.path.join(CODE_OUTPUT_DIR, "online"),
    "result_folder": os.path.join(RESULTS_DIR, "online"),
    "result_file": os.path.join(RESULTS_DIR, "online", "results.json"),
    "summary": False,
    "console_type": "simple",
    "autodialect_config_file": AUTODIALECT_CONFIG_FILE,
    "must_patch": True,
    "limit": None,
    "missing_testcase_hint": "Pass --testcase-file with an online-case JSON input.",
}

EXPERIMENT_RUN_CONFIGS = {"online": DEFAULT_RUN_CONFIG}


def build_default_run_configs() -> list[tuple[str, dict[str, Any]]]:
    """Return named in-file run configs used when no CLI args are provided."""
    return [
        ("default", dict(DEFAULT_RUN_CONFIG)),
    ]


# 3. JSON/result/testcase helpers.
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as rf:
        return json.load(rf)


def load_result_data(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValidationError(f"Result file must contain a JSON object: {path}")
    return data


def _has_current_validated_success(record: dict[str, Any]) -> bool:
    from code_utils.sql_ut.sql_ut_splitter import is_current_split_manifest

    validation = record.get("autodialect_validation")
    strict_passed = isinstance(validation, dict) and validation.get("passed") is True
    split_build_warning_passed = (
        isinstance(validation, dict)
        and validation.get("warning_stage") == "ut_split"
        and validation.get("warning_reason") == "split_manifest_build_failed"
    )
    return (
        record.get("is_success") is True
        and strict_passed
        and (
            is_current_split_manifest(record.get("split_manifest"))
            or split_build_warning_passed
        )
    )


def _should_skip_existing_record(
    existing_record: Any,
    *,
    agent_id: str,
    is_summary: bool,
) -> bool:
    if is_summary or not isinstance(existing_record, dict):
        return False
    return _has_current_validated_success(existing_record)


def _load_case_indices_file(path: str | None) -> list[int] | None:
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as rf:
        raw = rf.read().strip()
    if not raw:
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = [part for line in raw.splitlines() for part in line.split(",")]
    if isinstance(payload, dict):
        if "case_indices" in payload:
            payload = payload.get("case_indices")
        else:
            payload = payload.get("indices")
    if not isinstance(payload, list):
        raise ValidationError(f"Case indices file must contain a JSON list: {path}")

    indices: list[int] = []
    seen: set[int] = set()
    for item in payload:
        try:
            index = int(str(item).strip())
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid case index in {path}: {item!r}") from exc
        if index < 0:
            raise ValidationError(f"Case indices must be >= 0 in {path}: {index}")
        if index not in seen:
            indices.append(index)
            seen.add(index)
    return indices


def scrub_deprecated_success_fields(record: dict[str, Any]) -> bool:
    """Remove legacy top-level success aliases; is_success is the canonical field."""
    removed = False
    for key in (
        "legacy_is_success",
        "autodialect_is_success",
        "autodialect_validation_mode",
    ):
        if key in record:
            record.pop(key, None)
            removed = True
    return removed


def scrub_result_deprecated_success_fields(result_data: dict[str, Any]) -> bool:
    removed = False
    for record in (result_data or {}).values():
        if isinstance(record, dict):
            removed = scrub_deprecated_success_fields(record) or removed
    return removed


def save_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as wf:
            json.dump(payload, wf, indent=4, ensure_ascii=False)
            wf.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def take_first_n_items(items: Sequence[Any], limit: int | None) -> list[Any]:
    """Return the first `limit` items; None keeps every item."""
    if limit is None:
        return list(items)
    if limit < 0:
        raise ValidationError("limit must be >= 0")
    return list(items[:limit])


CASE_SRC_DIALECT_KEY = "_src_dialect"
CASE_TGT_DIALECT_KEY = "_tgt_dialect"
ONLINE_CASE_REQUIRED_FIELDS = (
    "id",
    "source_dialect",
    "target_dialect",
    "source_sql",
    "expected_sql",
)


def load_online_cases(testcase_file: str) -> list[dict[str, Any]]:
    """Load the standalone online-case JSON format.

    The input is a JSON list. Every item contains an id, source and target
    dialect names, source SQL, and expected SQL. The normalized internal keys
    keep the rest of the pipeline independent from the external schema.
    """
    payload = _load_json(testcase_file)
    if not isinstance(payload, list):
        raise ValidationError("Online testcase file must contain a JSON list.")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(payload):
        if not isinstance(raw_case, dict):
            raise ValidationError(f"Online testcase at index {index} must be a JSON object.")

        missing = [
            field
            for field in ONLINE_CASE_REQUIRED_FIELDS
            if raw_case.get(field) in (None, "")
        ]
        if missing:
            raise ValidationError(
                f"Online testcase at index {index} is missing required fields: "
                + ", ".join(missing)
            )

        raw_case_id = raw_case["id"]
        raw_source = raw_case["source_dialect"]
        raw_target = raw_case["target_dialect"]
        if isinstance(raw_case_id, bool) or not isinstance(raw_case_id, (str, int)):
            raise ValidationError(
                f"Online testcase at index {index} must use a string or integer id."
            )
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise ValidationError(
                f"Online testcase at index {index} must use a non-empty source_dialect string."
            )
        if not isinstance(raw_target, str) or not raw_target.strip():
            raise ValidationError(
                f"Online testcase at index {index} must use a non-empty target_dialect string."
            )

        case_id = str(raw_case_id).strip()
        source = raw_source.strip()
        target = raw_target.strip()
        source_sql = raw_case["source_sql"]
        expected_sql = raw_case["expected_sql"]

        if not case_id:
            raise ValidationError(f"Online testcase at index {index} has an empty id.")
        if case_id in seen_ids:
            raise ValidationError(f"Duplicate online testcase id: {case_id!r}")
        if source == target:
            raise ValidationError(
                f"Online testcase {case_id!r} must use different source and target dialects."
            )
        if not isinstance(source_sql, str) or not source_sql.strip():
            raise ValidationError(f"Online testcase {case_id!r} has empty source_sql.")
        if not isinstance(expected_sql, str) or not expected_sql.strip():
            raise ValidationError(f"Online testcase {case_id!r} has empty expected_sql.")

        seen_ids.add(case_id)
        item = dict(raw_case)
        item[CASE_SRC_DIALECT_KEY] = source
        item[CASE_TGT_DIALECT_KEY] = target
        item[source] = source_sql
        item[target] = expected_sql
        item["rule_pattern"] = f"online:{case_id}"
        cases.append(item)

    if not cases:
        raise ValidationError("Online testcase file must contain at least one case.")
    return cases


def case_src_dialect(spec: ExperimentSpec, case: dict[str, Any]) -> str:
    value = str(case.get(CASE_SRC_DIALECT_KEY) or "").strip()
    if not value:
        raise ValidationError("Case has no source_dialect.")
    return value


def case_tgt_dialect(spec: ExperimentSpec, case: dict[str, Any]) -> str:
    value = str(case.get(CASE_TGT_DIALECT_KEY) or "").strip()
    if not value:
        raise ValidationError("Case has no target_dialect.")
    return value


def case_source_sql(spec: ExperimentSpec, case: dict[str, Any]) -> str:
    value = case.get("source_sql")
    if not isinstance(value, str):
        raise ValidationError("Case has no source_sql.")
    return value


def case_expected_sql(spec: ExperimentSpec, case: dict[str, Any]) -> str:
    value = case.get("expected_sql")
    if not isinstance(value, str):
        raise ValidationError("Case has no expected_sql.")
    return value


# 4. Experiment layout, config resolution, and preflight helpers.
def model_alias(model_name: str) -> str:
    return model_name.split("/")[-1]


def derive_experiment_layout(
    agent_id: str,
    model_name: str,
    variant: str,
) -> tuple[str, str, str]:
    resolved_model_alias = model_alias(model_name)
    result_folder = os.path.join(
        PROJECT_ROOT,
        "results",
        f"{agent_id}_{resolved_model_alias}_{variant}",
    )
    dst_dir = os.path.join(
        CODE_OUTPUT_DIR,
        f"{SQLGLOT_WORKTREE_PREFIX}_{agent_id}_{variant}",
    )
    result_file = os.path.join(
        result_folder,
        f"{agent_id}_{resolved_model_alias}_{variant}_results.json",
    )
    return dst_dir, result_folder, result_file


def namespace_from_run_config(
    default_run_config: dict[str, Any],
    run_config: dict[str, Any],
) -> argparse.Namespace:
    """Convert one in-file run config dictionary into an argparse namespace."""
    return argparse.Namespace(**{**default_run_config, **run_config})


def add_experiment_selection_args(
    parser: argparse.ArgumentParser,
    *,
    exp_choices: Sequence[str],
    require_exp: bool = True,
) -> argparse.ArgumentParser:
    selection_group = parser.add_argument_group(
        "run selection",
        "Choose the run preset and optional output metadata overrides.",
    )
    path_group = parser.add_argument_group(
        "optional path overrides",
        "Leave these unset to use the selected run preset values.",
    )
    selection_group.add_argument(
        "--exp",
        required=require_exp,
        choices=sorted(exp_choices),
        help="Run preset.",
    )
    selection_group.add_argument(
        "--agent",
        choices=SUPPORTED_AGENT_IDS,
        help="Agent implementation used for the run.",
    )
    selection_group.add_argument(
        "--model-alias",
        help="Override model alias used in result naming.",
    )
    selection_group.add_argument(
        "--variant",
        help="Override run variant tag.",
    )
    selection_group.add_argument("--input-id", help="Optional input label for result metadata.")
    path_group.add_argument(
        "--testcase-file",
        help="Online-case JSON file path.",
    )
    path_group.add_argument(
        "--src-dir",
        help="Translation-engine repository path.",
    )
    path_group.add_argument(
        "--dst-dir",
        help="Worktree output directory.",
    )
    path_group.add_argument(
        "--result-folder",
        help="Result directory.",
    )
    path_group.add_argument(
        "--result-file",
        help="Result JSON file path.",
    )
    return parser


def resolve_experiment_run_config(
    args: argparse.Namespace,
    *,
    override_fields: Sequence[str],
) -> dict[str, Any]:
    """Resolve CLI args or an in-file namespace into one concrete experiment config."""
    if args.exp not in EXPERIMENT_RUN_CONFIGS:
        raise ValidationError(f"Unsupported experiment instance: {args.exp}")

    config = dict(EXPERIMENT_RUN_CONFIGS[args.exp])

    identity_changed = False
    for arg_name, config_key in (
        ("agent", "agent"),
        ("model_alias", "model_alias"),
        ("variant", "variant"),
        ("input_id", "input_id"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None and value != config[config_key]:
            config[config_key] = value
            identity_changed = True

    if identity_changed:
        dst_dir, result_folder, result_file = derive_experiment_layout(
            config["agent"],
            config["model_alias"],
            config["variant"],
        )
        config["dst_dir"] = dst_dir
        config["result_folder"] = result_folder
        config["result_file"] = result_file

    for arg_name in override_fields:
        value = getattr(args, arg_name, None)
        if value is not None:
            config[arg_name] = value

    return config


def validate_experiment_spec(spec: ExperimentSpec) -> None:
    errors = []

    if not os.path.isfile(spec.testcase_file):
        message = f"Testcase file does not exist: {spec.testcase_file}"
        if spec.missing_testcase_hint:
            message = f"{message} {spec.missing_testcase_hint}"
        errors.append(message)

    if not os.path.isdir(spec.src_dir):
        errors.append(f"Source translation-engine directory does not exist: {spec.src_dir}")

    if errors:
        raise ValidationError("Preflight validation failed:\n- " + "\n- ".join(errors))


def build_case_info(
    project_path: str,
    src_db: str,
    tgt_db: str,
    src_sql: str,
    tgt_sql: str,
) -> dict[str, str]:
    source_path = f"{project_path}/sqlglot/dialects/{src_db}.py"
    target_path = f"{project_path}/sqlglot/dialects/{tgt_db}.py"
    return {
        "src_db": src_db,
        "tgt_db": tgt_db,
        "src_sql": src_sql,
        "tgt_sql": tgt_sql,
        "source_path": source_path,
        "target_path": target_path,
    }


def _case_metadata(spec: ExperimentSpec, rule_name: str, case: dict[str, Any]) -> dict[str, Any]:
    source = case_src_dialect(spec, case)
    target = case_tgt_dialect(spec, case)
    source_sql = case_source_sql(spec, case)
    expected_sql = case_expected_sql(spec, case)
    metadata = {
        "rule_name": rule_name,
        "src_dialect": source,
        "tgt_dialect": target,
        "source_sql": source_sql,
        "expected_sql": expected_sql,
        "input_id": spec.input_id,
        "variant": spec.variant,
        "testcase_file": spec.testcase_file,
    }
    optional_fields = {"rule_pattern": case.get("rule_pattern")}
    metadata.update({key: value for key, value in optional_fields.items() if value is not None})
    return metadata


def build_autodialect_extra_args(
    result_folder: str,
    agent_id: str,
    model_name: str,
    rule_name: str,
    *,
    config_file: str = AUTODIALECT_CONFIG_FILE,
    must_patch: bool,
    console_type: str | None,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "config_file": config_file,
        "must_patch": must_patch,
        "patch_path": os.path.join(
            result_folder,
            rule_name,
            f"{agent_id}_{model_alias(model_name)}_{rule_name}_patch.txt",
        ),
        "trajectory_file": os.path.join(
            result_folder,
            "trajectories",
            f"{agent_id}_{model_alias(model_name)}_trajectory_{rule_name}_{timestamp}.json",
        ),
        "console_type": console_type,
        "agent_type": agent_id,
    }


def _clear_case_trajectory_dir(spec: ExperimentSpec, rule_name: str) -> None:
    trajectory_dir = os.path.join(spec.result_folder, "trajectories", rule_name)
    if os.path.isdir(trajectory_dir):
        shutil.rmtree(trajectory_dir)


def _case_trajectory_dir(spec: ExperimentSpec, rule_name: str) -> str:
    return os.path.join(spec.result_folder, "trajectories", rule_name)


def _stage_autodialect_args(runner_args: dict[str, Any], stage: str) -> dict[str, Any]:
    """Return AUTODIALECT args with stage-specific patch and trajectory paths."""
    stage_args = dict(runner_args)
    for key in ("patch_path", "trajectory_file"):
        value = stage_args.get(key)
        if not value:
            continue
        root, ext = os.path.splitext(value)
        stage_args[key] = f"{root}_{stage}{ext}"
    return stage_args


def _copy_autodialect_trajectory(
    *,
    spec: ExperimentSpec,
    rule_name: str,
    trajectory_file: str,
    native_trajectory_files: list[str],
    copied_trajectory_files: list[str],
    stage: str | None = None,
    trajectory_manifest_entries: list[dict[str, Any]] | None = None,
) -> bool:
    """Copy one AUTODIALECT trajectory file into the per-case trajectory folder."""
    if not os.path.isfile(trajectory_file):
        return False
    trajectory_dir = _case_trajectory_dir(spec, rule_name)
    os.makedirs(trajectory_dir, exist_ok=True)
    copied_path = os.path.join(trajectory_dir, os.path.basename(trajectory_file))
    if os.path.exists(copied_path):
        stem, ext = os.path.splitext(os.path.basename(trajectory_file))
        copied_path = os.path.join(trajectory_dir, f"{stem}_{len(copied_trajectory_files) + 1}{ext}")
    shutil.copyfile(trajectory_file, copied_path)
    native_trajectory_files.append(trajectory_file)
    copied_trajectory_files.append(copied_path)
    if trajectory_manifest_entries is not None:
        trajectory_manifest_entries.append(
            {
                "stage": stage or "unknown",
                "native_path": trajectory_file,
                "copied_path": copied_path,
            }
        )
    return True


def _write_trajectory_manifest(
    spec: ExperimentSpec,
    rule_name: str,
    *,
    trajectory_manifest_entries: list[dict[str, Any]],
    native_trajectory_files: list[str],
    copied_trajectory_files: list[str],
    pipeline_stages: list[dict[str, Any]],
) -> str:
    """Write one per-case trajectory index while preserving existing trajectory files."""
    if not trajectory_manifest_entries and not native_trajectory_files and not copied_trajectory_files:
        return ""

    entries = list(trajectory_manifest_entries)
    if not entries:
        for index, copied_path in enumerate(copied_trajectory_files):
            native_path = native_trajectory_files[index] if index < len(native_trajectory_files) else ""
            entries.append(
                {
                    "stage": "agent_run",
                    "native_path": native_path,
                    "copied_path": copied_path,
                }
            )

    trajectory_dir = _case_trajectory_dir(spec, rule_name)
    os.makedirs(trajectory_dir, exist_ok=True)
    manifest_file = os.path.join(trajectory_dir, "trajectory_manifest.json")
    save_json(
        manifest_file,
        {
            "schema": "autodialect_trajectory_manifest_v1",
            "rule_name": rule_name,
            "stage_count": len(pipeline_stages),
            "trajectory_files": entries,
            "native_trajectory_files": native_trajectory_files,
            "trajectory_copy_files": copied_trajectory_files,
        },
    )
    return manifest_file


@dataclass
class CaseTrajectoryCapture:
    """Mutable per-case trajectory capture state used by generation runs."""

    spec: ExperimentSpec
    rule_name: str
    native_trajectory_files: list[str] = field(default_factory=list)
    copied_trajectory_files: list[str] = field(default_factory=list)
    trajectory_manifest_entries: list[dict[str, Any]] = field(default_factory=list)
    autodialect_trajectory_copied_during_run: bool = False

    def clear_case_dir(self) -> None:
        _clear_case_trajectory_dir(self.spec, self.rule_name)

    def copy_autodialect_stage(self, trajectory_file: str, *, stage: str | None = None) -> bool:
        copied = _copy_autodialect_trajectory(
            spec=self.spec,
            rule_name=self.rule_name,
            trajectory_file=trajectory_file,
            native_trajectory_files=self.native_trajectory_files,
            copied_trajectory_files=self.copied_trajectory_files,
            stage=stage,
            trajectory_manifest_entries=self.trajectory_manifest_entries,
        )
        if copied:
            self.autodialect_trajectory_copied_during_run = True
        return copied

    def write_manifest(self, pipeline_stages: list[dict[str, Any]]) -> str:
        return _write_trajectory_manifest(
            self.spec,
            self.rule_name,
            trajectory_manifest_entries=self.trajectory_manifest_entries,
            native_trajectory_files=self.native_trajectory_files,
            copied_trajectory_files=self.copied_trajectory_files,
            pipeline_stages=pipeline_stages,
        )


# 6. Result path helpers.
def validated_result_file(result_file: str) -> str:
    root, ext = os.path.splitext(result_file)
    return f"{root}_validated{ext or '.json'}"
