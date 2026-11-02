"""Shared low-level utilities used across AutoDialect scripts.

This module is for helpers that are not tied to one pipeline stage:
1. SQL text comparison and compact diff helpers.
2. Lightweight metadata cleanup shared by prompt, memory, and validation code.
3. SQLGlot worker/transpile wrappers reused by generation and evaluation.
4. File/worktree helpers for copying rule trees and collecting git diffs.

Stage orchestration, parser gates, memory audit aggregation, and handoff payloads
belong in code_utils.execution_runtime instead.
"""

from __future__ import annotations

import atexit
import difflib
import json
import os
import re
import selectors
import shutil
import subprocess
import time
from typing import Any

from code_utils.constants import (
    AUTODIALECT_SQLGLOT_WORKER_INIT_TIMEOUT_S,
    AUTODIALECT_SQLGLOT_WORKER_TIMEOUT_S,
    AUTODIALECT_PYTHON_BIN,
)

DEFAULT_SQLGLOT_WORKER_TIMEOUT_S = AUTODIALECT_SQLGLOT_WORKER_TIMEOUT_S
DEFAULT_SQLGLOT_WORKER_INIT_TIMEOUT_S = AUTODIALECT_SQLGLOT_WORKER_INIT_TIMEOUT_S
MAX_SQLGLOT_WORKER_STDERR_BYTES = 20000
_WORKER_CODE = r"""
import contextlib
import json
import os
import sys

package_root = sys.argv[1]
syspath = os.environ.get("AUTODIALECT_SYSPATH", "")
syspath_list = [p for p in syspath.split(os.pathsep) if p]
sys.path = [package_root] + syspath_list

import sqlglot

def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


_emit({"ok": True, "sqlglot_file": getattr(sqlglot, "__file__", None), "pid": os.getpid()})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    req_id = req.get("id")
    sql = req.get("sql")
    read = req.get("read")
    write = req.get("write")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            out = sqlglot.transpile(sql=sql, read=read, write=write)[0]
        _emit({"ok": True, "id": req_id, "result": out})
    except Exception as e:
        _emit({"ok": False, "id": req_id, "error": str(e), "error_type": type(e).__name__})
"""


# 1. SQL text comparison and compact diff helpers.
_SQL_DIFF_TOKEN_RE = re.compile(
    r"`[^`]*`|\"[^\"]*\"|'(?:''|[^'])*'|[A-Za-z_][A-Za-z0-9_]*|"
    r"\d+\.\d+|\d+|<=|>=|<>|!=|::|[^\s]"
)


def sql_key(sql: str) -> str:
    """Normalize SQL text for benchmark-style equality checks."""
    return re.sub(r"\s+", "", sql.strip().rstrip(";")).lower()


def _raw_diff_tokens(sql: str) -> list[str]:
    return _SQL_DIFF_TOKEN_RE.findall(sql or "")


def _comparison_token_records(sql: str) -> list[tuple[str, str, int, int]]:
    """Map original tokens onto the whitespace-free benchmark character key."""
    normalized = (sql or "").strip().rstrip(";")
    display_tokens = _SQL_DIFF_TOKEN_RE.findall(normalized)
    records: list[tuple[str, str, int, int]] = []
    cursor = 0
    for token in display_tokens:
        key = re.sub(r"\s+", "", token).lower()
        records.append((key, token, cursor, cursor + len(key)))
        cursor += len(key)
    return records


def _token_diff_lines(actual_tokens: list[str], expected_tokens: list[str], max_lines: int) -> list[str]:
    lines = list(difflib.unified_diff(actual_tokens, expected_tokens, lineterm="", n=2))
    return [line for line in lines if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))][:max_lines]


def _record_indexes_overlapping(
    records: list[tuple[str, str, int, int]],
    start: int,
    end: int,
) -> list[int]:
    if start >= end:
        return []
    return [
        index
        for index, (_key, _display, token_start, token_end) in enumerate(records)
        if token_end > start and token_start < end
    ]


def _comparison_char_opcodes(actual_key: str, expected_key: str) -> list[tuple[str, int, int, int, int]]:
    """Return exact changed opcodes after trimming equal outer character runs."""
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


def _actionable_token_changes(actual: str, expected: str) -> list[tuple[str, str]]:
    """Locate benchmark-visible character changes and render their source tokens."""
    actual_records = _comparison_token_records(actual)
    expected_records = _comparison_token_records(expected)
    actual_key = "".join(record[0] for record in actual_records)
    expected_key = "".join(record[0] for record in expected_records)
    changes: list[tuple[str, str]] = []
    seen_actual: set[int] = set()
    seen_expected: set[int] = set()
    for _tag, i1, i2, j1, j2 in _comparison_char_opcodes(actual_key, expected_key):
        for index in _record_indexes_overlapping(actual_records, i1, i2):
            if index not in seen_actual:
                changes.append(("-", actual_records[index][1]))
                seen_actual.add(index)
        for index in _record_indexes_overlapping(expected_records, j1, j2):
            if index not in seen_expected:
                changes.append(("+", expected_records[index][1]))
                seen_expected.add(index)
    return changes


def _comparison_token_delta(actual: str, expected: str) -> tuple[list[str], list[str]]:
    actual_only: list[str] = []
    expected_only: list[str] = []
    for prefix, token in _actionable_token_changes(actual, expected):
        (actual_only if prefix == "-" else expected_only).append(token)
    return actual_only, expected_only


def _actionable_token_diff_lines(actual: str, expected: str, max_lines: int) -> list[str]:
    return [f"{prefix}{token}" for prefix, token in _actionable_token_changes(actual, expected)][:max_lines]


def raw_diff_summary(actual: str, expected: str, max_lines: int = 12) -> list[str]:
    """Return the legacy case-sensitive diff for non-model-facing debugging."""
    return _token_diff_lines(_raw_diff_tokens(actual), _raw_diff_tokens(expected), max_lines)


def diff_summary(actual: str, expected: str, max_lines: int = 12) -> list[str]:
    """Return an actionable diff aligned with ``sql_key`` pass/fail semantics.

    Case, whitespace, and trailing-semicolon changes cannot become repair
    instructions. The early equality check is authoritative; token and
    character diffs are diagnostics only.
    """
    if max_lines <= 0 or sql_key(actual) == sql_key(expected):
        return []
    lines = _actionable_token_diff_lines(actual, expected, max_lines)
    if lines:
        return lines

    # A tokenization edge case must not hide a benchmark-visible mismatch.
    return _token_diff_lines(list(sql_key(actual)), list(sql_key(expected)), max_lines)


def model_facing_diagnostics(value: Any) -> Any:
    """Project validation data into prompt-safe actionable diagnostics.

    Legacy payloads may contain a case-sensitive ``diff_summary``. Recompute it
    whenever the corresponding SQL pair is available, and never expose an
    explicitly stored raw diff to a model-facing prompt.
    """
    if isinstance(value, list):
        return [model_facing_diagnostics(item) for item in value]
    if not isinstance(value, dict):
        return value

    projected = {
        key: model_facing_diagnostics(item)
        for key, item in value.items()
        if key != "raw_diff_summary"
    }
    if "diff_summary" in projected and not projected.get("error_type"):
        sql_pairs = (
            ("actual_sql", "expected_sql"),
            ("actual_node_sql", "expected_node_sql"),
            ("current_output", "expected_output"),
        )
        for actual_key, expected_key in sql_pairs:
            actual = projected.get(actual_key)
            expected = projected.get(expected_key)
            if isinstance(actual, str) and isinstance(expected, str):
                actionable = diff_summary(actual, expected)
                if actionable:
                    projected["diff_summary"] = actionable
                else:
                    projected.pop("diff_summary", None)
                break

    actual_node = projected.get("actual_node_sql")
    expected_node = projected.get("expected_node_sql")
    if isinstance(actual_node, str) and isinstance(expected_node, str):
        actual_only, expected_only = _comparison_token_delta(actual_node, expected_node)
        for key, tokens in (
            ("actual_only", actual_only),
            ("actual_changed_tokens", actual_only),
            ("expected_only", expected_only),
            ("expected_changed_tokens", expected_only),
        ):
            if key in projected:
                projected[key] = tokens

    if (
        isinstance(projected.get("transition_summary"), dict)
        and not projected.get("error_type")
        and isinstance(projected.get("source_sql"), str)
        and isinstance(projected.get("actual_sql"), str)
        and isinstance(projected.get("expected_sql"), str)
    ):
        projected["transition_summary"] = transition_summary(
            projected["source_sql"],
            projected["actual_sql"],
            projected["expected_sql"],
        )
    return projected


def transition_summary(source_sql: str, actual_sql: str, expected_sql: str, max_items: int = 8) -> dict[str, list[str]]:
    """Summarize likely rule-defining token transitions without using identifiers."""
    token_pattern = r"`[^`]*`|\"[^\"]*\"|'[^']*'|[A-Za-z_][A-Za-z0-9_]*|<=|>=|<>|!=|::|[+\-*/%]|\d+\.\d+|\d+"

    def tokens(sql: str) -> list[str]:
        raw_tokens = re.findall(token_pattern, sql or "")
        kept = []
        for token in raw_tokens:
            lower = token.lower()
            if token.startswith(("'", '"', "`")):
                if "%" in token or "yyyy" in lower or "mm" in lower or "dd" in lower:
                    kept.append(re.sub(r"\s+", "", lower))
                continue
            if lower.startswith("column_") or lower.startswith("table_"):
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", token):
                continue
            kept.append(lower)
        return kept

    source_tokens = tokens(source_sql)
    actual_tokens = tokens(actual_sql)
    expected_tokens = tokens(expected_sql)
    return {
        "source_only": [token for token in source_tokens if token not in expected_tokens][:max_items],
        "actual_only": [token for token in actual_tokens if token not in expected_tokens][:max_items],
        "expected_only": [token for token in expected_tokens if token not in actual_tokens][:max_items],
    }


# 2. Lightweight metadata cleanup helpers.
def compact_output_fields(value: Any) -> Any:
    """Recursively omit absent containers while preserving False and zero."""
    if isinstance(value, dict):
        compact = {key: compact_output_fields(item) for key, item in value.items()}
        return {
            key: item
            for key, item in compact.items()
            if isinstance(item, bool) or item not in (None, [], {})
        }
    if isinstance(value, list):
        compact = [compact_output_fields(item) for item in value]
        return [
            item
            for item in compact
            if isinstance(item, bool) or item not in (None, [], {})
        ]
    return value


def compact_top_level_fields(value: dict[str, Any]) -> dict[str, Any]:
    """Omit absent top-level fields without rewriting nested protocol objects."""
    return {
        key: item
        for key, item in value.items()
        if isinstance(item, bool) or item not in (None, [], {})
    }


def scrub_legacy_family_text(value: str) -> str:
    """Remove legacy family hints from free-form text."""
    scrubbed = re.sub(r"(?im)^family_hint=.*$", "", value)
    scrubbed = re.sub(r"\bfamily_hint=[^\s,;]+", "", scrubbed)
    scrubbed = re.sub(r"\bfamily:[^\s,;]+", "", scrubbed)
    return scrubbed.strip()


def is_legacy_family_key(key: Any) -> bool:
    """Return true for metadata keys that encode deprecated family labels."""
    return "family" in str(key).lower()


def strip_legacy_family_metadata(value: Any) -> Any:
    """Remove legacy family metadata recursively from prompt/audit payloads."""
    if isinstance(value, str):
        return scrub_legacy_family_text(value)
    if isinstance(value, list):
        return [
            item
            for item in (strip_legacy_family_metadata(item) for item in value)
            if item not in (None, [], {})
        ]
    if isinstance(value, dict):
        return {
            key: item
            for key, raw_item in value.items()
            if not is_legacy_family_key(key)
            for item in [strip_legacy_family_metadata(raw_item)]
            if item not in (None, [], {})
        }
    return value


# 3. SQLGlot worker/transpile wrappers.
class _SqlglotWorker:
    def __init__(self, package_root: str, sys_path_backup):
        self.package_root = package_root
        self._next_id = 1
        self._closed = False
        self._stdout_buffer = b""
        self._stderr_buffer = bytearray()
        env = os.environ.copy()
        env.setdefault("PYTHONHASHSEED", "0")
        env["AUTODIALECT_SYSPATH"] = os.pathsep.join(sys_path_backup)
        self.proc = subprocess.Popen(
            [AUTODIALECT_PYTHON_BIN, "-u", "-c", _WORKER_CODE, package_root],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._selector = selectors.DefaultSelector()
        if self.proc.stdout:
            self._selector.register(self.proc.stdout, selectors.EVENT_READ, "stdout")
        if self.proc.stderr:
            self._selector.register(self.proc.stderr, selectors.EVENT_READ, "stderr")
        atexit.register(self.close)
        try:
            init = self._read_json_response(
                timeout_s=DEFAULT_SQLGLOT_WORKER_INIT_TIMEOUT_S,
                stage="init",
            )
            if not init.get("ok"):
                raise RuntimeError(f"worker init failed: {init}; stderr: {self._stderr_text()}")
            sqlglot_file = init.get("sqlglot_file") or ""
            if package_root not in sqlglot_file:
                raise RuntimeError(f"worker loaded unexpected sqlglot: {sqlglot_file}")
        except Exception:
            self.close()
            raise

    def _kill_worker(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass

    def _append_stderr(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._stderr_buffer.extend(chunk)
        if len(self._stderr_buffer) > MAX_SQLGLOT_WORKER_STDERR_BYTES:
            del self._stderr_buffer[:-MAX_SQLGLOT_WORKER_STDERR_BYTES]

    def _stderr_text(self) -> str:
        return bytes(self._stderr_buffer).decode("utf-8", errors="replace").strip()

    def _read_ready_streams(self, timeout_s: float) -> list[dict]:
        responses: list[dict] = []
        try:
            events = self._selector.select(timeout=max(0.0, timeout_s))
        except (ValueError, OSError):
            return responses

        for key, _mask in events:
            stream = key.fileobj
            stream_name = key.data
            try:
                chunk = os.read(stream.fileno(), 4096)
            except BlockingIOError:
                continue
            except OSError:
                chunk = b""

            if not chunk:
                try:
                    self._selector.unregister(stream)
                except Exception:
                    pass
                continue

            if stream_name == "stderr":
                self._append_stderr(chunk)
                continue

            self._stdout_buffer += chunk
            while b"\n" in self._stdout_buffer:
                raw_line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    responses.append(json.loads(raw_line.decode("utf-8", errors="replace")))
                except json.JSONDecodeError:
                    self._append_stderr(b"invalid worker stdout: " + raw_line[:500] + b"\n")
        return responses

    def _read_json_response(
        self,
        *,
        timeout_s: float,
        stage: str,
        expected_id: int | None = None,
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            responses = self._read_ready_streams(min(remaining, 0.25))
            for resp in responses:
                if expected_id is None or resp.get("id") == expected_id:
                    return resp

            if self.proc.poll() is not None:
                for resp in self._read_ready_streams(0.0):
                    if expected_id is None or resp.get("id") == expected_id:
                        return resp
                stderr = self._stderr_text()
                raise RuntimeError(f"worker exited unexpectedly during {stage}, stderr: {stderr}")

        self._kill_worker()
        stderr = self._stderr_text()
        raise TimeoutError(f"worker {stage} timed out after {timeout_s:.1f}s, stderr: {stderr}")

    def transpile(self, sql: str, read: str, write: str, timeout_s: float | None = None) -> str:
        timeout_s = (
            DEFAULT_SQLGLOT_WORKER_TIMEOUT_S
            if timeout_s is None
            else timeout_s
        )
        req_id = self._next_id
        self._next_id += 1
        if not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("worker streams not available")
        payload = json.dumps({"id": req_id, "sql": sql, "read": read, "write": write}) + "\n"
        try:
            self.proc.stdin.write(payload.encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"worker stdin unavailable: {exc}; stderr: {self._stderr_text()}") from exc

        resp = self._read_json_response(
            timeout_s=timeout_s,
            stage="transpile",
            expected_id=req_id,
        )
        if resp.get("ok"):
            return resp.get("result", "")
        raise RuntimeError(f"{resp.get('error_type')}: {resp.get('error')}")

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        try:
            atexit.unregister(self.close)
        except Exception:
            pass

        proc = getattr(self, "proc", None)
        if not proc:
            return

        try:
            self._selector.close()
        except Exception:
            pass

        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._kill_worker()
            except Exception:
                pass
        else:
            try:
                proc.wait(timeout=0)
            except Exception:
                pass

        for stream_name in ("stdout", "stderr"):
            stream = getattr(proc, stream_name, None)
            try:
                if stream:
                    stream.close()
            except Exception:
                pass




# 4. File/worktree helpers.
def prepare_case_worktree(src_dir, dst_dir):
    """Create a clean per-case copy of the translation-engine checkout."""

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    return True


def get_git_head(compile_folder):
    """Return the current git HEAD for a copied worktree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=compile_folder,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_name_only(args, cwd):
    """Read git path output as NUL-delimited names to preserve unusual filenames."""
    output = subprocess.check_output(args + ["-z"], cwd=cwd)
    return [os.fsdecode(path) for path in output.split(b"\0") if path]


def get_git_diff(compile_folder, result_folder, base_ref=None):
    """Get the git diff of the project."""
    if not os.path.isdir(compile_folder):
        return "", []

    try:
        _ = subprocess.check_output(["git", "add", "."], cwd=compile_folder).decode()
        staged_changes = subprocess.check_output(
            ["git", "--no-pager", "diff", "--staged"],
            cwd=compile_folder,
        ).decode()

        staged_files = _git_name_only(
            ["git", "--no-pager", "diff", "--staged", "--name-only"],
            cwd=compile_folder,
        )

        committed_changes = ""
        committed_files = []
        if base_ref:
            try:
                committed_changes = subprocess.check_output(
                    ["git", "--no-pager", "diff", f"{base_ref}..HEAD"],
                    cwd=compile_folder,
                ).decode()
                committed_files = _git_name_only(
                    ["git", "--no-pager", "diff", f"{base_ref}..HEAD", "--name-only"],
                    cwd=compile_folder,
                )
            except subprocess.CalledProcessError:
                committed_changes = ""
                committed_files = []

        code_changes = "\n".join(part for part in (committed_changes, staged_changes) if part)
        file_list = list(dict.fromkeys(committed_files + staged_files))
        for file in file_list:
            try:
                src = os.path.join(compile_folder, file)
                dst = os.path.join(result_folder, file)
                if not os.path.isfile(src):
                    continue
                dst_parent = os.path.dirname(dst)
                if dst_parent and not os.path.exists(dst_parent):
                    os.makedirs(dst_parent)

                shutil.copyfile(src, dst)
            except (OSError, ValueError):
                continue

    except (subprocess.CalledProcessError, FileNotFoundError):
        code_changes = ""
        file_list = []

    return code_changes, file_list
