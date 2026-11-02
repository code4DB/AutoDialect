# Copyright (c) 2023 Anthropic
# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT
#
# This file has been modified for AutoDialect.
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/anthropics/anthropic-quickstarts/blob/main/LICENSE
#
# This modified file is released under the same license.

"""Bash tool used by AutoDialect agent runs.

1. Maintain a persistent shell session for model-issued commands.
2. Route bare python/python3 commands in the Unix shell through the exact
   AutoDialect Python executable configured for the experiment runtime.
3. Reject unsafe double-quoted python -c payloads that would let the shell
   expand SQL backticks before Python sees them.
4. Block git commands in the tool path to keep benchmark patches scoped to the
   provided task context.
"""

import asyncio
import contextlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import override

# ruff: noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path[:] = [str(ROOT_DIR), *[path for path in sys.path if path != str(ROOT_DIR)]]

from code_utils.constants import AUTODIALECT_PYTHON_BIN

from autodialect_agent.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter


def _double_quoted_python_payloads(command: str) -> list[str]:
    """Return payloads from python -c "..." command segments."""

    payloads: list[str] = []
    pattern = re.compile(r"(?:^|[\s;&|])(?:\S*/)?python(?:3(?:\.\d+)?)?\s+-c\s*\"")
    for match in pattern.finditer(command):
        quote_start = match.end() - 1
        i = quote_start + 1
        escaped = False
        while i < len(command):
            char = command[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                payloads.append(command[quote_start + 1:i])
                break
            i += 1
    return payloads


def _unsafe_python_c_error(command: str) -> str | None:
    for payload in _double_quoted_python_payloads(command):
        if "`" in payload or "$(" in payload:
            return (
                "Unsafe shell quoting detected: a double-quoted `python -c` payload contains "
                "backticks or command substitution syntax. Shell will evaluate those before "
                "Python starts, which corrupts SQL strings. Use a quoted heredoc instead:\n"
                "python <<'PY'\n"
                "# Python code here; SQL backticks are safe in this block.\n"
                "PY"
            )
    return None


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _timed_out: bool

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = ",,,,bash-command-exit-__ERROR_CODE__-banner,,,,"  # `__ERROR_CODE__` will be replaced by `$?` or `!errorlevel!` later

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False
        self._process: asyncio.subprocess.Process | None = None

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    async def start(self) -> None:
        if self._started:
            return

        # Windows compatibility: os.setsid not available
        env = os.environ.copy()
        python_bin = shutil.which(AUTODIALECT_PYTHON_BIN, path=env.get("PATH")) or AUTODIALECT_PYTHON_BIN
        env["AUTODIALECT_TOOL_PYTHON_BIN"] = python_bin
        if os.path.isabs(python_bin):
            python_bin_dir = os.path.dirname(python_bin)
            existing_path = env.get("PATH", "")
            env["PATH"] = (
                f"{python_bin_dir}{os.pathsep}{existing_path}" if existing_path else python_bin_dir
            )

        if os.name != "nt":  # Unix-like systems
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,
                env=env,
            )
        else:
            self._process = await asyncio.create_subprocess_shell(
                "cmd.exe /v:on",  # enable delayed expansion to allow `echo !errorlevel!`
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

        if os.name != "nt":
            assert self._process.stdin
            self._process.stdin.write(
                b'python() { "$AUTODIALECT_TOOL_PYTHON_BIN" "$@"; }\n'
                b'python3() { "$AUTODIALECT_TOOL_PYTHON_BIN" "$@"; }\n'
            )
            await self._process.stdin.drain()

        self._started = True

    async def stop(self, *, force: bool = False) -> None:
        """Terminate the bash shell."""
        if not self._started:
            return
        if self._process is None:
            self._started = False
            return
        if self._process.returncode is not None:
            self._started = False
            return

        if force:
            if os.name != "nt":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._process.pid, 9)
            else:
                self._process.kill()
        else:
            self._process.terminate()
        if self._process.stdin and not self._process.stdin.is_closing():
            self._process.stdin.close()

        # Wait until the process has truly terminated.
        try:
            await asyncio.wait_for(self._process.communicate(), timeout=5 if not force else 1)
        except asyncio.TimeoutError:
            if os.name != "nt":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._process.pid, 9)
            else:
                self._process.kill()
            await self._process.communicate()
        finally:
            self._started = False

    async def run(self, command: str) -> ToolExecResult:
        """Execute a command in the bash shell."""
        if not self._started or self._process is None:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolExecResult(
                error=f"bash has exited with returncode {self._process.returncode}. tool must be restarted.",
                error_code=-1,
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        error_code = 0

        sentinel_before, pivot, sentinel_after = self._sentinel.partition("__ERROR_CODE__")
        assert pivot == "__ERROR_CODE__"

        errcode_retriever = "!errorlevel!" if os.name == "nt" else "$?"
        command_sep = "&" if os.name == "nt" else ";"

        # send command to the process
        self._process.stdin.write(
            b"(\n"
            + command.encode()
            + f"\n){command_sep} echo {self._sentinel.replace('__ERROR_CODE__', errcode_retriever)}\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output: str = self._process.stdout._buffer.decode()  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
                    if sentinel_before in output:
                        # strip the sentinel from output
                        output, pivot, exit_banner = output.rpartition(sentinel_before)
                        assert pivot

                        # get error code inside banner
                        error_code_str, pivot, _ = exit_banner.partition(sentinel_after)
                        if not pivot or not error_code_str.isdecimal():
                            continue

                        error_code = int(error_code_str)
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            output = output[:-1]  # pyright: ignore[reportUnknownVariableType]

        error: str = self._process.stderr._buffer.decode()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        if error.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            error = error[:-1]  # pyright: ignore[reportUnknownVariableType]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        return ToolExecResult(output=output, error=error, error_code=error_code)  # pyright: ignore[reportUnknownArgumentType]


class BashTool(Tool):
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)
        self._session: _BashSession | None = None

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "bash"

    @override
    def get_description(self) -> str:
        return """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        restart_required = self.model_provider == "openai"

        return [
            ToolParameter(
                name="command",
                type="string",
                description="The bash command to run.",
                required=True,
            ),
            ToolParameter(
                name="restart",
                type="boolean",
                description="Set to true to restart the bash session.",
                required=restart_required,
            ),
        ]

    async def _restart_session(self, *, force: bool = False) -> None:
        previous_session = self._session
        if previous_session:
            await previous_session.stop(force=force)
        self._session = _BashSession()
        await self._session.start()

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        if arguments.get("restart"):
            await self._restart_session()

            return ToolExecResult(output="tool has been restarted.")

        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )
        unsafe_error = _unsafe_python_c_error(command)
        if unsafe_error:
            return ToolExecResult(error=unsafe_error, error_code=-1)

        if self._session is None:
            try:
                self._session = _BashSession()
                await self._session.start()
            except Exception as e:
                return ToolExecResult(error=f"Error starting bash session: {e}", error_code=-1)
        elif self._session.timed_out:
            try:
                await self._restart_session(force=True)
            except Exception as e:
                return ToolExecResult(
                    error=f"Error restarting timed-out bash session: {e}",
                    error_code=-1,
                )

        if "git " in command:
            return ToolExecResult(
                output="The `git` command is not allowed in this project."
            )
        if "rm " in command:
            return ToolExecResult(
                output="The `rm` command is not allowed in this project."
            )
        session = self._session
        try:
            return await session.run(command)
        except Exception as e:
            error = f"Error running bash command: {e}"
            if session.timed_out:
                try:
                    await self._restart_session(force=True)
                    error += " The timed-out session was restarted automatically."
                except Exception as restart_error:
                    error += f" Automatic session restart failed: {restart_error}"
            return ToolExecResult(error=error, error_code=-1)

    @override
    async def close(self):
        """Properly close self._process."""
        if self._session:
            ret = await self._session.stop()
            self._session = None
            return ret
