# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""Base Agent class for LLM-based agents.

1. Own the generic step loop shared by AutoDialect agents, including tool execution,
   retry prompts, finalization, and trajectory recording.
2. Initialize configured tools while tolerating optional local dependencies
   that are not required by the AutoDialect path.
3. Expose overridable completion hooks used by AutoDialectAgent to layer
   AutoDialect-specific validation on top of the base loop.
"""

from abc import ABC, abstractmethod

from autodialect_agent.agent.agent_basics import AgentExecution, AgentState, AgentStep, AgentStepState
from autodialect_agent.prompt.agent_prompt import (
    NO_TOOL_CALL_RETRY_PROMPT,
    TASK_INCOMPLETE_RETRY_PROMPT,
)
from autodialect_agent.tools import tools_registry
from autodialect_agent.tools.base import Tool, ToolCall, ToolExecutor, ToolResult
from autodialect_agent.utils.cli import CLIConsole
from autodialect_agent.utils.config import AgentConfig, ModelConfig
from autodialect_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from autodialect_agent.utils.llm_clients.llm_client import LLMClient
from autodialect_agent.utils.trajectory_recorder import TrajectoryRecorder

def available_configured_tool_names(tool_names: list[str]) -> list[str]:
    """Validate and select the configured AutoDialect tools."""
    unknown = [
        tool_name
        for tool_name in tool_names
        if tool_name not in tools_registry
    ]
    if unknown:
        raise KeyError(unknown[0])
    return [tool_name for tool_name in tool_names if tool_name in tools_registry]


class BaseAgent(ABC):
    """Base class for LLM-based agents."""

    def __init__(self, agent_config: AgentConfig):
        """Initialize the agent.

        Args:
            agent_config: Configuration object containing model parameters and other settings.
        """
        self._llm_client = LLMClient(agent_config.model)
        self._model_config = agent_config.model
        self._max_steps = agent_config.max_steps
        self._initial_messages: list[LLMMessage] = []
        self._task: str = ""
        self._tools: list[Tool] = [
            tools_registry[tool_name](model_provider=self._model_config.model_provider.provider)
            for tool_name in available_configured_tool_names(agent_config.tools)
        ]
        self._tool_caller: ToolExecutor = ToolExecutor([])
        self._cli_console: CLIConsole | None = None

        # Trajectory recorder
        self._trajectory_recorder: TrajectoryRecorder | None = None

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @property
    def trajectory_recorder(self) -> TrajectoryRecorder | None:
        """Get the trajectory recorder for this agent."""
        return self._trajectory_recorder

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """Set the trajectory recorder for this agent."""
        self._trajectory_recorder = recorder
        # Also set it on the LLM client
        self._llm_client.set_trajectory_recorder(recorder)

    @property
    def cli_console(self) -> CLIConsole | None:
        """Get the CLI console for this agent."""
        return self._cli_console

    def set_cli_console(self, cli_console: CLIConsole | None) -> None:
        """Set the CLI console for this agent."""
        self._cli_console = cli_console

    @property
    def tools(self) -> list[Tool]:
        """Get the tools available to this agent."""
        return self._tools

    @property
    def task(self) -> str:
        """Get the current task of the agent."""
        return self._task

    @task.setter
    def task(self, value: str):
        """Set the current task of the agent."""
        self._task = value

    @property
    def model_config(self) -> ModelConfig:
        """Get the model config for the agent."""
        return self._model_config

    @property
    def max_steps(self) -> int:
        """Get the maximum number of steps for the agent."""
        return self._max_steps

    @abstractmethod
    def new_task(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        """Create a new task."""
        pass

    async def execute_task(self) -> AgentExecution:
        """Execute a task using the agent."""
        import time

        start_time = time.time()
        execution = AgentExecution(task=self._task, steps=[])
        step: AgentStep | None = None

        try:
            messages = self._initial_messages
            step_number = 1
            execution.agent_state = AgentState.RUNNING

            while step_number <= self._max_steps:
                step = AgentStep(step_number=step_number, state=AgentStepState.THINKING)
                try:
                    messages = await self._run_llm_step(step, messages, execution)
                    await self._finalize_step(
                        step, messages, execution
                    )  # record trajectory for this step and update the CLI console
                    if execution.agent_state == AgentState.COMPLETED:
                        break
                    step_number += 1
                except Exception as error:
                    execution.agent_state = AgentState.ERROR
                    execution.final_result = (
                        f"Agent execution failed at step {step_number}: "
                        f"{type(error).__name__}: {error}"
                    )
                    step.state = AgentStepState.ERROR
                    step.error = str(error)
                    await self._finalize_step(step, messages, execution)
                    break
            if step_number > self._max_steps and not execution.success:
                step_limit_result = self._finalize_after_step_limit()
                if step_limit_result is not None:
                    execution.final_result = step_limit_result
                    execution.agent_state = AgentState.COMPLETED
                    execution.success = True
                else:
                    execution.final_result = "Task execution exceeded maximum steps without completion."
                    execution.agent_state = AgentState.ERROR

        except Exception as e:
            execution.final_result = f"Agent execution failed: {str(e)}"

        # Ensure tool resources are released whether an exception occurs or not.
        await self._close_tools()

        execution.execution_time = time.time() - start_time

        self._update_cli_console(step, execution)
        return execution

    async def _close_tools(self):
        """Release tool resources, mainly about BashTool object."""
        if self._tool_caller:
            # Ensure all tool resources are properly released.
            res = await self._tool_caller.close_tools()
            return res

    async def _run_llm_step(
        self, step: "AgentStep", messages: list["LLMMessage"], execution: "AgentExecution"
    ) -> list["LLMMessage"]:
        # Display thinking state
        step.state = AgentStepState.THINKING
        self._update_cli_console(step, execution)
        # Get LLM response
        llm_response = self._llm_client.chat(messages, self._model_config, self._tools)
        step.llm_response = llm_response

        # Display step with LLM response
        self._update_cli_console(step, execution)

        # Update token usage
        self._update_llm_usage(llm_response, execution)

        if self.llm_indicates_task_completed(llm_response):
            if self._is_task_completed(llm_response):
                execution.agent_state = AgentState.COMPLETED
                execution.final_result = llm_response.content
                execution.success = True
                return messages
            else:
                execution.agent_state = AgentState.RUNNING
                return [LLMMessage(role="user", content=self.task_incomplete_message())]
        else:
            tool_calls = llm_response.tool_calls
            return await self._tool_call_handler(tool_calls, step)

    async def _finalize_step(
        self, step: "AgentStep", messages: list["LLMMessage"], execution: "AgentExecution"
    ) -> None:
        step.state = AgentStepState.COMPLETED
        self._record_handler(step, messages)
        self._update_cli_console(step, execution)
        execution.steps.append(step)

    def reflect_on_result(self, tool_results: list[ToolResult]) -> str | None:
        """Reflect on tool execution result. Override for custom reflection logic."""
        if len(tool_results) == 0:
            return None

        reflection = "\n".join(
            f"The tool execution failed with error: {tool_result.error}. Consider trying a different approach or fixing the parameters."
            for tool_result in tool_results
            if not tool_result.success
        )

        return reflection

    def llm_indicates_task_completed(self, llm_response: LLMResponse) -> bool:
        """Check if the LLM indicates that the task is completed. Override for custom logic."""
        completion_indicators = [
            "task completed",
            "task finished",
            "done",
            "completed successfully",
            "finished successfully",
        ]

        response_lower = llm_response.content.lower()
        return any(indicator in response_lower for indicator in completion_indicators)

    def _is_task_completed(self, llm_response: LLMResponse) -> bool:  # pyright: ignore[reportUnusedParameter]
        """Check if the task is completed based on the response. Override for custom logic."""
        return True

    def task_incomplete_message(self) -> str:
        """Return a message indicating that the task is incomplete. Override for custom logic."""
        return TASK_INCOMPLETE_RETRY_PROMPT

    def _finalize_after_step_limit(self) -> str | None:
        """Return a success message when final state validation passes.

        Generic agents have no authoritative state validator, so they retain
        the historical maximum-step failure behavior. AutoDialect agents
        override this hook to validate the current worktree before failing.
        """
        return None

    def _update_cli_console(
        self, step: AgentStep | None = None, agent_execution: AgentExecution | None = None
    ) -> None:
        if self.cli_console:
            self.cli_console.update_status(step, agent_execution)

    def _update_llm_usage(self, llm_response: LLMResponse, execution: AgentExecution):
        if not llm_response.usage:
            return
        # Initialize token usage from the first response, then accumulate it.
        if not execution.total_tokens:
            execution.total_tokens = llm_response.usage
        else:
            execution.total_tokens += llm_response.usage

    def _record_handler(self, step: AgentStep, messages: list[LLMMessage]) -> None:
        if self.trajectory_recorder:
            self.trajectory_recorder.record_agent_step(
                step_number=step.step_number,
                state=step.state.value,
                llm_messages=messages,
                llm_response=step.llm_response,
                tool_calls=step.tool_calls,
                tool_results=step.tool_results,
                reflection=step.reflection,
                error=step.error,
            )

    async def _tool_call_handler(
        self, tool_calls: list[ToolCall] | None, step: AgentStep
    ) -> list[LLMMessage]:
        messages: list[LLMMessage] = []
        if not tool_calls or len(tool_calls) <= 0:
            messages = [
                LLMMessage(
                    role="user",
                    content=NO_TOOL_CALL_RETRY_PROMPT,
                )
            ]
            return messages

        step.state = AgentStepState.CALLING_TOOL
        step.tool_calls = tool_calls
        self._update_cli_console(step)

        # AutoDialect tools share case state. In particular, every bash call
        # uses the same persistent shell, and edit calls mutate one worktree.
        # Preserve model-issued order even when the provider returns a batch.
        tool_results = await self._tool_caller.sequential_tool_call(tool_calls)
        step.tool_results = tool_results
        self._update_cli_console(step)
        for tool_result in tool_results:
            # Add tool result to conversation
            message = LLMMessage(role="user", tool_result=tool_result)
            messages.append(message)

        reflection = self.reflect_on_result(tool_results)
        if reflection:
            step.state = AgentStepState.REFLECTING
            step.reflection = reflection

            # Display reflection
            self._update_cli_console(step)

            messages.append(LLMMessage(role="assistant", content=reflection))

        return messages
