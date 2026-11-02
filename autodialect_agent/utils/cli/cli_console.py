# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

"""Base CLI Console classes for AutoDialect Agent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rich.table import Table

from autodialect_agent.agent.agent_basics import AgentExecution, AgentStep, AgentStepState


AGENT_STATE_INFO = {
    AgentStepState.THINKING: ("blue", "🤔"),
    AgentStepState.CALLING_TOOL: ("yellow", "🔧"),
    AgentStepState.REFLECTING: ("magenta", "💭"),
    AgentStepState.COMPLETED: ("green", "✅"),
    AgentStepState.ERROR: ("red", "❌"),
}


@dataclass
class ConsoleStep:
    """Track whether an agent step has been printed."""

    agent_step: AgentStep
    agent_step_printed: bool = False


class CLIConsole(ABC):
    """Base class for CLI console implementations."""

    def __init__(self):
        """Initialize the CLI console."""
        self.console_step_history: dict[int, ConsoleStep] = {}
        self.agent_execution: AgentExecution | None = None

    @abstractmethod
    async def start(self):
        """Start the console display. Should be implemented by subclasses."""
        pass

    @abstractmethod
    def update_status(
        self, agent_step: AgentStep | None = None, agent_execution: AgentExecution | None = None
    ):
        """Update the console with agent status.

        Args:
            agent_step: Current agent step information
            agent_execution: Complete agent execution information
        """
        pass

    @abstractmethod
    def print_task_details(self, details: dict[str, str]):
        """Print initial task configuration details."""
        pass

    @abstractmethod
    def print(self, message: str, color: str = "blue", bold: bool = False):
        """Print a message to the console."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the console and cleanup resources."""
        pass


def generate_agent_step_table(agent_step: AgentStep) -> Table:
    """Log an agent step to the console."""
    color, emoji = AGENT_STATE_INFO.get(agent_step.state, ("white", "❓"))

    # Print the step state in a table
    table = Table(show_header=False, width=120)
    table.add_column("Step Number", style="cyan", width=15)
    table.add_column(f"{agent_step.step_number}", style="green", width=105)

    # Add status row
    table.add_row(
        "Status",
        f"[{color}]{emoji} Step {agent_step.step_number}: {agent_step.state.value.title()}[/{color}]",
    )

    # Add LLM response row
    if agent_step.llm_response and agent_step.llm_response.content:
        table.add_row("LLM Response", f"💬 {agent_step.llm_response.content}")

    # Add tool calls row
    if agent_step.tool_calls:
        tool_names = [f"[cyan]{call.name}[/cyan]" for call in agent_step.tool_calls]
        table.add_row("Tools", f"🔧 {', '.join(tool_names)}")

        for tool_call in agent_step.tool_calls:
            # Build a tool call table with tool name, arguments and result
            tool_call_table = Table(show_header=False, width=100)
            tool_call_table.add_column("Arguments", style="green", width=50)
            tool_call_table.add_column("Result", style="green", width=50)
            tool_result_str = ""
            for tool_result in agent_step.tool_results or []:
                if tool_result.call_id == tool_call.call_id:
                    tool_result_str = tool_result.result or ""
                    break
            tool_call_table.add_row(f"{tool_call.arguments}", f"{tool_result_str}")
            table.add_row(tool_call.name, tool_call_table)

    # Add reflection row
    if agent_step.reflection:
        table.add_row("Reflection", f"💭 {agent_step.reflection}")

    # Add error row
    if agent_step.error:
        table.add_row("Error", f"❌ {agent_step.error}")

    return table
