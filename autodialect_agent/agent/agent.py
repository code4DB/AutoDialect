import asyncio
from enum import Enum

from autodialect_agent.utils.cli.cli_console import CLIConsole
from autodialect_agent.utils.config import AgentConfig, Config
from autodialect_agent.utils.trajectory_recorder import TrajectoryRecorder


class AgentType(Enum):
    AutoDialectAgent = "autodialect_agent"


class Agent:
    def __init__(
        self,
        agent_type: AgentType | str,
        config: Config,
        trajectory_file: str | None = None,
        cli_console: CLIConsole | None = None,
    ):
        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type)
        self.agent_type: AgentType = agent_type

        # Set up trajectory recording
        if trajectory_file is not None:
            self.trajectory_file: str = trajectory_file
            self.trajectory_recorder: TrajectoryRecorder = TrajectoryRecorder(trajectory_file)
        else:
            # Auto-generate trajectory file path
            self.trajectory_recorder = TrajectoryRecorder()
            self.trajectory_file = self.trajectory_recorder.get_trajectory_path()

        match self.agent_type:
            case AgentType.AutoDialectAgent:
                if config.autodialect_agent is None:
                    raise ValueError("autodialect_agent_config is required for AutoDialectAgent")
                from .autodialect_agent import AutoDialectAgent

                self.agent_config: AgentConfig = config.autodialect_agent

                self.agent: AutoDialectAgent = AutoDialectAgent(self.agent_config)
                self.agent.set_cli_console(cli_console)

        self.agent.set_trajectory_recorder(self.trajectory_recorder)

    async def run(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        self.agent.new_task(task, extra_args, tool_names)

        if self.agent.cli_console:
            task_details = {
                "Task": task,
                "Model Provider": self.agent_config.model.model_provider.provider,
                "Model": self.agent_config.model.model,
                "Max Steps": str(self.agent_config.max_steps),
                "Trajectory File": self.trajectory_file,
                "Tools": ", ".join([tool.name for tool in self.agent.tools]),
            }
            if extra_args:
                for key, value in extra_args.items():
                    task_details[key.capitalize()] = value
            self.agent.cli_console.print_task_details(task_details)

        cli_console_task = (
            asyncio.create_task(self.agent.cli_console.start()) if self.agent.cli_console else None
        )

        try:
            execution = await self.agent.execute_task()
        finally:
            if cli_console_task:
                await cli_console_task

        return execution
