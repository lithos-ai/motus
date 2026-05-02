import logging
from pathlib import Path
from typing import Literal

from lithos.agent import ReActAgent
from lithos.tools import FunctionTool

from .agent_prompts import MAIN_AGENT_PROMPT, agent_prompt
from .tools import build_harbor_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Application")

# Type alias for available sub-agent names
SubAgentName = Literal["executor", "debugger", "syntax_verifier", "file_verifier"]


class ActusBot(ReActAgent):
    def __init__(self, environment, *args, **kwargs):
        # Set main agent system prompt if not provided
        if "system_prompt" not in kwargs:
            kwargs["system_prompt"] = MAIN_AGENT_PROMPT

        super().__init__(*args, **kwargs)
        self.environment = environment

        # Add launch_sub_agent tool after super().__init__
        launch_sub_agent_tool = FunctionTool(self.launch_sub_agent)
        if self.tools is None:
            self.tools = {"launch_sub_agent": launch_sub_agent_tool}
        else:
            self.tools.update({"launch_sub_agent": launch_sub_agent_tool})

    async def launch_sub_agent(
        self,
        task_prompt: str,
        agent_name: str,
    ) -> str:
        """
        Launch a specialized sub-agent to handle a specific task.

        Args:
            task_prompt: Clear, specific instructions with COMPLETE context for what the
                        sub-agent should do. Include goal, constraints, expected deliverables,
                        and ALL relevant background information. The sub-agent should have
                        everything needed to complete its task independently.
            agent_name: The type of sub-agent to launch. Available options:
                - "executor": Task execution - completes the ENTIRE task end-to-end in one go.
                              Only invoke multiple times if syntax_verifier reports failures.
                - "debugger": Diagnose and fix failures (only when executor fails)
                - "syntax_verifier": Syntax/import check only - checks code has valid syntax and imports (does NOT check correctness)
                - "file_verifier": File structure validation - checks workspace is clean

        Returns:
            The sub-agent's response after completing the task.
        """
        if agent_name not in agent_prompt:
            return f"Error: Unknown agent '{agent_name}'. Available agents: {list(agent_prompt.keys())}"

        sub_agent = ReActAgent(
            client=self.client,
            model_name=self.model_name,
            system_prompt=agent_prompt[agent_name],
            tools=build_harbor_tools(self.environment),
            enable_memory=False,
            skill_dir=str(Path(__file__).parent / "skills")
            if agent_name == "executor"
            else None,
        )

        logger.info(
            f"Launching {agent_name} sub-agent with task: {task_prompt[:100]}..."
        )
        result = await sub_agent(task_prompt)
        logger.info(f"Sub-agent {agent_name} completed task")

        return result
