"""
Tool wrappers that adapt Harbor's BaseEnvironment into lithos FunctionTools.
"""

from typing import Any, Dict

from lithos.tools import FunctionTool


def build_harbor_tools(environment: Any) -> Dict[str, FunctionTool]:
    """Wrap Harbor's environment.exec() as a lithos FunctionTool."""

    async def sandbox_sh(command: str, timeout_sec: int = 120) -> str:
        """Execute a sandbox_sh command in the task environment. Returns stdout and stderr."""
        result = await environment.exec(command, timeout_sec=timeout_sec)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return output if output else "(no output)"

    return {"sandbox_sh": FunctionTool(sandbox_sh)}
