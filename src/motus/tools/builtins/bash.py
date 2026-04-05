import asyncio

from pydantic import ConfigDict, Field

from ..core import InputSchema, Sandbox
from ..core.decorators import tool
from ._helpers import BASH_DEFAULT_TIMEOUT_MS, BASH_MAX_TIMEOUT_MS, truncate_output


class BashInput(InputSchema):
    command: str = Field(
        description="Shell command to execute",
    )
    timeout: float | None = Field(
        default=None,
        description="Optional timeout in milliseconds (max 600000). Default is 120000 (2 minutes).",
    )
    description: str | None = Field(
        default=None,
        description="Short, clear description of what the command does, in active voice.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def make_bash_tool(sandbox: Sandbox):
    @tool(schema=BashInput)
    async def bash(command: str, timeout: float | None = None, **_kwargs) -> str:
        """Execute a shell command in the sandbox and return its output.

        IMPORTANT: Use this tool for shell operations like git, npm, docker,
        pip, etc. Do NOT use it for file operations (reading, writing, editing,
        searching) -- use the dedicated read_file, write_file, edit_file,
        glob_search, and grep_search tools instead, as they provide better
        output formatting and safety limits.

        Usage notes:
          - Always quote file paths that contain spaces with double quotes.
          - Output is truncated to 30000 characters. For large outputs, pipe
            through head, tail, or grep to narrow results.
          - Default timeout is 120 seconds (max 600 seconds). Long-running
            commands will be terminated and return a timeout error.
          - A short description parameter helps with tracing and debugging.
        """
        timeout_ms = timeout or BASH_DEFAULT_TIMEOUT_MS
        timeout_ms = min(timeout_ms, BASH_MAX_TIMEOUT_MS)
        timeout_s = timeout_ms / 1000.0

        try:
            result = await asyncio.wait_for(
                sandbox.sh(command),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return (
                f"Error: Command timed out after {timeout_s:.0f} seconds. "
                "Consider increasing the timeout parameter or breaking "
                "the command into smaller steps."
            )

        return truncate_output(result)

    return bash
