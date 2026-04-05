import shlex

from pydantic import ConfigDict, Field

from ..core import InputSchema, Sandbox
from ..core.decorators import tool
from ._helpers import READ_DEFAULT_LINES, add_line_numbers


class ReadInput(InputSchema):
    file_path: str = Field(
        description="Absolute path of the file to read",
    )
    offset: int | None = Field(
        default=None,
        description="Line number to start reading from (1-based). "
        "Use only when the file is large and you need a specific range.",
    )
    limit: int | None = Field(
        default=None,
        description="Number of lines to read. Defaults to 2000. "
        "Use only when the file is large.",
    )

    model_config = ConfigDict(extra="forbid")


class WriteInput(InputSchema):
    file_path: str = Field(
        description="Absolute path of the file to write (must be absolute, not relative)",
    )
    content: str = Field(
        description="Text content to write into the file",
    )

    model_config = ConfigDict(extra="forbid")


class EditInput(InputSchema):
    file_path: str = Field(
        description="Absolute path of the file to edit",
    )
    old_string: str = Field(
        description="Text to find and replace",
    )
    new_string: str = Field(
        description="Replacement text (must differ from old_string)",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default false)",
    )

    model_config = ConfigDict(extra="forbid")


def make_file_tools(sandbox: Sandbox):
    @tool(schema=ReadInput)
    async def read_file(
        file_path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        """Read a file from the sandbox filesystem and return its contents with line numbers.

        Output is formatted with line numbers (like cat -n) so you can reference
        specific lines when using edit_file. Lines longer than 2000 characters
        are truncated.

        Usage notes:
          - file_path must be an absolute path, not relative.
          - By default reads up to 2000 lines from the start of the file.
          - Use offset and limit for large files: offset is the starting line
            number (1-based), limit is how many lines to read.
          - To read a specific range, e.g. lines 50-100: offset=50, limit=51.
        """
        start = offset or 1
        effective_limit = limit or READ_DEFAULT_LINES
        end = start + effective_limit - 1

        cmd = f"sed -n '{start},{end}p' {shlex.quote(file_path)}"
        raw = await sandbox.sh(cmd)

        if not raw and start == 1:
            # File might not exist or be empty — try cat to get the error
            result = await sandbox.exec("cat", file_path)
            if result:
                return add_line_numbers(result, start=start)
            return result

        return add_line_numbers(raw, start=start)

    @tool(schema=WriteInput)
    async def write_file(file_path: str, content: str) -> str:
        """Write text content to a file in the sandbox, creating parent directories as needed.

        This completely replaces the file contents. To make surgical edits to
        an existing file, use edit_file instead.

        Usage notes:
          - file_path must be an absolute path.
          - Parent directories are created automatically if they don't exist.
          - The file is created if it doesn't exist, or overwritten if it does.
        """
        await sandbox.sh(f"mkdir -p $(dirname {shlex.quote(file_path)})")
        await sandbox.exec("tee", file_path, input=content)
        return f"Successfully wrote to {file_path}"

    @tool(schema=EditInput)
    async def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Perform exact string replacement in a sandbox file.

        IMPORTANT: The old_string must match EXACTLY what is in the file,
        including all whitespace and indentation. Do NOT include line numbers
        from read_file output -- those are display-only and not part of the
        file content.

        Usage notes:
          - old_string must appear exactly once in the file (unless
            replace_all=true).
          - If the match is ambiguous (multiple occurrences), include more
            surrounding context in old_string to make it unique.
          - Whitespace and indentation must match precisely.
          - To create a new file, use write_file instead.
          - To append to a file, use the file's last line(s) as old_string and
            include the new content after them in new_string.
        """
        current = await sandbox.exec("cat", file_path)
        if old_string not in current:
            return f"Error: the specified text was not found in {file_path}"
        if not replace_all:
            count = current.count(old_string)
            if count > 1:
                return (
                    f"Error: the specified text appears {count} times in {file_path}. "
                    "Set replace_all=true or provide more surrounding context to make it unique."
                )
            result = current.replace(old_string, new_string, 1)
        else:
            result = current.replace(old_string, new_string)
        await sandbox.exec("tee", file_path, input=result)
        return f"Successfully edited {file_path}"

    return [read_file, write_file, edit_file]
