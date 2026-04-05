import shlex

from pydantic import ConfigDict, Field

from ..core import InputSchema, Sandbox
from ..core.decorators import tool
from ._helpers import FILE_TYPE_MAP, truncate_output


class GlobInput(InputSchema):
    pattern: str = Field(
        description="Glob pattern to match files. Supports ** for recursive matching "
        '(e.g. "**/*.py"), standard wildcards (*, ?, [abc]).',
    )
    path: str | None = Field(
        default=None,
        description="Directory to search. If omitted, uses current working directory. "
        'IMPORTANT: omit this field to use the default. Do not pass "undefined" / "null". '
        "Must be a valid directory when provided.",
    )

    model_config = ConfigDict(extra="forbid")


class GrepInput(InputSchema):
    pattern: str = Field(
        description="Regex pattern to search in file contents",
    )
    path: str | None = Field(
        default=None,
        description="File or directory to search. Defaults to current working directory.",
    )
    glob: str | None = Field(
        default=None,
        description='Glob filter for files (e.g. "*.js", "*.{ts,tsx}") — maps to grep --include',
    )
    output_mode: str | None = Field(
        default=None,
        description='Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit); '
        '"files_with_matches" shows file paths (supports head_limit); '
        '"count" shows match counts (supports head_limit). Defaults to "files_with_matches".',
    )
    B: int | None = Field(
        default=None,
        alias="-B",
        description='Number of lines before each match (grep -B). Requires output_mode: "content"; ignored otherwise.',
    )
    A: int | None = Field(
        default=None,
        alias="-A",
        description='Number of lines after each match (grep -A). Requires output_mode: "content"; ignored otherwise.',
    )
    C: int | None = Field(
        default=None,
        alias="-C",
        description='Number of lines before and after each match (grep -C). Requires output_mode: "content"; ignored otherwise.',
    )
    n: bool | None = Field(
        default=None,
        alias="-n",
        description='Show line numbers (grep -n). Requires output_mode: "content"; ignored otherwise. '
        "Defaults to true.",
    )
    i: bool | None = Field(
        default=None,
        alias="-i",
        description="Case-insensitive search (grep -i)",
    )
    type: str | None = Field(
        default=None,
        description="File type to search. Common types: js, py, ts, rust, go, java, c, cpp, rb, etc. "
        'Maps to grep --include (e.g. type="py" searches only *.py files).',
    )
    head_limit: int | None = Field(
        default=None,
        description='Limit output to first N lines/entries, like "| head -N". '
        "Works across all output modes. Defaults to 0 (unlimited).",
    )
    offset: int | None = Field(
        default=None,
        description='Skip first N lines/entries before head_limit, like "| tail -n +N | head -N". '
        "Works across all output modes. Defaults to 0.",
    )
    multiline: bool | None = Field(
        default=None,
        description="Enable multiline mode where . matches newlines and patterns can span lines. "
        "Not yet supported in all sandbox environments. Default: false.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def make_search_tools(sandbox: Sandbox):
    @tool(schema=GlobInput)
    async def glob_search(pattern: str, path: str | None = None) -> str:
        """Find files matching a glob pattern in the sandbox filesystem.

        Returns matching file paths sorted alphabetically.

        Usage notes:
          - Supports ** for recursive matching: "**/*.py" finds all Python files
            in any subdirectory.
          - Supports standard glob wildcards: *, ?, [abc].
          - path defaults to the current working directory if omitted.
          - Use this tool to find files by name. To search file contents, use
            grep_search instead.
        """
        target = shlex.quote(path) if path else "."

        if pattern.startswith("**/"):
            # **/*.py -> find is already recursive, just match the name part
            name_part = pattern[3:]  # strip **/
            if "/" in name_part:
                # **/src/*.py -> use -path with wildcard
                find_pattern = name_part.replace("**", "*")
                cmd = f"find {target} -path {shlex.quote('./' + find_pattern)} -type f 2>/dev/null | sort"
            else:
                # **/*.py -> same as -name *.py (find is recursive by default)
                cmd = f"find {target} -name {shlex.quote(name_part)} -type f 2>/dev/null | sort"
        elif "/" in pattern or "**" in pattern:
            # src/**/*.py -> use -path (find's * matches / in -path)
            find_pattern = pattern.replace("**", "*")
            cmd = f"find {target} -path {shlex.quote('./' + find_pattern)} -type f 2>/dev/null | sort"
        else:
            # Simple name pattern: use find -name
            cmd = (
                f"find {target} -name {shlex.quote(pattern)} -type f 2>/dev/null | sort"
            )

        result = await sandbox.sh(cmd)
        return truncate_output(result)

    @tool(schema=GrepInput)
    async def grep_search(pattern: str, path: str | None = None, **_kwargs) -> str:
        """Search file contents for lines matching a regex pattern in the sandbox.

        Recursively searches through files. Default output_mode is
        "files_with_matches" which returns only file paths containing matches.

        Output modes:
          - "files_with_matches" (default): returns file paths only.
          - "content": returns matching lines with context. Supports -A, -B, -C
            for context lines, -n for line numbers (on by default).
          - "count": returns match counts per file.

        Usage notes:
          - pattern is a regular expression (not a literal string).
          - Use glob parameter to filter by extension: glob="*.py".
          - Use type parameter for common file types: type="py", type="js".
          - Use head_limit to restrict output size.
          - Use offset with head_limit for pagination.
          - Case-insensitive search: set -i to true.
          - Use this for content search. To find files by name, use glob_search.
        """
        parts = ["grep", "-r"]
        if _kwargs.get("i"):
            parts.append("-i")
        if _kwargs.get("n", True):
            parts.append("-n")
        context = _kwargs.get("C")
        if context:
            parts.extend(["-C", str(context)])
        else:
            before = _kwargs.get("B")
            after = _kwargs.get("A")
            if before:
                parts.extend(["-B", str(before)])
            if after:
                parts.extend(["-A", str(after)])
        output_mode = _kwargs.get("output_mode")
        if output_mode == "files_with_matches":
            parts.append("-l")
        elif output_mode == "count":
            parts.append("-c")

        # Glob filter
        glob_filter = _kwargs.get("glob")
        if glob_filter:
            parts.append(f"--include={shlex.quote(glob_filter)}")

        # Type filter -> --include mapping
        file_type = _kwargs.get("type")
        if file_type:
            include_pattern = FILE_TYPE_MAP.get(file_type, f"*.{file_type}")
            parts.append(f"--include={shlex.quote(include_pattern)}")

        parts.append(shlex.quote(pattern))
        target = shlex.quote(path) if path else "."
        parts.append(target)
        cmd = " ".join(parts)

        # Offset: skip first N entries
        offset = _kwargs.get("offset")
        if offset and offset > 0:
            cmd += f" | tail -n +{offset + 1}"

        # Head limit: restrict output size
        head_limit = _kwargs.get("head_limit")
        if head_limit and head_limit > 0:
            cmd += f" | head -n {head_limit}"

        result = await sandbox.sh(cmd)
        return truncate_output(result)

    return [glob_search, grep_search]
