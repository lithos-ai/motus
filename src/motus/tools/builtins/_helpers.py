"""Shared constants and utilities for builtin tools."""

# --- Output safety limits ---
BASH_OUTPUT_LIMIT = 30_000  # Max characters returned from bash
READ_DEFAULT_LINES = 2000  # Default max lines for read_file
READ_LINE_TRUNCATE = 2000  # Max characters per line in read_file

# --- Timeout ---
BASH_DEFAULT_TIMEOUT_MS = 120_000  # 2 minutes
BASH_MAX_TIMEOUT_MS = 600_000  # 10 minutes

# --- Type map for grep --include ---
FILE_TYPE_MAP = {
    "js": "*.js",
    "ts": "*.ts",
    "tsx": "*.tsx",
    "jsx": "*.jsx",
    "py": "*.py",
    "rust": "*.rs",
    "rs": "*.rs",
    "go": "*.go",
    "java": "*.java",
    "c": "*.c",
    "cpp": "*.cpp",
    "h": "*.h",
    "rb": "*.rb",
    "php": "*.php",
    "swift": "*.swift",
    "kt": "*.kt",
    "scala": "*.scala",
    "sh": "*.sh",
    "bash": "*.sh",
    "css": "*.css",
    "html": "*.html",
    "json": "*.json",
    "yaml": "*.yaml",
    "yml": "*.yml",
    "toml": "*.toml",
    "md": "*.md",
    "sql": "*.sql",
    "xml": "*.xml",
}


def truncate_output(text: str, limit: int = BASH_OUTPUT_LIMIT) -> str:
    """Truncate text to *limit* characters, appending a notice if truncated."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    return (
        truncated
        + f"\n\n... Output truncated ({len(text)} chars total, showing first {limit})"
    )


def truncate_line(line: str, limit: int = READ_LINE_TRUNCATE) -> str:
    """Truncate a single line to *limit* characters."""
    if len(line) <= limit:
        return line
    return line[:limit] + "..."


def add_line_numbers(text: str, start: int = 1) -> str:
    """Add line numbers in cat -n format: ``     1\\tline content``.

    This format enables precise line references in subsequent edit_file calls.
    """
    lines = text.split("\n")
    # Remove trailing empty line that cat/file reads typically produce
    if lines and lines[-1] == "":
        lines = lines[:-1]
    numbered = []
    for i, line in enumerate(lines, start=start):
        truncated = truncate_line(line)
        numbered.append(f"{i:>6}\t{truncated}")
    return "\n".join(numbered)
