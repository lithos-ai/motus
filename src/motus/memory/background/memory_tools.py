"""
File tools for the memory agent, scoped to the memory root directory.

All paths are relative to the memory root. The agent cannot read or write
outside the memory tree. Tools mirror the motus file builtins but operate
directly on the local filesystem without a sandbox.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def make_memory_tools(root: Path):
    """
    Return file tools scoped to the memory root directory.

    All file_path arguments are relative to root. Absolute paths that fall
    outside root are rejected.
    """

    def _resolve(rel_path: str) -> Path:
        """Resolve a relative path and ensure it stays within root."""
        resolved = (root / rel_path).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            raise ValueError(f"Path {rel_path!r} is outside the memory root")
        return resolved

    async def read_file(
        file_path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        """Read a file from the memory tree.

        Args:
            file_path: Path relative to the memory root (e.g. "recent/recent.md")
            offset: Line number to start reading from (1-based).
            limit: Number of lines to read. Defaults to 500.
        """
        path = _resolve(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"
        lines = path.read_text().splitlines()
        start = (offset or 1) - 1
        end = start + (limit or 500)
        selected = lines[start:end]
        return "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))

    async def write_file(file_path: str, content: str) -> str:
        """Write content to a file in the memory tree, creating parent dirs as needed.

        This completely replaces the file. Use edit_file for partial changes.

        Args:
            file_path: Path relative to the memory root.
            content: Full file content to write.
        """
        path = _resolve(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Wrote {file_path}"

    async def edit_file(
        file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Perform exact string replacement in a memory file.

        old_string must match exactly (including whitespace).

        Args:
            file_path: Path relative to the memory root.
            old_string: Text to find and replace.
            new_string: Replacement text.
            replace_all: Replace all occurrences (default: false).
        """
        path = _resolve(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"
        current = path.read_text()
        if old_string not in current:
            return f"Error: text not found in {file_path}"
        if not replace_all:
            count = current.count(old_string)
            if count > 1:
                return f"Error: text appears {count} times. Use replace_all=true or add more context."
            result = current.replace(old_string, new_string, 1)
        else:
            result = current.replace(old_string, new_string)
        path.write_text(result)
        return f"Edited {file_path}"

    async def delete_file(file_path: str) -> str:
        """Delete a file from the memory tree.

        Cannot delete directories or files in raw_logs/ (immutable).

        Args:
            file_path: Path relative to the memory root.
        """
        path = _resolve(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"
        if not path.is_file():
            return f"Error: not a file: {file_path}"
        if "raw_logs" in path.parts:
            return f"Error: cannot delete raw logs: {file_path}"
        path.unlink()
        return f"Deleted {file_path}"

    async def glob_search(pattern: str) -> str:
        """Find files matching a glob pattern within the memory tree.

        Args:
            pattern: Glob pattern relative to memory root (e.g. "projects/**/*.md")
        """
        matches = sorted(root.glob(pattern))
        if not matches:
            return "No files found."
        return "\n".join(str(p.relative_to(root)) for p in matches)

    async def grep_search(pattern: str, path: str | None = None) -> str:
        """Search file contents for lines matching a regex pattern.

        Args:
            pattern: Regex pattern to search for.
            path: Subdirectory to search (relative to memory root). Defaults to root.
        """
        search_path = _resolve(path) if path else root
        try:
            result = subprocess.run(
                ["grep", "-r", "-n", "--include=*.md", pattern, str(search_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip()
            if not output:
                return "No matches found."
            # Make paths relative to root
            lines = []
            for line in output.splitlines():
                abs_path, rest = line.split(":", 1) if ":" in line else (line, "")
                try:
                    rel = Path(abs_path).relative_to(root)
                    lines.append(f"{rel}:{rest}")
                except ValueError:
                    lines.append(line)
            return "\n".join(lines)
        except subprocess.TimeoutExpired:
            return "Error: grep timed out"

    async def list_files(path: str = ".") -> str:
        """List files and directories in the memory tree with metadata.

        Returns one entry per line with size and last modified time.
        Use this to discover the memory store structure before reading files.

        Args:
            path: Directory to list, relative to memory root. Defaults to root.
        """
        from datetime import datetime, timezone

        target = _resolve(path)
        if not target.exists():
            return f"Error: directory not found: {path}"
        if not target.is_dir():
            return f"Error: not a directory: {path}"
        entries = []
        for item in sorted(target.iterdir()):
            rel = item.relative_to(root)
            stat = item.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
            if item.is_dir():
                count = sum(1 for _ in item.iterdir())
                entries.append(f"{rel}/  ({count} items, modified {mtime})")
            else:
                size = stat.st_size
                entries.append(f"{rel}  ({size} bytes, modified {mtime})")
        if not entries:
            return "(empty directory)"
        return "\n".join(entries)

    return [
        read_file,
        write_file,
        edit_file,
        delete_file,
        list_files,
        glob_search,
        grep_search,
    ]
