"""Build the coding agent system prompt from a template + project context.

The prompt template ships with placeholders (``{working_directory}``,
``{today}``, etc.) that get filled in at agent construction time. Any
``AGENTS.md`` or ``CLAUDE.md`` found in *project_root* is appended as a
system-reminder block so the model picks up project-specific guidance
the same way Claude Code / Codex do.
"""

from __future__ import annotations

import datetime
import os
import platform
from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompts" / "coding_agent.md"

# Files we look for, in priority order. AGENTS.md is the convention Codex
# established and Claude Code adopted; CLAUDE.md is the older Anthropic
# convention. If both exist we include both.
PROJECT_CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md")


def _gather_env(model_name: str, project_root: Path) -> dict[str, str]:
    return {
        "working_directory": str(project_root),
        "is_git_repo": "yes" if (project_root / ".git").exists() else "no",
        "platform": platform.system(),
        "os_version": platform.platform(),
        "today": datetime.date.today().isoformat(),
        "model_name": model_name,
    }


def _read_project_context(project_root: Path) -> str:
    """Return the concatenated contents of any AGENTS.md / CLAUDE.md found.

    Returns an empty string if neither file exists.
    """
    blocks: list[str] = []
    for name in PROJECT_CONTEXT_FILES:
        path = project_root / name
        if path.is_file():
            try:
                blocks.append(f"## {name}\n\n{path.read_text(encoding='utf-8')}")
            except OSError:
                continue
    return "\n\n".join(blocks)


def build_system_prompt(
    model_name: str,
    project_root: str | os.PathLike[str] | None = None,
    extra: str | None = None,
) -> str:
    """Build the coding-agent system prompt.

    Args:
        model_name: The model identifier; injected into the env block.
        project_root: Directory whose ``AGENTS.md`` / ``CLAUDE.md`` to
            include. Defaults to the current working directory.
        extra: Additional text appended at the very end of the prompt.

    Returns:
        The fully rendered system prompt.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    env = _gather_env(model_name, root)
    prompt = template.format(**env)

    project_ctx = _read_project_context(root)
    if project_ctx:
        prompt += (
            "\n\n<system-reminder>\n"
            "Project-specific instructions follow. Treat them as authoritative "
            "for this codebase.\n\n"
            f"{project_ctx}\n"
            "</system-reminder>\n"
        )

    if extra:
        prompt += f"\n\n{extra}\n"

    return prompt
