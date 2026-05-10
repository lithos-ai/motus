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

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_PATH = PROMPTS_DIR / "coding_agent.md"
SECTIONS_DIR = PROMPTS_DIR / "sections"

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


def _build_plan_mode_section(*, enable_web: bool, enable_subagents: bool) -> str:
    """Render the plan-mode section, dropping cross-references to
    disabled tools so the model isn't told about tools it doesn't have.
    """
    template = (SECTIONS_DIR / "plan_mode.md").read_text(encoding="utf-8")
    return template.format(
        plan_mode_web_clause=", `web_fetch`, `web_search`" if enable_web else "",
        plan_mode_task_clause=" and the `task` dispatcher" if enable_subagents else "",
        plan_mode_explore_hint=(
            " (use `Explore` subagent instead)" if enable_subagents else ""
        ),
    )


def build_system_prompt(
    model_name: str,
    project_root: str | os.PathLike[str] | None = None,
    extra: str | None = None,
    *,
    enable_web: bool = True,
    enable_subagents: bool = True,
    enable_plan_mode: bool = True,
) -> str:
    """Build the coding-agent system prompt.

    Args:
        model_name: The model identifier; injected into the env block.
        project_root: Directory whose ``AGENTS.md`` / ``CLAUDE.md`` to
            include. Defaults to the current working directory.
        extra: Additional text appended at the very end of the prompt.
        enable_web: Whether ``web_search`` / ``web_fetch`` are wired into
            the agent. When ``False``, the "Web tools" section is
            omitted and references in other sections are dropped.
        enable_subagents: Whether the ``task`` tool is wired in. When
            ``False``, the "Subagents" section is omitted.
        enable_plan_mode: Whether ``enter_plan_mode`` / ``exit_plan_mode``
            are wired in. When ``False``, the "Plan mode" section is
            omitted.

    Returns:
        The fully rendered system prompt.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    env = _gather_env(model_name, root)

    plan_mode_section = (
        _build_plan_mode_section(
            enable_web=enable_web,
            enable_subagents=enable_subagents,
        )
        if enable_plan_mode
        else ""
    )
    subagents_section = (
        (SECTIONS_DIR / "subagents.md").read_text(encoding="utf-8")
        if enable_subagents
        else ""
    )
    web_tools_section = (
        (SECTIONS_DIR / "web_tools.md").read_text(encoding="utf-8")
        if enable_web
        else ""
    )

    prompt = template.format(
        plan_mode_section=plan_mode_section,
        subagents_section=subagents_section,
        web_tools_section=web_tools_section,
        **env,
    )

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
