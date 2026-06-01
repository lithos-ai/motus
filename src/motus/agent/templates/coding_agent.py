"""``CodingAgent`` — a ReActAgent template for software-engineering tasks.

The defaults match the harness shape of Claude Code / Codex: a curated
set of file, search, and shell tools, a system prompt that embodies the
"do what's asked, no more" philosophy, and automatic injection of any
``AGENTS.md`` / ``CLAUDE.md`` found in the project root.

Example::

    from motus.agent.templates import CodingAgent
    from motus.models import AnthropicChatClient

    agent = CodingAgent(
        client=AnthropicChatClient(),
        model_name="claude-sonnet-4-6",
    )
    result = await agent("Find the bug in src/parser.py and fix it.")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from motus.agent.react_agent import ReActAgent
from motus.models import CachePolicy, ReasoningConfig
from motus.tools import normalize_tools
from motus.tools.builtins import (
    PLAN_MODE_TOOLS,
    builtin_tools,
    make_plan_mode_tools,
    make_task_tool,
    make_web_fetch,
    make_web_search,
)
from motus.tools.core import Sandbox

from ._project_context import build_system_prompt
from .subagents import DEFAULT_SUBAGENTS, SubAgentSpec


class CodingAgent(ReActAgent):
    """ReActAgent preconfigured for coding tasks.

    Bundles ``bash``, ``read_file``/``write_file``/``edit_file``,
    ``glob_search``/``grep_search``, ``to_do``, and (optionally)
    ``load_skill``, behind a system prompt that mirrors the working
    style of Claude Code / Codex. ``AGENTS.md`` / ``CLAUDE.md`` from the
    project root are picked up automatically.

    All :class:`~motus.agent.ReActAgent` keyword arguments pass through
    unchanged, so any default can be overridden. Pass ``extra_tools=`` to
    add tools alongside the builtins, or ``tools=`` to replace them
    entirely.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        sandbox: Sandbox | None = None,
        project_root: str | os.PathLike[str] | None = None,
        skills_dir: str | os.PathLike[str] | None = None,
        enable_web: bool = True,
        web_search_api_key: str | None = None,
        web_fetch_extraction_model: str | None = None,
        enable_subagents: bool = True,
        subagent_specs: dict[str, SubAgentSpec] | None = None,
        enable_plan_mode: bool = True,
        plan_mode_allowed_tools: frozenset[str] | None = None,
        extra_tools: list | None = None,
        system_prompt_extra: str | None = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        memory_type: str = "compact",
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        cache_policy: CachePolicy | str = CachePolicy.AUTO,
        max_steps: int | None = None,
        **react_kwargs,
    ) -> None:
        """Construct the coding agent.

        Args:
            client: Chat client to use for LLM calls.
            model_name: Model identifier (e.g. ``"claude-sonnet-4-6"``).
            sandbox: Sandbox the builtin tools execute in. Defaults to
                a :class:`~motus.tools.providers.local.LocalShell` when
                omitted.
            project_root: Directory whose ``AGENTS.md`` / ``CLAUDE.md``
                to inject. Defaults to the current working directory.
            skills_dir: If set, a ``load_skill`` tool is added that can
                load skill packs from this directory.
            enable_web: When ``True`` (default), adds ``web_fetch`` and
                ``web_search`` tools. ``web_search`` gracefully no-ops
                with an informative message when no Brave API key is
                available.
            web_search_api_key: Brave API key for ``web_search``.
                Defaults to the ``BRAVE_API_KEY`` env var.
            web_fetch_extraction_model: Optional override for the model
                used inside ``web_fetch`` to extract page content.
                Defaults to the agent's main model. Use a small fast
                model (e.g. ``claude-haiku-4-5``) to keep costs low.
            enable_subagents: When ``True`` (default), adds a ``task``
                tool that lets the agent dispatch self-contained work
                to specialized subagents (``general-purpose``,
                ``Explore``, ``Plan``). Subagents themselves cannot
                launch further subagents.
            subagent_specs: Map of subagent type name to
                :class:`SubAgentSpec`. Defaults to
                :data:`DEFAULT_SUBAGENTS`. Pass a custom dict to add or
                replace types.
            enable_plan_mode: When ``True`` (default), adds
                ``enter_plan_mode`` / ``exit_plan_mode`` tools that let
                the agent switch into a read-only investigation phase
                before making changes.
            plan_mode_allowed_tools: Optional override for the read-only
                tool name set used in plan mode. Defaults to
                :data:`PLAN_MODE_TOOLS`.
            extra_tools: Additional tools to add alongside the builtins.
            system_prompt_extra: Extra text appended to the default
                system prompt.
            system_prompt: If provided, replaces the default system
                prompt entirely (``system_prompt_extra`` is ignored).
            tools: If provided, replaces the default tool set entirely
                (``extra_tools`` and ``skills_dir`` are ignored).
            memory_type: ``"compact"`` (default) or ``"basic"``. Coding
                sessions usually benefit from compaction.
            reasoning: Reasoning effort. Defaults to
                :meth:`ReasoningConfig.auto`.
            cache_policy: Prompt caching strategy. Defaults to
                :attr:`CachePolicy.AUTO`.
            max_steps: Max reasoning steps. ``None`` (default) means no
                limit — let the loop run until the agent finishes or
                hits a timeout.
            **react_kwargs: Forwarded to :class:`ReActAgent`.
        """
        # Persist for fork() reconstruction.
        self._sandbox = sandbox
        self._project_root = Path(project_root) if project_root is not None else None
        self._skills_dir = Path(skills_dir) if skills_dir is not None else None
        self._enable_web = enable_web
        self._web_search_api_key = web_search_api_key
        self._web_fetch_extraction_model = web_fetch_extraction_model
        self._enable_subagents = enable_subagents
        self._subagent_specs = subagent_specs or DEFAULT_SUBAGENTS
        self._enable_plan_mode = enable_plan_mode
        self._plan_mode_allowed_tools = plan_mode_allowed_tools or PLAN_MODE_TOOLS
        self._plan_mode_active = False
        self._all_tools: dict | None = None  # set after super().__init__
        self._extra_tools = list(extra_tools) if extra_tools else None
        self._system_prompt_extra = system_prompt_extra

        if system_prompt is None:
            system_prompt = build_system_prompt(
                model_name=model_name,
                project_root=self._project_root,
                extra=system_prompt_extra,
                enable_web=enable_web,
                enable_subagents=enable_subagents,
                enable_plan_mode=enable_plan_mode,
            )

        if tools is None:
            bt = builtin_tools(sandbox=sandbox, skills_dir=self._skills_dir)
            tool_list = [*bt]
            if enable_web:
                tool_list.append(
                    make_web_fetch(
                        client=client,
                        model_name=model_name,
                        extraction_model=web_fetch_extraction_model,
                    )
                )
                tool_list.append(make_web_search(api_key=web_search_api_key))
            if enable_subagents:
                tool_list.append(
                    make_task_tool(
                        subagent_factory=self._build_subagent,
                        subagent_specs=self._subagent_specs,
                        parent_model=model_name,
                    )
                )
            if enable_plan_mode:
                enter_tool, exit_tool = make_plan_mode_tools(
                    enter_fn=self._enter_plan_mode,
                    exit_fn=self._exit_plan_mode,
                )
                tool_list.append(enter_tool)
                tool_list.append(exit_tool)
            if self._extra_tools:
                tool_list.extend(self._extra_tools)
            tools = normalize_tools(tool_list)

        super().__init__(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            tools=tools,
            memory_type=memory_type,
            reasoning=reasoning,
            cache_policy=cache_policy,
            max_steps=max_steps,
            **react_kwargs,
        )

        # Snapshot the full tool set so plan-mode toggles can restore it.
        # Captured after super() so ``self._tools`` has been normalized.
        if self._tools is not None:
            self._all_tools = dict(self._tools)

    def _fork_kwargs(self) -> dict:
        kwargs = super()._fork_kwargs()
        kwargs.update(
            {
                "sandbox": self._sandbox,
                "project_root": self._project_root,
                "skills_dir": self._skills_dir,
                "enable_web": self._enable_web,
                "web_search_api_key": self._web_search_api_key,
                "web_fetch_extraction_model": self._web_fetch_extraction_model,
                "enable_subagents": self._enable_subagents,
                "subagent_specs": self._subagent_specs,
                "enable_plan_mode": self._enable_plan_mode,
                "plan_mode_allowed_tools": self._plan_mode_allowed_tools,
                "extra_tools": self._extra_tools,
                "system_prompt_extra": self._system_prompt_extra,
            }
        )
        return kwargs

    # -------------------------------------------------------------------------
    # Plan mode
    # -------------------------------------------------------------------------

    @property
    def plan_mode_active(self) -> bool:
        """Whether the agent is currently in plan mode."""
        return self._plan_mode_active

    def _enter_plan_mode(self) -> bool:
        """Switch into plan mode — filter tools to the read-only subset.

        Returns ``True`` if the mode was toggled, ``False`` if already
        in plan mode.
        """
        if self._plan_mode_active or self._all_tools is None:
            return False
        self._plan_mode_active = True
        self._tools = {
            name: t
            for name, t in self._all_tools.items()
            if name in self._plan_mode_allowed_tools
        }
        return True

    def _exit_plan_mode(self) -> bool:
        """Restore the full tool set after plan mode.

        Returns ``True`` if the mode was toggled, ``False`` if not in
        plan mode.
        """
        if not self._plan_mode_active or self._all_tools is None:
            return False
        self._plan_mode_active = False
        self._tools = dict(self._all_tools)
        return True

    def _dispatch_tool_call(self, call: Any):
        """Dispatch a tool call, with plan-mode-aware error recovery.

        Models occasionally call a write tool (``bash``, ``write_file``,
        ``edit_file``, ``task``) while in plan mode despite the prompt
        telling them not to. The default dispatcher would raise
        ``KeyError`` and abort the run; instead, return a recovery
        message describing why the tool is unavailable so the model can
        ``exit_plan_mode`` and try again.
        """
        name = call.function.name
        if (
            self._plan_mode_active
            and name not in self._tools
            and self._all_tools is not None
            and name in self._all_tools
        ):
            allowed = ", ".join(sorted(self._tools))

            async def _blocked() -> str:
                return (
                    f"Error: tool '{name}' is blocked while in plan mode. "
                    f"Allowed tools: {allowed}. "
                    f"Call `exit_plan_mode(plan=...)` first if you need "
                    f"to make changes."
                )

            return _blocked()
        return super()._dispatch_tool_call(call)

    def _build_subagent(
        self,
        spec_name: str,
        model_id: str,
    ) -> "CodingAgent":
        """Construct a fresh subagent for the ``task`` tool.

        Each call produces a brand-new :class:`CodingAgent` — fresh
        memory, scoped tools, no parent-message inheritance — with the
        spec's ``extra_prompt`` appended and its ``allowed_tools``
        filter applied. Recursive subagents are disabled.
        """
        spec = self._subagent_specs[spec_name]

        sub = CodingAgent(
            client=self._client,
            model_name=model_id,
            sandbox=self._sandbox,
            project_root=self._project_root,
            skills_dir=self._skills_dir,
            enable_web=self._enable_web,
            web_search_api_key=self._web_search_api_key,
            web_fetch_extraction_model=self._web_fetch_extraction_model,
            enable_subagents=False,
            enable_plan_mode=False,
            system_prompt_extra=spec.extra_prompt,
            memory_type=self._memory_type,
            reasoning=self._reasoning,
            cache_policy=self._cache_policy,
            name=f"{spec_name}_subagent",
        )

        if spec.allowed_tools is not None:
            sub.tools = {
                name: t for name, t in sub.tools.items() if name in spec.allowed_tools
            }

        return sub
