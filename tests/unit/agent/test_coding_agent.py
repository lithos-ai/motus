"""Smoke tests for ``CodingAgent``.

Construction-level — no LLM round-trip. Validates that the template
wires up the right tools, renders the system prompt with env values,
and picks up project context (AGENTS.md / CLAUDE.md).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from motus.agent import CodingAgent
from motus.agent.templates import build_system_prompt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_client():
    """A stand-in chat client — never actually called in these tests."""
    return MagicMock(name="FakeChatClient")


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Empty project root under tmp_path."""
    return tmp_path


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_renders_env_placeholders(self, project_dir: Path):
        prompt = build_system_prompt(
            model_name="claude-sonnet-4-6",
            project_root=project_dir,
        )
        assert "{working_directory}" not in prompt
        assert "{today}" not in prompt
        assert "{model_name}" not in prompt
        assert str(project_dir) in prompt
        assert "claude-sonnet-4-6" in prompt
        assert datetime.date.today().isoformat() in prompt

    def test_no_project_context_when_files_absent(self, project_dir: Path):
        prompt = build_system_prompt(model_name="m", project_root=project_dir)
        # No project-context section should be appended.
        assert "Project-specific instructions follow" not in prompt
        assert "## AGENTS.md" not in prompt
        assert "## CLAUDE.md" not in prompt

    def test_picks_up_agents_md(self, project_dir: Path):
        (project_dir / "AGENTS.md").write_text("# Project Rules\nUse 4 spaces.")
        prompt = build_system_prompt(model_name="m", project_root=project_dir)
        assert "<system-reminder>" in prompt
        assert "## AGENTS.md" in prompt
        assert "Use 4 spaces." in prompt

    def test_picks_up_claude_md(self, project_dir: Path):
        (project_dir / "CLAUDE.md").write_text("Legacy instructions.")
        prompt = build_system_prompt(model_name="m", project_root=project_dir)
        assert "## CLAUDE.md" in prompt
        assert "Legacy instructions." in prompt

    def test_includes_both_when_both_exist(self, project_dir: Path):
        (project_dir / "AGENTS.md").write_text("primary")
        (project_dir / "CLAUDE.md").write_text("fallback")
        prompt = build_system_prompt(model_name="m", project_root=project_dir)
        assert "primary" in prompt
        assert "fallback" in prompt
        # AGENTS.md comes first per priority.
        assert prompt.index("AGENTS.md") < prompt.index("CLAUDE.md")

    def test_extra_appended(self, project_dir: Path):
        prompt = build_system_prompt(
            model_name="m",
            project_root=project_dir,
            extra="EXTRA_MARKER",
        )
        assert prompt.rstrip().endswith("EXTRA_MARKER")

    def test_git_repo_marker(self, project_dir: Path):
        (project_dir / ".git").mkdir()
        prompt = build_system_prompt(model_name="m", project_root=project_dir)
        assert "Is git repo: yes" in prompt

    def test_default_includes_optional_sections(self, project_dir: Path):
        prompt = build_system_prompt(model_name="m", project_root=project_dir)
        assert "## Plan mode" in prompt
        assert "## Subagents" in prompt
        assert "## Web tools" in prompt

    def test_disable_web_drops_web_section_and_references(self, project_dir: Path):
        prompt = build_system_prompt(
            model_name="m",
            project_root=project_dir,
            enable_web=False,
        )
        assert "## Web tools" not in prompt
        # No leftover references inside other sections.
        assert "web_search" not in prompt
        assert "web_fetch" not in prompt

    def test_disable_subagents_drops_section_and_references(self, project_dir: Path):
        prompt = build_system_prompt(
            model_name="m",
            project_root=project_dir,
            enable_subagents=False,
        )
        assert "## Subagents" not in prompt
        # The plan-mode "task dispatcher" mention is also dropped.
        assert "task` dispatcher" not in prompt
        # And the "use Explore subagent instead" hint.
        assert "Explore` subagent" not in prompt

    def test_disable_plan_mode_drops_section(self, project_dir: Path):
        prompt = build_system_prompt(
            model_name="m",
            project_root=project_dir,
            enable_plan_mode=False,
        )
        assert "## Plan mode" not in prompt
        assert "enter_plan_mode" not in prompt
        assert "exit_plan_mode" not in prompt

    def test_all_optional_sections_disabled(self, project_dir: Path):
        prompt = build_system_prompt(
            model_name="m",
            project_root=project_dir,
            enable_web=False,
            enable_subagents=False,
            enable_plan_mode=False,
        )
        for marker in (
            "## Plan mode",
            "## Subagents",
            "## Web tools",
            "web_search",
            "web_fetch",
            "enter_plan_mode",
        ):
            assert marker not in prompt, f"unexpected reference: {marker}"
        # Always-on sections should still be present.
        assert "## File-edit safety" in prompt
        assert "## Task management" in prompt


# ---------------------------------------------------------------------------
# CodingAgent — default tool wiring
# ---------------------------------------------------------------------------


EXPECTED_DEFAULT_TOOLS = {
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "glob_search",
    "grep_search",
    "to_do",
    "web_fetch",
    "web_search",
    "task",
    "enter_plan_mode",
    "exit_plan_mode",
}


class TestCodingAgentDefaults:
    def test_default_tools_wired(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        names = set(agent.tools)
        missing = EXPECTED_DEFAULT_TOOLS - names
        assert not missing, f"Missing default tools: {missing}"

    def test_no_load_skill_without_skills_dir(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        assert "load_skill" not in agent.tools

    def test_load_skill_added_when_skills_dir_provided(
        self, fake_client, project_dir: Path
    ):
        skills_root = project_dir / "skills"
        skills_root.mkdir()
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            skills_dir=skills_root,
        )
        assert "load_skill" in agent.tools

    def test_enable_web_false_drops_web_tools(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            enable_web=False,
        )
        assert "web_fetch" not in agent.tools
        assert "web_search" not in agent.tools
        # Other builtins still wired.
        assert "bash" in agent.tools
        assert "read_file" in agent.tools

    def test_enable_subagents_false_drops_task_tool(
        self, fake_client, project_dir: Path
    ):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            enable_subagents=False,
        )
        assert "task" not in agent.tools
        # Other builtins still wired.
        assert "bash" in agent.tools

    def test_enable_plan_mode_false_drops_plan_mode_tools(
        self, fake_client, project_dir: Path
    ):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            enable_plan_mode=False,
        )
        assert "enter_plan_mode" not in agent.tools
        assert "exit_plan_mode" not in agent.tools
        assert "bash" in agent.tools


# ---------------------------------------------------------------------------
# CodingAgent — plan mode
# ---------------------------------------------------------------------------


PLAN_MODE_REMAINING = {
    "read_file",
    "glob_search",
    "grep_search",
    "to_do",
    "web_fetch",
    "web_search",
    "exit_plan_mode",
}

PLAN_MODE_BLOCKED = {
    "bash",
    "write_file",
    "edit_file",
    "task",
    "enter_plan_mode",
}


class TestCodingAgentPlanMode:
    def test_starts_outside_plan_mode(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        assert agent.plan_mode_active is False

    def test_enter_filters_tool_set(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        toggled = agent._enter_plan_mode()
        assert toggled is True
        assert agent.plan_mode_active is True
        names = set(agent.tools)
        assert PLAN_MODE_REMAINING <= names
        assert not (PLAN_MODE_BLOCKED & names), (
            f"Blocked tools leaked into plan mode: {PLAN_MODE_BLOCKED & names}"
        )

    def test_re_enter_is_idempotent(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        agent._enter_plan_mode()
        # Second call should be a no-op.
        toggled = agent._enter_plan_mode()
        assert toggled is False
        assert agent.plan_mode_active is True

    def test_exit_restores_full_tool_set(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        original = set(agent.tools)
        agent._enter_plan_mode()
        toggled = agent._exit_plan_mode()
        assert toggled is True
        assert agent.plan_mode_active is False
        assert set(agent.tools) == original

    def test_exit_when_not_in_plan_mode_is_noop(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        toggled = agent._exit_plan_mode()
        assert toggled is False
        assert agent.plan_mode_active is False

    def test_custom_plan_mode_allowed_tools(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            plan_mode_allowed_tools=frozenset({"read_file", "exit_plan_mode"}),
        )
        agent._enter_plan_mode()
        names = set(agent.tools)
        assert names == {"read_file", "exit_plan_mode"}

    def test_subagents_have_no_plan_mode_tools(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        sub = agent._build_subagent("general-purpose", "m")
        assert "enter_plan_mode" not in sub.tools
        assert "exit_plan_mode" not in sub.tools

    @pytest.mark.asyncio
    async def test_dispatch_blocked_tool_in_plan_mode_returns_recovery_message(
        self, fake_client, project_dir: Path
    ):
        """Calling a write tool in plan mode returns an error message,
        not a KeyError, so the model can exit plan mode and recover.
        """
        from types import SimpleNamespace

        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        agent._enter_plan_mode()
        call = SimpleNamespace(
            function=SimpleNamespace(name="bash", arguments={"command": "ls"})
        )
        result = await agent._dispatch_tool_call(call)
        assert "blocked while in plan mode" in result
        assert "exit_plan_mode" in result

    def test_dispatch_unknown_tool_outside_plan_mode_still_raises(
        self, fake_client, project_dir: Path
    ):
        """Outside plan mode, the dispatcher keeps raising on unknown
        tool names — the graceful fallback is intentionally scoped to
        plan-mode filtering only.
        """
        from types import SimpleNamespace

        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        call = SimpleNamespace(
            function=SimpleNamespace(name="nonexistent_tool", arguments={})
        )
        with pytest.raises(KeyError):
            agent._dispatch_tool_call(call)


# ---------------------------------------------------------------------------
# CodingAgent — subagent factory
# ---------------------------------------------------------------------------


class TestCodingAgentSubagentFactory:
    def test_explore_subagent_has_read_only_tools(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="parent-m",
            project_root=project_dir,
        )
        sub = agent._build_subagent("Explore", "parent-m")

        # Read tools present
        assert "read_file" in sub.tools
        assert "glob_search" in sub.tools
        assert "grep_search" in sub.tools
        # Write tools filtered out
        assert "write_file" not in sub.tools
        assert "edit_file" not in sub.tools
        assert "bash" not in sub.tools

    def test_plan_subagent_has_read_only_tools(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="parent-m",
            project_root=project_dir,
        )
        sub = agent._build_subagent("Plan", "parent-m")

        assert "read_file" in sub.tools
        assert "edit_file" not in sub.tools
        assert "bash" not in sub.tools

    def test_general_purpose_subagent_keeps_all_tools(
        self, fake_client, project_dir: Path
    ):
        agent = CodingAgent(
            client=fake_client,
            model_name="parent-m",
            project_root=project_dir,
        )
        sub = agent._build_subagent("general-purpose", "parent-m")

        # general-purpose has no allowlist, so all builtins are present.
        assert "read_file" in sub.tools
        assert "write_file" in sub.tools
        assert "edit_file" in sub.tools
        assert "bash" in sub.tools

    def test_subagent_cannot_recurse(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="parent-m",
            project_root=project_dir,
        )
        sub = agent._build_subagent("general-purpose", "parent-m")

        # Recursive subagents disabled — no task tool inside the subagent.
        assert "task" not in sub.tools

    def test_subagent_uses_specified_model(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="parent-m",
            project_root=project_dir,
        )
        sub = agent._build_subagent("Explore", "claude-haiku-4-5")
        assert sub.model_name == "claude-haiku-4-5"

    def test_subagent_has_specialization_in_prompt(
        self, fake_client, project_dir: Path
    ):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        sub = agent._build_subagent("Explore", "m")
        assert "Explore sub-agent" in sub.system_prompt

    def test_subagent_has_fresh_memory(self, fake_client, project_dir: Path):
        import asyncio

        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )

        # Add a message to the parent. The subagent must not see it.
        asyncio.run(agent.add_user_message("PARENT_SECRET"))

        sub = agent._build_subagent("Explore", "m")
        sub_messages = [m.content for m in sub.messages if m.content]
        assert not any("PARENT_SECRET" in (c or "") for c in sub_messages)

    def test_default_memory_is_compact(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        # CompactionMemory has _token_threshold; BasicMemory does not.
        from motus.memory.compaction_memory import CompactionMemory

        assert isinstance(agent.memory, CompactionMemory)

    def test_extra_tools_appear_alongside_builtins(
        self, fake_client, project_dir: Path
    ):
        from motus.tools.core import InputSchema
        from motus.tools.core.decorators import tool

        class _NoArgs(InputSchema):
            pass

        @tool(schema=_NoArgs)
        async def my_extra() -> str:
            """Test tool."""
            return "ok"

        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            extra_tools=[my_extra],
        )
        names = set(agent.tools)
        assert "my_extra" in names
        assert EXPECTED_DEFAULT_TOOLS <= names

    def test_explicit_tools_replace_builtins(self, fake_client, project_dir: Path):
        from motus.tools.core import InputSchema
        from motus.tools.core.decorators import tool

        class _NoArgs(InputSchema):
            pass

        @tool(schema=_NoArgs)
        async def lonely() -> str:
            """Only tool."""
            return "ok"

        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            tools=[lonely],
        )
        names = set(agent.tools)
        assert names == {"lonely"}


# ---------------------------------------------------------------------------
# CodingAgent — system prompt
# ---------------------------------------------------------------------------


class TestCodingAgentSystemPrompt:
    def test_uses_built_prompt_by_default(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="claude-sonnet-4-6",
            project_root=project_dir,
        )
        sp = agent.system_prompt
        assert "claude-sonnet-4-6" in sp
        assert "Coding Agent" in sp

    def test_explicit_system_prompt_replaces_default(
        self, fake_client, project_dir: Path
    ):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            system_prompt="Hello.",
        )
        assert agent.system_prompt == "Hello."

    def test_system_prompt_extra_appended(self, fake_client, project_dir: Path):
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
            system_prompt_extra="ZZ_TAIL_MARKER",
        )
        assert "ZZ_TAIL_MARKER" in agent.system_prompt

    def test_agents_md_injected_into_prompt(self, fake_client, project_dir: Path):
        (project_dir / "AGENTS.md").write_text("Project rule: be terse.")
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        assert "Project rule: be terse." in agent.system_prompt


# ---------------------------------------------------------------------------
# CodingAgent — fork()
# ---------------------------------------------------------------------------


class TestCodingAgentFork:
    def test_fork_preserves_tools_and_prompt(self, fake_client, project_dir: Path):
        (project_dir / "AGENTS.md").write_text("FORK_MARKER")
        agent = CodingAgent(
            client=fake_client,
            model_name="m",
            project_root=project_dir,
        )
        forked = agent.fork()
        assert isinstance(forked, CodingAgent)
        assert set(forked.tools) == set(agent.tools)
        assert "FORK_MARKER" in forked.system_prompt
