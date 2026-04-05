"""Tests for motus.google_adk.agents.llm_agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.adk")

from motus.google_adk.agents.llm_agent import Agent
from motus.models import ChatMessage


def _make_event(*, text="hello", is_final=True):
    """Create a mock ADK Event."""
    part = MagicMock()
    part.text = text
    content = MagicMock()
    content.parts = [part]
    event = MagicMock()
    event.is_final_response.return_value = is_final
    event.content = content
    return event


@pytest.fixture
def agent():
    return Agent(
        model="gemini-2.5-flash",
        name="test_agent",
        description="A test agent.",
        instruction="You are a test agent.",
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class TestAgent:
    def test_subclasses_adk_agent(self):
        from google.adk.agents.llm_agent import Agent as ADKAgent

        assert issubclass(Agent, ADKAgent)

    def test_has_run_turn(self, agent):
        assert hasattr(agent, "run_turn")
        assert callable(agent.run_turn)

    def test_construction_matches_adk(self, agent):
        assert agent.name == "test_agent"
        assert agent.model == "gemini-2.5-flash"
        assert agent.description == "A test agent."


# ---------------------------------------------------------------------------
# run_turn
# ---------------------------------------------------------------------------


class TestRunTurn:
    async def _run_with_events(self, agent, events, message_text="hi"):
        """Patch InMemoryRunner and run a turn with the given mock events."""

        async def _fake_run_async(**kwargs):
            for e in events:
                yield e

        mock_runner = MagicMock()
        mock_runner.session_service.create_session = AsyncMock(return_value=MagicMock())
        mock_runner.session_service.append_event = AsyncMock()
        mock_runner.run_async = _fake_run_async
        mock_runner.close = AsyncMock()

        with patch(
            "motus.google_adk.agents.llm_agent.InMemoryRunner",
            return_value=mock_runner,
        ) as mock_cls:
            msg = ChatMessage.user_message(message_text)
            response, state = await agent.run_turn(msg, [])
            return response, state, mock_cls, mock_runner

    async def test_returns_final_response_text(self, agent):
        event = _make_event(text="mock reply")
        response, state, _, _ = await self._run_with_events(agent, [event])

        assert response.role == "assistant"
        assert response.content == "mock reply"

    async def test_state_appended(self, agent):
        event = _make_event(text="reply")

        response, state, _, _ = await self._run_with_events(
            agent, [event], message_text="hello"
        )

        assert len(state) == 2
        assert state[0].role == "user"
        assert state[0].content == "hello"
        assert state[1].role == "assistant"
        assert state[1].content == "reply"

    async def test_preserves_existing_state(self, agent):
        event = _make_event(text="reply")
        prior = [
            ChatMessage.user_message("prev"),
            ChatMessage.assistant_message(content="prev reply"),
        ]

        async def _fake_run_async(**kwargs):
            for e in [event]:
                yield e

        mock_runner = MagicMock()
        mock_runner.session_service.create_session = AsyncMock(return_value=MagicMock())
        mock_runner.session_service.append_event = AsyncMock()
        mock_runner.run_async = _fake_run_async
        mock_runner.close = AsyncMock()

        with patch(
            "motus.google_adk.agents.llm_agent.InMemoryRunner",
            return_value=mock_runner,
        ):
            msg = ChatMessage.user_message("new")
            response, state = await agent.run_turn(msg, prior)

        assert len(state) == 4
        assert state[0].content == "prev"
        assert state[3].content == "reply"

    async def test_history_replayed_into_session(self, agent):
        """Prior state is appended to the ADK session so the model sees full context."""
        event = _make_event(text="reply")
        prior = [
            ChatMessage.user_message("first"),
            ChatMessage.assistant_message(content="first reply"),
        ]

        async def _fake_run_async(**kwargs):
            yield event

        mock_session = MagicMock()
        mock_runner = MagicMock()
        mock_runner.session_service.create_session = AsyncMock(
            return_value=mock_session
        )
        mock_runner.session_service.append_event = AsyncMock()
        mock_runner.run_async = _fake_run_async
        mock_runner.close = AsyncMock()

        with patch(
            "motus.google_adk.agents.llm_agent.InMemoryRunner",
            return_value=mock_runner,
        ):
            await agent.run_turn(ChatMessage.user_message("new"), prior)

        assert mock_runner.session_service.append_event.await_count == 2
        calls = mock_runner.session_service.append_event.call_args_list

        user_event = calls[0].args[1]
        assert user_event.author == "user"
        assert user_event.content.parts[0].text == "first"

        assistant_event = calls[1].args[1]
        assert assistant_event.author == agent.name
        assert assistant_event.content.parts[0].text == "first reply"

    async def test_skips_non_final_events(self, agent):
        events = [
            _make_event(text="intermediate", is_final=False),
            _make_event(text="final answer", is_final=True),
        ]
        response, _, _, _ = await self._run_with_events(agent, events)

        assert response.content == "final answer"

    async def test_concatenates_multi_part_response(self, agent):
        part1 = MagicMock()
        part1.text = "part one "
        part2 = MagicMock()
        part2.text = "part two"
        content = MagicMock()
        content.parts = [part1, part2]
        event = MagicMock()
        event.is_final_response.return_value = True
        event.content = content

        response, _, _, _ = await self._run_with_events(agent, [event])

        assert response.content == "part one part two"

    async def test_no_response_fallback(self, agent):
        event = MagicMock()
        event.is_final_response.return_value = False
        event.content = None

        response, _, _, _ = await self._run_with_events(agent, [event])

        assert response.content == "(no response)"

    async def test_runner_closed(self, agent):
        event = _make_event(text="reply")
        _, _, _, mock_runner = await self._run_with_events(agent, [event])

        mock_runner.close.assert_awaited_once()

    async def test_creates_session(self, agent):
        event = _make_event(text="reply")
        _, _, _, mock_runner = await self._run_with_events(agent, [event])

        mock_runner.session_service.create_session.assert_awaited_once()


# ---------------------------------------------------------------------------
# serve worker integration
# ---------------------------------------------------------------------------


class TestServeIntegration:
    def test_resolve_import_path_finds_run_turn(self):
        """serve worker resolves an Agent instance that exposes run_turn."""
        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.agent:root_agent")
        assert hasattr(obj, "run_turn") and callable(obj.run_turn)

    def test_resolved_run_turn_is_async(self):
        import inspect

        from motus.serve.worker import _resolve_import_path

        obj = _resolve_import_path("examples.google_adk.agent:root_agent")
        assert inspect.iscoroutinefunction(obj.run_turn)
