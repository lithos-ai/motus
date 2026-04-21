"""Tests for the /eval/judge endpoint and judge logic."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from motus.models import ChatMessage
from motus.serve import AgentServer
from motus.serve.judge import run_llm_judge
from motus.serve.schemas import JudgeResponse


def _echo(message, state):
    response = ChatMessage.assistant_message(content=f"echo: {message.content}")
    return response, state + [message, response]


@pytest.fixture
def server():
    return AgentServer(_echo, max_workers=1)


@pytest.fixture
async def client(server):
    async with AsyncClient(
        transport=ASGITransport(app=server.app), base_url="http://test"
    ) as c:
        yield c


def _mock_openai_client(parsed=None, raise_exc=None):
    """Build a mock AsyncOpenAI that returns a completion with the given parsed result.

    parsed: JudgeResponse instance to return (or None for no parsed object)
    raise_exc: exception to raise from .parse() instead of returning
    """
    parse = AsyncMock()
    if raise_exc is not None:
        parse.side_effect = raise_exc
    else:
        message = SimpleNamespace(parsed=parsed)
        choice = SimpleNamespace(message=message)
        completion = SimpleNamespace(choices=[choice])
        parse.return_value = completion

    client = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=parse),
            ),
        ),
    )
    return client, parse


class TestJudgeLogic:
    """Direct tests of run_llm_judge (the judge function in motus)."""

    async def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        result = await run_llm_judge(
            model="claude-haiku-4-5",
            criteria="test",
            user_input="hi",
            agent_output="hello",
        )
        assert result is None

    async def test_missing_base_url_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        result = await run_llm_judge(
            model="claude-haiku-4-5",
            criteria="test",
            user_input="hi",
            agent_output="hello",
        )
        assert result is None

    async def test_returns_parsed_judge_response(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        parsed = JudgeResponse(score=0.8, passed=True, reason="good")
        fake_client, parse = _mock_openai_client(parsed=parsed)

        with patch("motus.serve.judge.AsyncOpenAI", return_value=fake_client):
            result = await run_llm_judge(
                model="claude-haiku-4-5",
                criteria="rubric",
                user_input="input-text",
                agent_output="output-text",
            )

        assert result is not None
        assert result.score == 0.8
        assert result.passed is True
        assert result.reason == "good"

        # Verify the request shape
        parse.assert_awaited_once()
        kwargs = parse.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["response_format"] is JudgeResponse
        assert kwargs["max_tokens"] == 512
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][1]["role"] == "user"
        assert "input-text" in kwargs["messages"][1]["content"]
        assert "output-text" in kwargs["messages"][1]["content"]
        assert "rubric" in kwargs["messages"][1]["content"]

    async def test_clamps_score_to_0_1(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        # Bypass Pydantic validation to simulate a model drifting outside [0,1].
        parsed = JudgeResponse.model_construct(score=1.7, passed=True, reason="")
        fake_client, _ = _mock_openai_client(parsed=parsed)

        with patch("motus.serve.judge.AsyncOpenAI", return_value=fake_client):
            result = await run_llm_judge(
                model="m", criteria="c", user_input="a", agent_output="b"
            )

        assert result is not None
        assert result.score == 1.0

    async def test_no_parsed_result_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        fake_client, _ = _mock_openai_client(parsed=None)

        with patch("motus.serve.judge.AsyncOpenAI", return_value=fake_client):
            result = await run_llm_judge(
                model="m", criteria="c", user_input="a", agent_output="b"
            )

        assert result is None

    async def test_exception_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        fake_client, _ = _mock_openai_client(raise_exc=RuntimeError("upstream"))

        with patch("motus.serve.judge.AsyncOpenAI", return_value=fake_client):
            result = await run_llm_judge(
                model="m", criteria="c", user_input="a", agent_output="b"
            )

        assert result is None


class TestJudgeEndpoint:
    """HTTP-level tests of the /eval/judge route."""

    async def test_happy_path(self, client, monkeypatch):
        async def fake_judge(**kwargs):
            assert kwargs["model"] == "claude-haiku-4-5"
            assert kwargs["user_input"] == "question"
            assert kwargs["agent_output"] == "answer"
            return JudgeResponse(score=0.9, passed=True, reason="great")

        monkeypatch.setattr("motus.serve.judge.run_llm_judge", fake_judge)

        r = await client.post(
            "/eval/judge",
            json={
                "input": "question",
                "output": "answer",
                "model": "claude-haiku-4-5",
                "criteria": "rubric",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["score"] == 0.9
        assert data["passed"] is True
        assert data["reason"] == "great"

    async def test_judge_failure_returns_502(self, client, monkeypatch):
        async def fake_judge(**kwargs):
            return None

        monkeypatch.setattr("motus.serve.judge.run_llm_judge", fake_judge)

        r = await client.post(
            "/eval/judge",
            json={"input": "q", "output": "a", "model": "m", "criteria": "c"},
        )
        assert r.status_code == 502

    async def test_request_validation(self, client):
        r = await client.post("/eval/judge", json={"input": "q"})
        assert r.status_code == 422
