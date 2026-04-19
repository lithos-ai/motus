"""Tests for the /eval/judge endpoint and judge logic."""

import json
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient, Request, Response

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


class TestJudgeLogic:
    """Direct tests of run_llm_judge (the judge function in motus)."""

    async def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        result = await run_llm_judge(
            model="claude-haiku-4-5",
            criteria="judge: {input} / {output}",
            user_input="hi",
            agent_output="hello",
        )
        assert result is None

    async def test_missing_base_url_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        result = await run_llm_judge(
            model="claude-haiku-4-5",
            criteria="judge: {input} / {output}",
            user_input="hi",
            agent_output="hello",
        )
        assert result is None

    async def test_parses_score_passed_reason(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        def handler(request: Request) -> Response:
            # Verify we hit /chat/completions on the configured proxy
            assert str(request.url) == "http://model-proxy/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer lithos_test"
            body = json.loads(request.content)
            assert body["model"] == "claude-haiku-4-5"
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"
            assert "input-text" in body["messages"][1]["content"]
            assert "output-text" in body["messages"][1]["content"]
            assert body["response_format"] == {"type": "json_object"}
            return Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"score": 0.8, "passed": True, "reason": "good"}
                                )
                            }
                        }
                    ]
                },
            )

        RealAsyncClient = httpx.AsyncClient

        def fake_client(*a, **k):
            return RealAsyncClient(transport=httpx.MockTransport(handler))

        with patch("motus.serve.judge.httpx.AsyncClient", side_effect=fake_client):
            result = await run_llm_judge(
                model="claude-haiku-4-5",
                criteria="input={input} output={output}",
                user_input="input-text",
                agent_output="output-text",
            )

        assert result is not None
        assert result.score == 0.8
        assert result.passed is True
        assert result.reason == "good"

    async def test_handles_markdown_code_fences(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        def handler(request: Request) -> Response:
            return Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '```json\n{"score": 1.0, "passed": true, "reason": "ok"}\n```'
                            }
                        }
                    ]
                },
            )

        RealAsyncClient = httpx.AsyncClient

        def fake_client(*a, **k):
            return RealAsyncClient(transport=httpx.MockTransport(handler))

        with patch("motus.serve.judge.httpx.AsyncClient", side_effect=fake_client):
            result = await run_llm_judge(
                model="claude-haiku-4-5",
                criteria="j",
                user_input="a",
                agent_output="b",
            )

        assert result is not None
        assert result.score == 1.0
        assert result.passed is True

    async def test_clamps_score_to_0_1(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        def handler(request: Request) -> Response:
            return Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"score": 1.7, "passed": true, "reason": ""}'
                            }
                        }
                    ]
                },
            )

        RealAsyncClient = httpx.AsyncClient

        def fake_client(*a, **k):
            return RealAsyncClient(transport=httpx.MockTransport(handler))

        with patch("motus.serve.judge.httpx.AsyncClient", side_effect=fake_client):
            result = await run_llm_judge(
                model="m",
                criteria="j",
                user_input="a",
                agent_output="b",
            )

        assert result is not None
        assert result.score == 1.0

    async def test_http_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        def handler(request: Request) -> Response:
            return Response(500, text="internal error")

        RealAsyncClient = httpx.AsyncClient

        def fake_client(*a, **k):
            return RealAsyncClient(transport=httpx.MockTransport(handler))

        with patch("motus.serve.judge.httpx.AsyncClient", side_effect=fake_client):
            result = await run_llm_judge(
                model="m",
                criteria="j",
                user_input="a",
                agent_output="b",
            )

        assert result is None

    async def test_invalid_json_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "lithos_test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy/v1")

        def handler(request: Request) -> Response:
            return Response(
                200,
                json={"choices": [{"message": {"content": "not json at all"}}]},
            )

        RealAsyncClient = httpx.AsyncClient

        def fake_client(*a, **k):
            return RealAsyncClient(transport=httpx.MockTransport(handler))

        with patch("motus.serve.judge.httpx.AsyncClient", side_effect=fake_client):
            result = await run_llm_judge(
                model="m",
                criteria="j",
                user_input="a",
                agent_output="b",
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
                "criteria": "prompt",
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
            json={
                "input": "q",
                "output": "a",
                "model": "m",
                "criteria": "p",
            },
        )
        assert r.status_code == 502

    async def test_request_validation(self, client):
        # Missing required fields
        r = await client.post("/eval/judge", json={"input": "q"})
        assert r.status_code == 422
