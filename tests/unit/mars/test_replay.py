import asyncio
import json

import pytest

from motus.mars.tracing import AgentTrace
from motus.models import ChatCompletion


def _trace(trace_id="trace-1", agent_id="agent-1", turns=None):
    return AgentTrace.model_validate(
        {
            "schema": "motustracing.agent_trace.v1",
            "trace_id": trace_id,
            "agent": {
                "agent_instance_id": agent_id,
                "agent_class_id": "class-1",
            },
            "system_prompt": {"text": "system"},
            "turns": turns
            or [
                {
                    "turn_index": 0,
                    "input_delta": {"kind": "user", "text": "first"},
                    "output_tokens": 3,
                    "tools": [
                        {"name": "bash", "args": {"command": "pwd"}, "duration_ms": 10}
                    ],
                    "is_terminal": False,
                },
                {
                    "turn_index": 1,
                    "input_delta": {"kind": "tool_result", "text": "second"},
                    "output_tokens": 2,
                    "tools": [],
                    "is_terminal": True,
                },
            ],
        }
    )


class FakeClient:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        return ChatCompletion(
            id=f"completion-{idx}",
            model=kwargs["model"],
            content=f"generated-{idx}",
            finish_reason="length",
            usage={"completion_tokens": kwargs["max_tokens"]},
        )


class FailingClient:
    async def create(self, **kwargs):
        raise RuntimeError("context too long")


def test_build_agent_replay_uses_minimal_fields():
    from motus.mars.replay import build_agent_replay

    trace = _trace()
    replay = build_agent_replay(trace, trace.turns[0])

    assert replay == {
        "schema_version": 1,
        "trace_id": "trace-1",
        "turn_index": 0,
        "planned_tools": [
            {
                "name": "bash",
                "args": {"command": "pwd"},
                "duration_ms": 10,
            }
        ],
    }


def test_build_sampling_kwargs_forces_fixed_output_length():
    from motus.mars.replay import build_sampling_kwargs

    assert build_sampling_kwargs(5) == {
        "max_tokens": 5,
        "temperature": 0,
        "extra_body": {
            "min_tokens": 5,
            "ignore_eos": True,
        },
    }


@pytest.mark.asyncio
async def test_runner_appends_actual_generated_output_to_next_turn_context():
    from motus.mars.replay import TraceReplayRunner

    client = FakeClient()
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    runner = TraceReplayRunner(
        client=client,
        model="model",
        concurrency=1,
        sleep=sleep,
    )

    result = await runner.run_trace(_trace())

    assert result.trace_id == "trace-1"
    assert result.turns_completed == 2
    assert sleeps == [0.01]
    assert client.calls[0]["max_tokens"] == 3
    assert "min_tokens" not in client.calls[0]
    assert "ignore_eos" not in client.calls[0]
    assert client.calls[0]["extra_body"]["min_tokens"] == 3
    assert client.calls[0]["extra_body"]["ignore_eos"] is True
    assert client.calls[0]["extra_body"]["is_last_step"] is False
    assert client.calls[0]["extra_body"]["agent_replay"]["planned_tools"][0]["name"] == "bash"

    second_messages = client.calls[1]["messages"]
    assert [m.role for m in second_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[2].content == "generated-0"
    assert second_messages[3].content == "second"
    assert client.calls[1]["extra_body"]["is_last_step"] is True


@pytest.mark.asyncio
async def test_run_many_honors_concurrency_limit():
    from motus.mars.replay import TraceReplayRunner

    class SlowClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0

        async def create(self, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            try:
                return await super().create(**kwargs)
            finally:
                self.active -= 1

    client = SlowClient()
    runner = TraceReplayRunner(
        client=client,
        model="model",
        concurrency=2,
        sleep=lambda seconds: asyncio.sleep(0),
    )

    await runner.run_many([_trace("a", "a"), _trace("b", "b"), _trace("c", "c")])

    assert client.max_active <= 2


@pytest.mark.asyncio
async def test_run_many_errors_include_trace_identity(tmp_path):
    from motus.mars.replay import TraceReplayRunner

    trace = _trace("too-long", "agent-too-long")
    trace_path = tmp_path / "too-long.json"
    trace_path.write_text(
        json.dumps(trace.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    runner = TraceReplayRunner(
        client=FailingClient(),
        model="model",
        concurrency=1,
        sleep=lambda seconds: asyncio.sleep(0),
    )

    summary = await runner.run_many([trace_path])

    assert summary.completed_traces == 0
    assert summary.failed_traces == 1
    assert summary.errors[0].trace_id == "too-long"
    assert summary.errors[0].trace_path == str(trace_path)
    assert summary.errors[0].error == "context too long"


@pytest.mark.asyncio
async def test_subagent_trace_file_is_resolved_relative_to_parent(tmp_path):
    from motus.mars.replay import TraceReplayRunner

    child = _trace("child", "child-agent", turns=[
        {
            "turn_index": 0,
            "input_delta": {"kind": "user", "text": "child-input"},
            "output_tokens": 1,
            "tools": [],
            "is_terminal": True,
        }
    ])
    child_path = tmp_path / "child.json"
    child_path.write_text(
        json.dumps(child.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    parent = _trace(
        "parent",
        "parent-agent",
        turns=[
            {
                "turn_index": 0,
                "input_delta": {"kind": "user", "text": "parent-input"},
                "output_tokens": 1,
                "tools": [
                    {
                        "name": "task",
                        "args": {"prompt": "run child"},
                        "subagent_trace_file": "child.json",
                    }
                ],
                "is_terminal": False,
            },
            {
                "turn_index": 1,
                "input_delta": {"kind": "tool_result", "text": "parent-done"},
                "output_tokens": 1,
                "tools": [],
                "is_terminal": True,
            },
        ],
    )
    parent_path = tmp_path / "parent.json"

    client = FakeClient()
    runner = TraceReplayRunner(
        client=client,
        model="model",
        concurrency=1,
        sleep=lambda seconds: asyncio.sleep(0),
    )

    await runner.run_trace(parent, trace_path=parent_path)

    agent_ids = [call["extra_body"]["agent_instance_id"] for call in client.calls]
    assert agent_ids == ["parent-agent", "child-agent", "parent-agent"]
