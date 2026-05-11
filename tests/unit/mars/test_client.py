from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from motus.models import ChatMessage


@pytest.mark.asyncio
async def test_client_merges_mars_fields_into_extra_body():
    from motus.mars import MarsOpenAIChatClient

    client = MarsOpenAIChatClient(
        api_key="test",
        agent_instance_id="agent-1",
        agent_class_id="class-1",
        is_last_step=False,
    )

    with patch(
        "motus.models.openai_client.OpenAIChatClient.create",
        new=AsyncMock(return_value="completion"),
    ) as create:
        result = await client.create(
            model="model",
            messages=[ChatMessage.user_message("hi")],
            extra_body={"temperature_seed": 7},
        )

    assert result == "completion"
    kwargs = create.await_args.kwargs
    assert kwargs["extra_body"] == {
        "temperature_seed": 7,
        "agent_instance_id": "agent-1",
        "agent_class_id": "class-1",
        "is_last_step": False,
    }


@pytest.mark.asyncio
async def test_client_preserves_existing_extra_body():
    from motus.mars import MarsOpenAIChatClient

    extra_body = {"agent_instance_id": "per-request", "ignore_eos": True}
    client = MarsOpenAIChatClient(
        api_key="test",
        agent_instance_id="client-default",
        agent_class_id="class-1",
    )

    with patch(
        "motus.models.openai_client.OpenAIChatClient.create",
        new=AsyncMock(return_value="completion"),
    ) as create:
        await client.create(
            model="model",
            messages=[ChatMessage.user_message("hi")],
            extra_body=extra_body,
        )

    assert extra_body == {"agent_instance_id": "per-request", "ignore_eos": True}
    assert create.await_args.kwargs["extra_body"] == {
        "agent_instance_id": "per-request",
        "ignore_eos": True,
        "agent_class_id": "class-1",
    }


@pytest.mark.asyncio
async def test_client_does_not_include_none_fields():
    from motus.mars import MarsOpenAIChatClient

    client = MarsOpenAIChatClient(api_key="test", agent_class_id="class-1")

    with patch(
        "motus.models.openai_client.OpenAIChatClient.create",
        new=AsyncMock(return_value="completion"),
    ) as create:
        await client.create(
            model="model",
            messages=[ChatMessage.user_message("hi")],
        )

    assert create.await_args.kwargs["extra_body"] == {
        "agent_class_id": "class-1",
    }


@pytest.mark.asyncio
async def test_agent_replay_can_be_set_per_request():
    from motus.mars import MarsOpenAIChatClient

    replay = {
        "schema_version": 1,
        "trace_id": "trace-1",
        "turn_index": 0,
        "planned_tools": [{"name": "bash", "args": {}, "duration_ms": 10}],
    }
    client = MarsOpenAIChatClient(api_key="test", agent_instance_id="agent-1")

    with patch(
        "motus.models.openai_client.OpenAIChatClient.create",
        new=AsyncMock(return_value="completion"),
    ) as create:
        await client.create(
            model="model",
            messages=[ChatMessage.user_message("hi")],
            extra_body={"agent_replay": replay},
        )

    assert create.await_args.kwargs["extra_body"] == {
        "agent_replay": replay,
        "agent_instance_id": "agent-1",
    }


@pytest.mark.asyncio
async def test_create_non_streaming_merges_mars_fields():
    from motus.mars import MarsOpenAIChatClient

    client = MarsOpenAIChatClient(api_key="test", agent_instance_id="agent-1")
    response = SimpleNamespace(
        id="cmpl-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="length",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=2,
            completion_tokens=3,
            total_tokens=5,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )
    client._client.chat.completions.create = AsyncMock(return_value=response)

    completion = await client.create_non_streaming(
        model="model",
        messages=[ChatMessage.user_message("hi")],
        max_tokens=3,
        extra_body={"ignore_eos": True},
    )

    assert completion.finish_reason == "length"
    assert completion.usage["completion_tokens"] == 3
    request = client._client.chat.completions.create.await_args.kwargs
    assert request["extra_body"] == {
        "ignore_eos": True,
        "agent_instance_id": "agent-1",
    }
