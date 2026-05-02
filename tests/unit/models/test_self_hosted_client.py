"""Tests for SelfHostedChatClient (sglang / vllm OpenAI-compatible servers)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from motus.models import SelfHostedChatClient


class _FakeModelEntry:
    def __init__(self, _id: str) -> None:
        self.id = _id


class _FakeModelList:
    def __init__(self, ids: list[str]) -> None:
        self.data = [_FakeModelEntry(i) for i in ids]


def _make_client(**overrides: Any) -> SelfHostedChatClient:
    kw = {"base_url": "http://localhost:30000"}
    kw.update(overrides)
    return SelfHostedChatClient(**kw)


def test_base_url_normalization_appends_v1():
    client = _make_client(base_url="http://localhost:30000")
    assert str(client._client.base_url).rstrip("/") == "http://localhost:30000/v1"
    assert client._server_root == "http://localhost:30000"


def test_base_url_normalization_keeps_existing_v1():
    client = _make_client(base_url="http://10.0.0.5:8000/v1")
    assert str(client._client.base_url).rstrip("/") == "http://10.0.0.5:8000/v1"
    assert client._server_root == "http://10.0.0.5:8000"


def test_base_url_required():
    with pytest.raises(ValueError):
        SelfHostedChatClient(base_url="")


def test_engine_field_defaults_to_auto():
    assert _make_client().engine == "auto"
    assert _make_client(engine="sglang").engine == "sglang"
    assert _make_client(engine="vllm").engine == "vllm"


def test_default_api_key_is_nonempty():
    """The openai SDK rejects empty api_key; we default to 'EMPTY' so users
    can hit auth-less self-hosted servers without setting any env var."""
    client = _make_client()
    assert client._client.api_key  # truthy


@pytest.mark.asyncio
async def test_list_models_returns_ids():
    client = _make_client()
    client._client.models = AsyncMock()
    client._client.models.list = AsyncMock(
        return_value=_FakeModelList(["/models/Llama-3.1-8B-Instruct"])
    )
    ids = await client.list_models()
    assert ids == ["/models/Llama-3.1-8B-Instruct"]


@pytest.mark.asyncio
async def test_resolve_model_default_returns_first():
    client = _make_client()
    client._client.models = AsyncMock()
    client._client.models.list = AsyncMock(
        return_value=_FakeModelList(["a/Llama", "b/Qwen"])
    )
    assert await client.resolve_model() == "a/Llama"


@pytest.mark.asyncio
async def test_resolve_model_with_hint_picks_match():
    client = _make_client()
    client._client.models = AsyncMock()
    client._client.models.list = AsyncMock(
        return_value=_FakeModelList(["a/Llama-3.1-8B", "b/Qwen-7B"])
    )
    assert await client.resolve_model("qwen") == "b/Qwen-7B"


@pytest.mark.asyncio
async def test_resolve_model_raises_on_no_match():
    client = _make_client()
    client._client.models = AsyncMock()
    client._client.models.list = AsyncMock(
        return_value=_FakeModelList(["a/Llama-3.1-8B"])
    )
    with pytest.raises(RuntimeError):
        await client.resolve_model("qwen")


@pytest.mark.asyncio
async def test_resolve_model_raises_on_empty():
    client = _make_client()
    client._client.models = AsyncMock()
    client._client.models.list = AsyncMock(return_value=_FakeModelList([]))
    with pytest.raises(RuntimeError):
        await client.resolve_model()
