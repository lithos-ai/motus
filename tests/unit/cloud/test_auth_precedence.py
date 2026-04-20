"""Resolved api_key must win over any Authorization forwarded via extra_headers."""

from __future__ import annotations

import httpx
import pytest

from motus.cloud import AsyncClient, Client


def _capture(seen: list[httpx.Request]):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "max_workers": 1,
                "running_workers": 0,
                "total_sessions": 0,
            },
        )

    return handler


def test_constructor_extra_headers_cannot_override_api_key(fresh_env):
    seen: list[httpx.Request] = []
    with Client(
        base_url="http://x",
        api_key="sk-resolved",
        extra_headers={"Authorization": "Bearer sk-forwarded"},
        transport=httpx.MockTransport(_capture(seen)),
    ) as c:
        c.health()
    assert seen[0].headers["authorization"] == "Bearer sk-resolved"


def test_per_call_extra_headers_cannot_override_api_key(fresh_env):
    seen: list[httpx.Request] = []
    with Client(
        base_url="http://x",
        api_key="sk-resolved",
        transport=httpx.MockTransport(_capture(seen)),
    ) as c:
        c.health(extra_headers={"Authorization": "Bearer sk-forwarded"})
    assert seen[0].headers["authorization"] == "Bearer sk-resolved"


def test_env_api_key_overrides_extra_header_authorization(fresh_env):
    fresh_env.setenv("LITHOSAI_API_KEY", "sk-env")
    seen: list[httpx.Request] = []
    with Client(
        base_url="http://x",
        extra_headers={"Authorization": "Bearer sk-forwarded"},
        transport=httpx.MockTransport(_capture(seen)),
    ) as c:
        c.health()
    assert seen[0].headers["authorization"] == "Bearer sk-env"


def test_no_api_key_honors_extra_authorization(fresh_env):
    """If there is NO resolved key, extras' Authorization is used (escape hatch)."""
    seen: list[httpx.Request] = []
    with Client(
        base_url="http://x",
        extra_headers={"Authorization": "Bearer raw-forwarded"},
        transport=httpx.MockTransport(_capture(seen)),
    ) as c:
        c.health()
    assert seen[0].headers["authorization"] == "Bearer raw-forwarded"


@pytest.mark.asyncio
async def test_async_constructor_extra_headers_cannot_override_api_key(fresh_env):
    seen: list[httpx.Request] = []
    async with AsyncClient(
        base_url="http://x",
        api_key="sk-resolved",
        extra_headers={"Authorization": "Bearer sk-forwarded"},
        transport=httpx.MockTransport(_capture(seen)),
    ) as c:
        await c.health()
    assert seen[0].headers["authorization"] == "Bearer sk-resolved"
