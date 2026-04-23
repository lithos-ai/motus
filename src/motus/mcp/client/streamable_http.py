"""Motus-aware Streamable HTTP client for MCP.

Wraps ``mcp.client.streamable_http.streamable_http_client`` so the transport
always runs through an ``httpx.AsyncClient`` authenticated with motus's
``AUTH`` singleton — ``ConsoleAuth`` locally (browser OAuth / getpass
fallback), ``DaprAuth`` when a daprd sidecar is present.
"""

from contextlib import asynccontextmanager

from mcp.client.streamable_http import streamable_http_client as _upstream

from motus.secrets.httpx import client_factory


@asynccontextmanager
async def streamable_http_client(url: str, *, terminate_on_close: bool = True):
    """Drop-in replacement for ``mcp.client.streamable_http.streamable_http_client``.

    Yields the same ``(read, write, get_session_id)`` triple; the underlying
    httpx client carries motus auth.
    """
    async with (
        client_factory() as http_client,
        _upstream(
            url,
            http_client=http_client,
            terminate_on_close=terminate_on_close,
        ) as streams,
    ):
        yield streams
