"""httpx.Auth selector plus the Dapr and Console implementations.

``AUTH`` picks at import time: ``_DaprAuth`` if a daprd sidecar is present
(``DAPR_HTTP_PORT`` set), else ``_ConsoleAuth`` which caches one MCP
``OAuthClientProvider`` per origin and runs the OAuth flow per request.
"""

import asyncio
import atexit
import getpass
import os
import sys
import urllib.parse
from contextlib import ExitStack

import httpx

from mcp.client.auth import OAuthClientProvider

from .oauth import (
    get_static_bearer,
    has_cached_refresh_token,
    make_provider,
    open_auth,
    set_static_bearer,
)

_DAPR_HTTP_PORT = os.getenv("DAPR_HTTP_PORT")


def _origin_of(server_url: str) -> str:
    """Return ``scheme://host[:non-default-port]`` for keying per-server state."""
    url = httpx.URL(server_url)
    host = (url.host or "").lower()
    port = url.port
    if (url.scheme == "http" and port in (80, None)) or (
        url.scheme == "https" and port in (443, None)
    ):
        return f"{url.scheme}://{host}"
    return f"{url.scheme}://{host}:{port}"


class _DaprAuth(httpx.Auth):
    """Fetch the bearer for each outbound request from a Dapr secret store.

    Keyed by ``urllib.parse.quote(origin, safe="")``. The value may be an
    OAuth blob (``{"access_token": ...}``), a ``{"token": ...}`` entry, or
    a plain-string secret Dapr returns as a single-key dict.
    """

    requires_response_body = True

    def __init__(self):
        self._secret_store = os.getenv("MOTUS_SECRET_STORE")
        self._tokens: dict[str, str] = {}

    def auth_flow(self, request):
        origin = _origin_of(str(request.url))
        token = self._tokens.get(origin)
        if token is None:
            token = yield from self._fetch(origin)
            if token:
                self._tokens[origin] = token
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if response.status_code == 401 and token:
            self._tokens.pop(origin, None)
            fresh = yield from self._fetch(origin)
            if fresh and fresh != token:
                self._tokens[origin] = fresh
                request.headers["Authorization"] = f"Bearer {fresh}"
                yield request

    def _fetch(self, origin: str):
        if not _DAPR_HTTP_PORT:
            raise RuntimeError("DAPR_HTTP_PORT is not set; DaprAuth requires a daprd sidecar.")
        name = urllib.parse.quote(origin, safe="")
        url = f"http://localhost:{_DAPR_HTTP_PORT}/v1.0/secrets/{self._secret_store}/{name}"
        response = yield httpx.Request("GET", url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        return (
            body.get("access_token")
            or body.get("token")
            or (next(iter(body.values())) if len(body) == 1 else None)
        )


class _ConsoleAuth(httpx.Auth):
    """Caches one :class:`OAuthClientProvider` per origin, guarded by a
    per-origin lock. First sight of an origin without a cached refresh
    token binds a loopback callback server via :func:`open_auth` and adds
    it to an ``ExitStack`` closed at interpreter shutdown; origins with a
    cached refresh token get a lighter provider (no server) from
    :func:`make_provider`. If the OAuth flow raises (server lacks DCR,
    no AS metadata, etc.), falls back to a ``getpass`` bearer prompt and
    persists the result so subsequent flows short-circuit to it.
    """

    requires_response_body = True

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._providers: dict[str, OAuthClientProvider] = {}
        self._bearers: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        atexit.register(self._stack.close)

    async def async_auth_flow(self, request):
        origin = _origin_of(str(request.url))
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            bearer = self._bearers.get(origin) or get_static_bearer(origin)
            if bearer is not None:
                self._bearers[origin] = bearer
                request.headers["Authorization"] = f"Bearer {bearer}"
                yield request
                return

            if origin not in self._providers:
                if has_cached_refresh_token(origin):
                    self._providers[origin] = make_provider(origin)
                else:
                    self._providers[origin] = self._stack.enter_context(open_auth(origin))
            try:
                async for r in self._providers[origin].async_auth_flow(request):
                    yield r
            except Exception as exc:
                bearer = await self._prompt_bearer(origin, exc)
                request.headers["Authorization"] = f"Bearer {bearer}"
                yield request

    async def _prompt_bearer(self, origin: str, exc: Exception) -> str:
        print(
            f"[motus] OAuth failed for {origin}: {exc}; prompting for bearer token.",
            file=sys.stderr,
        )
        bearer = (
            await asyncio.to_thread(getpass.getpass, f"Bearer token for {origin}: ")
        ).strip()
        if not bearer:
            raise RuntimeError(f"No bearer token provided for {origin}.") from exc
        set_static_bearer(origin, bearer)
        self._bearers[origin] = bearer
        return bearer

    def sync_auth_flow(self, request):
        origin = _origin_of(str(request.url))
        token = getpass.getpass(f"Bearer token for {origin}: ").strip()
        if not token:
            raise RuntimeError(f"No bearer token provided for {origin}.")
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


AUTH = _DaprAuth() if _DAPR_HTTP_PORT else _ConsoleAuth()


def client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Factory matching ``mcp.shared._httpx_utils.McpHttpClientFactory``."""
    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth or AUTH)
