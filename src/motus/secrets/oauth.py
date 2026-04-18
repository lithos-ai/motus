"""Loopback OAuth callback server and per-origin token storage for the
MCP-spec OAuth flow (discovery → DCR → auth code + PKCE → token exchange).

:func:`open_auth` yields an ``OAuthClientProvider`` with a loopback callback
server bound — use when an auth code grant is likely (no cached refresh
token). :func:`make_provider` returns a provider without a callback server
— use when a cached refresh token should cover the flow.
:class:`motus.secrets.httpx._ConsoleAuth` owns the per-origin mapping and
decides which entry point to call.
"""

import asyncio
import http.server
import json
import sys
import threading
import urllib.parse
import webbrowser
from contextlib import closing, contextmanager

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from ..auth.credentials import CREDENTIALS_DIR

TOKENS_FILE = CREDENTIALS_DIR / "tokens.json"
CLIENT_NAME = "motus-cli"


class _HTTPServer(http.server.HTTPServer):
    data: tuple[str | None, str | None] | None = None


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _HTTPServer

    def do_GET(self):  # noqa: N802 — stdlib contract
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self.server.data = (
            (params.get("code") or [None])[0],
            (params.get("state") or [None])[0],
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<h3>Authentication complete. You may close this tab.</h3>"
            b"<script>window.close();</script>"
        )

    def log_message(self, *args, **kwargs):
        pass


class _CallbackServer:
    """Loopback HTTP server bound to a fresh ephemeral port.

    ``redirect_uri`` is usable as ``OAuthClientMetadata.redirect_uris[0]``;
    ``callback_handler`` matches the MCP SDK's ``callback_handler`` signature
    and blocks off the event loop until the browser hits ``/callback``,
    returning ``(code, state)``.
    """

    def __init__(self):
        self._server = _HTTPServer(("127.0.0.1", 0), _Handler)
        self._server.timeout = 300
        host, port = self._server.server_address[:2]
        self.redirect_uri = f"http://{host}:{port}/callback"

    async def callback_handler(self) -> tuple[str, str | None]:
        self._server.data = None
        await asyncio.to_thread(self._server.handle_request)
        code, state = self._server.data or (None, None)
        return code or "", state

    def close(self) -> None:
        self._server.server_close()


async def _open_browser(url: str) -> None:
    if not await asyncio.to_thread(webbrowser.open, url):
        print(f"[motus] Visit: {url}", file=sys.stderr)


class _Storage(TokenStorage):
    _lock = threading.Lock()

    def __init__(self, origin: str):
        self._origin = origin

    def _get(self, key: str) -> dict | None:
        try:
            return json.loads(TOKENS_FILE.read_text()).get(self._origin, {}).get(key)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def _put(self, key: str, value: dict) -> None:
        with self._lock:
            try:
                data = json.loads(TOKENS_FILE.read_text())
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                data = {}
            data.setdefault(self._origin, {})[key] = value
            TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
            TOKENS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            TOKENS_FILE.chmod(0o600)

    async def get_tokens(self) -> OAuthToken | None:
        raw = await asyncio.to_thread(self._get, "tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await asyncio.to_thread(self._put, "tokens", tokens.model_dump(mode="json"))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = await asyncio.to_thread(self._get, "client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(self._put, "client_info", info.model_dump(mode="json"))


@contextmanager
def open_auth(origin: str):
    """Yield an ``OAuthClientProvider`` with a loopback callback server bound.

    The server is closed on context exit. Use for origins likely to need an
    auth code grant (no cached refresh token on disk); the caller should add
    this to an exit stack so the socket is closed at shutdown.
    """
    with closing(_CallbackServer()) as server:
        yield OAuthClientProvider(
            server_url=origin,
            client_metadata=OAuthClientMetadata(
                client_name=CLIENT_NAME,
                redirect_uris=[server.redirect_uri],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
            ),
            storage=_Storage(origin),
            redirect_handler=_open_browser,
            callback_handler=server.callback_handler,
        )


def make_provider(origin: str) -> OAuthClientProvider:
    """Build an ``OAuthClientProvider`` without a callback server, for origins
    with a cached refresh token where the refresh path should cover the flow.
    """
    return OAuthClientProvider(
        server_url=origin,
        client_metadata=OAuthClientMetadata(
            client_name=CLIENT_NAME,
            redirect_uris=["http://127.0.0.1/placeholder"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        ),
        storage=_Storage(origin),
        redirect_handler=_open_browser,
    )


def has_cached_refresh_token(origin: str) -> bool:
    """True if ``origin`` has a refresh token persisted on disk."""
    tokens = _Storage(origin)._get("tokens")
    return bool(tokens and tokens.get("refresh_token"))


def get_static_bearer(origin: str) -> str | None:
    """Return a user-provided bearer token previously cached for ``origin``.

    Set by :func:`set_static_bearer` after an OAuth failure so that future
    flows short-circuit to the bearer instead of re-attempting OAuth.
    """
    entry = _Storage(origin)._get("static_bearer")
    return entry.get("token") if isinstance(entry, dict) else None


def set_static_bearer(origin: str, bearer: str) -> None:
    """Persist a user-provided bearer token for ``origin``."""
    _Storage(origin)._put("static_bearer", {"token": bearer})
