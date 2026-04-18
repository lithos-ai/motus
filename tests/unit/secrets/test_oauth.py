"""Tier-1 tests for ``motus.secrets.oauth``: storage persistence, refresh-token
detection, and the loopback callback server.

Do not exercise the MCP OAuth flow itself — that's a separate integration
effort. Here we verify the building blocks in isolation.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import httpx
import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from motus.secrets import oauth


@pytest.fixture
def tokens_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``TOKENS_FILE`` to a clean tmp path for each test."""
    path = tmp_path / "tokens.json"
    monkeypatch.setattr(oauth, "TOKENS_FILE", path)
    return path


class TestStorage:
    def test_get_returns_none_when_file_missing(self, tokens_file: Path):
        assert not tokens_file.exists()
        storage = oauth._Storage("https://a.example")
        assert storage._get("tokens") is None

    def test_get_returns_none_on_corrupt_json(self, tokens_file: Path):
        tokens_file.parent.mkdir(parents=True, exist_ok=True)
        tokens_file.write_text("{not valid json")
        storage = oauth._Storage("https://a.example")
        assert storage._get("tokens") is None

    def test_put_then_get_roundtrip(self, tokens_file: Path):
        storage = oauth._Storage("https://a.example")
        storage._put("tokens", {"access_token": "abc"})
        assert storage._get("tokens") == {"access_token": "abc"}

    def test_put_creates_parent_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        nested = tmp_path / "deeply" / "nested" / "tokens.json"
        monkeypatch.setattr(oauth, "TOKENS_FILE", nested)
        oauth._Storage("https://a.example")._put("tokens", {"access_token": "abc"})
        assert nested.exists()

    def test_put_file_mode_is_0600(self, tokens_file: Path):
        oauth._Storage("https://a.example")._put("tokens", {"access_token": "abc"})
        assert (tokens_file.stat().st_mode & 0o777) == 0o600

    def test_distinct_origins_are_isolated(self, tokens_file: Path):
        oauth._Storage("https://a.example")._put("tokens", {"access_token": "a"})
        oauth._Storage("https://b.example")._put("tokens", {"access_token": "b"})
        assert oauth._Storage("https://a.example")._get("tokens") == {"access_token": "a"}
        assert oauth._Storage("https://b.example")._get("tokens") == {"access_token": "b"}

    def test_put_preserves_other_keys_in_same_origin(self, tokens_file: Path):
        storage = oauth._Storage("https://a.example")
        storage._put("tokens", {"access_token": "abc"})
        storage._put("client_info", {"client_id": "xyz"})
        assert storage._get("tokens") == {"access_token": "abc"}
        assert storage._get("client_info") == {"client_id": "xyz"}

    def test_concurrent_puts_do_not_lose_writes(self, tokens_file: Path):
        """Hammer ``_put`` from many threads with distinct keys; every key
        must survive. The class-level ``threading.Lock`` is the contract
        under test.
        """
        n = 50
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                oauth._Storage(f"https://o{i}.example")._put("tokens", {"access_token": str(i)})
            except BaseException as exc:  # pragma: no cover — surfaced by assertion below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

        data = json.loads(tokens_file.read_text())
        for i in range(n):
            assert data[f"https://o{i}.example"]["tokens"] == {"access_token": str(i)}

    async def test_get_set_tokens_roundtrip_via_async_api(self, tokens_file: Path):
        storage = oauth._Storage("https://a.example")
        assert await storage.get_tokens() is None
        token = OAuthToken(access_token="abc", refresh_token="ref")
        await storage.set_tokens(token)
        loaded = await storage.get_tokens()
        assert loaded is not None
        assert loaded.access_token == "abc"
        assert loaded.refresh_token == "ref"

    async def test_get_set_client_info_roundtrip_via_async_api(self, tokens_file: Path):
        storage = oauth._Storage("https://a.example")
        assert await storage.get_client_info() is None
        info = OAuthClientInformationFull(
            redirect_uris=["http://127.0.0.1/callback"],
            client_id="client-123",
        )
        await storage.set_client_info(info)
        loaded = await storage.get_client_info()
        assert loaded is not None
        assert loaded.client_id == "client-123"


class TestHasCachedRefreshToken:
    def test_returns_false_when_origin_absent(self, tokens_file: Path):
        assert oauth.has_cached_refresh_token("https://never-seen.example") is False

    def test_returns_false_when_tokens_have_no_refresh_token(self, tokens_file: Path):
        oauth._Storage("https://a.example")._put("tokens", {"access_token": "abc"})
        assert oauth.has_cached_refresh_token("https://a.example") is False

    def test_returns_false_when_refresh_token_is_empty_string(self, tokens_file: Path):
        oauth._Storage("https://a.example")._put(
            "tokens", {"access_token": "abc", "refresh_token": ""}
        )
        assert oauth.has_cached_refresh_token("https://a.example") is False

    def test_returns_true_when_refresh_token_present(self, tokens_file: Path):
        oauth._Storage("https://a.example")._put(
            "tokens", {"access_token": "abc", "refresh_token": "ref"}
        )
        assert oauth.has_cached_refresh_token("https://a.example") is True


class TestStaticBearer:
    def test_get_returns_none_when_unset(self, tokens_file: Path):
        assert oauth.get_static_bearer("https://a.example") is None

    def test_set_then_get_roundtrip(self, tokens_file: Path):
        oauth.set_static_bearer("https://a.example", "tok-abc")
        assert oauth.get_static_bearer("https://a.example") == "tok-abc"

    def test_get_is_origin_scoped(self, tokens_file: Path):
        oauth.set_static_bearer("https://a.example", "tok-a")
        oauth.set_static_bearer("https://b.example", "tok-b")
        assert oauth.get_static_bearer("https://a.example") == "tok-a"
        assert oauth.get_static_bearer("https://b.example") == "tok-b"

    def test_set_replaces_previous(self, tokens_file: Path):
        oauth.set_static_bearer("https://a.example", "first")
        oauth.set_static_bearer("https://a.example", "second")
        assert oauth.get_static_bearer("https://a.example") == "second"

    def test_get_returns_none_when_entry_malformed(self, tokens_file: Path):
        # Simulate legacy / corrupt entry that isn't a dict.
        oauth._Storage("https://a.example")._put("static_bearer", {"other_key": "x"})
        assert oauth.get_static_bearer("https://a.example") is None


class TestCallbackServer:
    async def test_callback_handler_returns_code_and_state(self):
        server = oauth._CallbackServer()
        try:
            # Start waiting for the callback first; then fire the browser-like
            # request concurrently so ``handle_request`` unblocks.
            handler_task = asyncio.create_task(server.callback_handler())

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{server.redirect_uri}?code=the-code&state=the-state"
                )
                assert response.status_code == 200

            code, state = await handler_task
            assert code == "the-code"
            assert state == "the-state"
        finally:
            server._server.server_close()

    async def test_callback_handler_defaults_missing_params_to_empty_and_none(self):
        server = oauth._CallbackServer()
        try:
            handler_task = asyncio.create_task(server.callback_handler())

            async with httpx.AsyncClient() as client:
                response = await client.get(server.redirect_uri)
                assert response.status_code == 200

            code, state = await handler_task
            assert code == ""  # SDK expects a str, not None
            assert state is None
        finally:
            server._server.server_close()

    def test_redirect_uri_binds_loopback_ephemeral_port(self):
        server = oauth._CallbackServer()
        try:
            assert server.redirect_uri.startswith("http://127.0.0.1:")
            assert server.redirect_uri.endswith("/callback")
        finally:
            server._server.server_close()
