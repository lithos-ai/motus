"""Tests for motus.auth.credentials — especially ensure_authenticated recovery."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from motus.auth.credentials import (
    Credentials,
    ensure_authenticated,
    load_credentials,
    save_credentials,
)


@pytest.fixture
def stored_creds(tmp_path, monkeypatch):
    """Write a credential file and point the module at it."""
    creds = Credentials(
        cloud_api_url="https://api.lithosai.cloud",
        api_key="lithos_deadbeef1234567890abcdef1234567890abcdef12345678",
        key_id="a9df0fb6-bae2-4287-93df-daf562335e89",
    )
    monkeypatch.setattr("motus.auth.credentials.CREDENTIALS_DIR", tmp_path)
    monkeypatch.setattr(
        "motus.auth.credentials.CREDENTIALS_FILE", tmp_path / "credentials.json"
    )
    # Clear env vars so file credentials are used
    monkeypatch.delenv("LITHOSAI_API_KEY", raising=False)
    monkeypatch.delenv("LITHOSAI_API_URL", raising=False)
    save_credentials(creds)
    return creds


@pytest.fixture
def fresh_creds():
    """Credentials returned by a successful re-login."""
    return {
        "cloud_api_url": "https://api.lithosai.cloud",
        "api_key": "lithos_freshkey567890abcdef1234567890abcdef1234567890ab",
        "key_id": "6cfdf149-5cbc-4385-8bd2-8cdd523e8b2a",
    }


class TestEnsureAuthenticatedWithValidKey:
    """When the stored key is still valid, return it without re-login."""

    def test_returns_stored_credentials(self, stored_creds, monkeypatch):
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(return_value=httpx.Response(200)),
        )
        api_url, api_key = ensure_authenticated()
        assert api_url == stored_creds.cloud_api_url
        assert api_key == stored_creds.api_key

    def test_does_not_trigger_login(self, stored_creds, monkeypatch):
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(return_value=httpx.Response(200)),
        )
        with patch("motus.auth.login.login") as mock_login:
            ensure_authenticated()
            mock_login.assert_not_called()


class TestEnsureAuthenticatedWithDeletedKey:
    """When the stored key was externally deleted (403), re-authenticate."""

    def test_reauths_on_403(self, stored_creds, fresh_creds, monkeypatch):
        """Core bug reproduction: stored key gets 403, should re-login."""
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(return_value=httpx.Response(403)),
        )
        with patch("motus.auth.login.login", return_value=fresh_creds) as mock_login:
            api_url, api_key = ensure_authenticated()

        mock_login.assert_called_once()
        assert api_key == fresh_creds["api_key"]

    def test_reauths_on_401(self, stored_creds, fresh_creds, monkeypatch):
        """401 (malformed/expired token) should also trigger re-login."""
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(return_value=httpx.Response(401)),
        )
        with patch("motus.auth.login.login", return_value=fresh_creds) as mock_login:
            api_url, api_key = ensure_authenticated()

        mock_login.assert_called_once()
        assert api_key == fresh_creds["api_key"]

    def test_saves_new_credentials(
        self, stored_creds, fresh_creds, tmp_path, monkeypatch
    ):
        """After re-login, the new key should be persisted to disk."""
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(return_value=httpx.Response(403)),
        )
        with patch("motus.auth.login.login", return_value=fresh_creds):
            ensure_authenticated()

        reloaded = load_credentials()
        assert reloaded is not None
        assert reloaded.api_key == fresh_creds["api_key"]
        assert reloaded.key_id == fresh_creds["key_id"]

    def test_old_key_revocation_is_best_effort(
        self, stored_creds, fresh_creds, monkeypatch
    ):
        """If revoking the old key also fails (it will — key is gone), login still proceeds."""
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(return_value=httpx.Response(403)),
        )
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.delete",
            MagicMock(side_effect=httpx.HTTPError("connection refused")),
        )

        with patch("motus.auth.login.login", return_value=fresh_creds):
            api_url, api_key = ensure_authenticated()

        assert api_key == fresh_creds["api_key"]


class TestEnsureAuthenticatedWithNetworkError:
    """If the validation request itself fails (network down), trust stored creds."""

    def test_returns_stored_on_timeout(self, stored_creds, monkeypatch):
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(side_effect=httpx.ConnectError("timeout")),
        )
        api_url, api_key = ensure_authenticated()
        assert api_key == stored_creds.api_key

    def test_returns_stored_on_connection_error(self, stored_creds, monkeypatch):
        monkeypatch.setattr(
            "motus.auth.credentials.httpx.get",
            MagicMock(side_effect=httpx.ConnectError("connection refused")),
        )
        api_url, api_key = ensure_authenticated()
        assert api_key == stored_creds.api_key


class TestEnsureAuthenticatedWithEnvVar:
    """Env var credentials bypass validation (can't re-login in CI)."""

    def test_env_var_skips_validation(self, monkeypatch):
        monkeypatch.setenv("LITHOSAI_API_KEY", "lithos_envvarkey123")
        monkeypatch.setenv("LITHOSAI_API_URL", "https://api.lithosai.cloud")
        mock_get = MagicMock()
        monkeypatch.setattr("motus.auth.credentials.httpx.get", mock_get)
        api_url, api_key = ensure_authenticated()
        mock_get.assert_not_called()
        assert api_key == "lithos_envvarkey123"
