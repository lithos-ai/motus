"""Credential file management for ~/.motus/credentials.json."""

import os
import stat
from pathlib import Path

import httpx
from pydantic import BaseModel

CREDENTIALS_DIR = Path.home() / ".motus"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"


class Credentials(BaseModel):
    cloud_api_url: str
    api_key: str
    key_id: str


def save_credentials(creds: Credentials) -> None:
    """Write credentials to ~/.motus/credentials.json (chmod 600)."""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.write_text(creds.model_dump_json(indent=2) + "\n")
    os.chmod(CREDENTIALS_FILE, stat.S_IRUSR | stat.S_IWUSR)


def load_credentials() -> Credentials | None:
    """Read credentials file, or None if missing/invalid."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        return Credentials.model_validate_json(CREDENTIALS_FILE.read_text())
    except Exception:
        return None


def get_api_key() -> str | None:
    """Return API key from env var or login credentials."""
    if val := os.getenv("LITHOSAI_API_KEY"):
        return val
    creds = load_credentials()
    return creds.api_key if creds else None


def get_api_url() -> str:
    """Return API URL from env var, login credentials, or default."""
    if val := os.getenv("LITHOSAI_API_URL"):
        return val
    creds = load_credentials()
    return creds.cloud_api_url if creds else "https://api.lithosai.cloud"


def clear_credentials() -> None:
    """Delete credentials file."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def _validate_stored_key(creds: Credentials) -> bool:
    """Check whether a stored API key is still accepted by the server.

    Returns True if the key is valid or if the check can't be performed
    (network error, server error, etc.) — we only return False on a
    definitive 401/403 rejection.
    """
    try:
        resp = httpx.get(
            f"{creds.cloud_api_url}/api-keys",
            headers={"Authorization": f"Bearer {creds.api_key}"},
            timeout=5,
        )
        return resp.status_code not in (401, 403)
    except httpx.HTTPError:
        # Network issue — trust stored credentials rather than forcing a
        # re-login that also can't reach the server.
        return True


def ensure_authenticated() -> tuple[str, str]:
    """Return (api_url, api_key), triggering interactive login if needed.

    Checks env vars and credentials file first. If stored credentials
    exist, validates them against the server — if the key was externally
    revoked (e.g. deleted from the web console) the user is seamlessly
    re-authenticated via the OAuth device flow.
    """
    # Env-var credentials are used as-is (CI / deployed agents can't
    # run an interactive login).
    if os.getenv("LITHOSAI_API_KEY"):
        return get_api_url(), get_api_key()

    api_url = get_api_url()
    creds = load_credentials()

    if creds and _validate_stored_key(creds):
        return creds.cloud_api_url, creds.api_key

    # Credentials are missing or rejected — run interactive login.
    import logging

    from motus.auth.login import login

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    if creds:
        print("Stored API key is no longer valid. Re-authenticating...")

    # Best-effort revocation of the old key (will fail if already deleted).
    if creds:
        try:
            httpx.delete(
                f"{creds.cloud_api_url}/api-keys/{creds.key_id}",
                headers={"Authorization": f"Bearer {creds.api_key}"},
                timeout=10,
            )
        except Exception:
            pass

    result = login(api_url)
    new_creds = Credentials(**result)
    save_credentials(new_creds)
    prefix = new_creds.api_key[:12]
    print(f"Logged in to {new_creds.cloud_api_url} ({prefix}...)")
    return new_creds.cloud_api_url, new_creds.api_key
