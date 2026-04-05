"""Credential file management for ~/.motus/credentials.json."""

import os
import stat
from pathlib import Path

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
    val = os.environ.get("LITHOSAI_API_KEY")
    if val:
        return val
    creds = load_credentials()
    return creds.api_key if creds else None


def get_api_url() -> str | None:
    """Return API URL from env var or login credentials."""
    val = os.environ.get("LITHOS_CLOUD_API_URL") or os.environ.get("LITHOSAI_API_URL")
    if val:
        return val
    creds = load_credentials()
    return creds.cloud_api_url if creds else None


def clear_credentials() -> None:
    """Delete credentials file."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()
