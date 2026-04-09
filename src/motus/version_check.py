"""Non-blocking check for newer motus versions on PyPI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PACKAGE = "lithosai-motus"
_CHECK_INTERVAL = 86400  # 24 hours
_TIMEOUT = 1.0  # seconds
_CACHE = Path.home() / ".motus" / "version_check.json"


def _current_version() -> str:
    from importlib.metadata import version

    return version(_PACKAGE)


def _read_cache() -> dict:
    try:
        return json.loads(_CACHE.read_text())
    except Exception:
        return {}


def _write_cache(data: dict) -> None:
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(data))
    except Exception:
        pass


def check_for_update() -> None:
    """Print a message to stderr if a newer version is available on PyPI.

    Checks at most once every 24 hours.  All errors are silently ignored
    so this never blocks or disrupts the CLI.
    """
    try:
        cache = _read_cache()
        now = time.time()

        if now - cache.get("last_check", 0) < _CHECK_INTERVAL:
            # Still within the check interval — use cached result
            latest = cache.get("latest")
            current = _current_version()
            if latest and latest != current:
                _print_update_message(current, latest)
            return

        import urllib.request

        url = f"https://pypi.org/pypi/{_PACKAGE}/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())

        latest = data["info"]["version"]
        current = _current_version()

        _write_cache({"last_check": now, "latest": latest})

        if latest != current:
            _print_update_message(current, latest)

    except Exception:
        pass


def _print_update_message(current: str, latest: str) -> None:
    print(
        f"motus update available: {latest} (current: {current}). "
        f"Run: uv tool upgrade {_PACKAGE}",
        file=sys.stderr,
    )
