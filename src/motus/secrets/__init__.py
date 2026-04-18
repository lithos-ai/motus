"""Secret-store / token helpers shared across motus's httpx.Auth layers.

The submodule ``motus.secrets.httpx`` shadows the third-party ``httpx``
package once imported, so we bind ``URL`` directly to avoid the collision.
"""

from httpx import URL as _URL


def _origin_of(server_url: str) -> str:
    """Return ``scheme://host[:non-default-port]`` for keying per-server state."""
    url = _URL(server_url)
    host = (url.host or "").lower()
    port = url.port
    if (url.scheme == "http" and port in (80, None)) or (
        url.scheme == "https" and port in (443, None)
    ):
        return f"{url.scheme}://{host}"
    return f"{url.scheme}://{host}:{port}"
