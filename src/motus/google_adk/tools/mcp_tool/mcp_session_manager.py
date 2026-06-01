"""motus-aware re-export of ``google.adk.tools.mcp_tool.mcp_session_manager``.

``StreamableHTTPServerParams`` (and its alias ``StreamableHTTPConnectionParams``)
default the ``httpx_client_factory`` field to ``motus.secrets.httpx.client_factory``,
so every outbound MCP call is authenticated with the motus-selected ``AUTH``
(``DaprAuth`` on the cloud, ``ConsoleAuth`` locally) without the caller
having to wire it up.
"""

from google.adk.tools.mcp_tool.mcp_session_manager import (
    CheckableMcpHttpClientFactory,
)
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams as _Upstream,
)

from motus.secrets.httpx import client_factory


class StreamableHTTPServerParams(_Upstream):
    """Subclass defaulting ``httpx_client_factory`` to motus ``client_factory``."""

    httpx_client_factory: CheckableMcpHttpClientFactory = client_factory


# Upstream defines ``StreamableHTTPServerParams`` as an alias of
# ``StreamableHTTPConnectionParams``; we mirror that.
StreamableHTTPConnectionParams = StreamableHTTPServerParams

__all__ = ["StreamableHTTPServerParams", "StreamableHTTPConnectionParams"]
