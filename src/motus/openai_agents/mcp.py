"""motus-aware re-export of ``agents.mcp``.

``MCPServerStreamableHttp`` here auto-injects motus's ``AUTH`` singleton
into the params dict when the caller hasn't supplied their own auth.
``MCPServerSse`` does the same. Everything else is re-exported as-is.
"""

from agents.mcp import (  # noqa: F401
    MCPServer,
    MCPServerSseParams,
    MCPServerStdio,
    MCPServerStdioParams,
    MCPServerStreamableHttpParams,
    MCPUtil,
)
from agents.mcp import (
    MCPServerSse as _MCPServerSse,
)
from agents.mcp import (
    MCPServerStreamableHttp as _MCPServerStreamableHttp,
)

from motus.secrets.httpx import AUTH


class MCPServerStreamableHttp(_MCPServerStreamableHttp):
    """Subclass that injects ``motus.auth.AUTH`` into params by default."""

    def __init__(self, params: MCPServerStreamableHttpParams, *args, **kwargs):
        params = {**params}
        params.setdefault("auth", AUTH)
        super().__init__(params, *args, **kwargs)


class MCPServerSse(_MCPServerSse):
    """Subclass that injects ``motus.auth.AUTH`` into params by default."""

    def __init__(self, params: MCPServerSseParams, *args, **kwargs):
        params = {**params}
        params.setdefault("auth", AUTH)
        super().__init__(params, *args, **kwargs)


__all__ = [
    "MCPServer",
    "MCPServerSse",
    "MCPServerSseParams",
    "MCPServerStdio",
    "MCPServerStdioParams",
    "MCPServerStreamableHttp",
    "MCPServerStreamableHttpParams",
    "MCPUtil",
]
