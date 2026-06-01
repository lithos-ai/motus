"""motus-aware re-export of the ``mcp`` package.

The submodules here wrap upstream MCP clients to auto-inject motus's
``AUTH`` singleton (``DaprAuth`` when deployed, ``ConsoleAuth`` locally).
Everything not wrapped is re-exported as-is from ``mcp``.
"""

from mcp import ClientSession

__all__ = ["ClientSession"]
