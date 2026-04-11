from __future__ import annotations

import atexit

from .builtins import BuiltinTools, builtin_tools
from .core import (
    AgentTool,
    CompositeToolProvider,
    DictTools,
    FunctionTool,
    InputSchema,
    MCPProvider,
    MCPSession,
    MCPTool,
    Parameters,
    ReturnType,
    Sandbox,
    SandboxProvider,
    Tool,
    Tools,
    normalize_tools,
    tool,
    tools,
    tools_from,
)
from .providers import (
    CloudSandbox,
    CloudSandboxToolProvider,
    DockerSandbox,
    DockerToolProvider,
    LocalShell,
    WebSearchTool,
)

__all__ = [
    # Core
    "AgentTool",
    "CompositeToolProvider",
    "DictTools",
    "FunctionTool",
    "InputSchema",
    "MCPProvider",
    "MCPSession",
    "MCPTool",
    "normalize_tools",
    "Parameters",
    "ReturnType",
    "Sandbox",
    "SandboxProvider",
    "Tool",
    "tool",
    "tools",
    "Tools",
    "tools_from",
    # Providers
    "CloudSandbox",
    "CloudSandboxToolProvider",
    "DockerSandbox",
    "DockerToolProvider",
    "LocalShell",
    "WebSearchTool",
    # Builtins
    "BuiltinTools",
    "builtin_tools",
    # Convenience
    "DEFAULT_TOOL_PROVIDER",
    "get_mcp",
    "get_sandbox",
]


def DEFAULT_TOOL_PROVIDER():
    return CompositeToolProvider([CloudSandboxToolProvider(), DockerToolProvider()])


# ---------------------------------------------------------------------------
# Global provider singleton — lazily initialized, cleaned up at exit.
# ---------------------------------------------------------------------------

_provider: SandboxProvider | None = None


def _get_default_provider() -> SandboxProvider:
    """Lazily initialize and return the global default provider."""
    global _provider
    if _provider is None:
        _provider = DEFAULT_TOOL_PROVIDER()
        _provider.__enter__()
        atexit.register(_shutdown_provider)
    return _provider


def _shutdown_provider() -> None:
    """Best-effort cleanup of the global provider at interpreter exit."""
    global _provider
    if _provider is not None:
        p = _provider
        _provider = None
        p.close()


def get_sandbox(
    *,
    image: str = "python:3.12",
    dockerfile: str | None = None,
    name: str | None = None,
    env: dict[str, str] | None = None,
    mounts: dict[str, str] | None = None,
    connect: str | None = None,
    ports: dict[int, int | None] | None = None,
) -> Sandbox:
    """Create a sandbox (or connect to an existing container).

    See ``examples/omni/gpuos_omni_demo.py`` for a full working example.

    Usage::

        with get_sandbox() as sb:
            await sb.sh("echo hi")

        with get_sandbox(image="node:20") as sb: ...
        with get_sandbox(mounts={"/local": "/workspace"}) as sb: ...
        with get_sandbox(connect="my-container") as sb: ...
        with get_sandbox(image="node:20", ports={8080: None}) as sb: ...
    """
    provider = _get_default_provider()
    return provider.get_sandbox(
        image=image,
        dockerfile=dockerfile,
        name=name,
        env=env,
        mounts=mounts,
        connect=connect,
        ports=ports,
    )


def get_mcp(
    *,
    url: str | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    image: str | None = None,
    sandbox: Sandbox | None = None,
    port: int = 8080,
    **kwargs,
) -> MCPSession:
    """Create an MCPSession.

    Usage::

        # Pure HTTP
        async with get_mcp(url="http://...") as session: ...

        # Pure stdio (local process)
        async with get_mcp(command="npx", args=[...]) as session: ...

        # Sandbox — server uses --port flag (e.g. Playwright)
        get_mcp(image="node:20", command="npx",
                args=["@playwright/mcp", "--port", "8080"], port=8080)

        # Sandbox — server uses PORT env var (e.g. server-everything)
        get_mcp(image="node:20", command="npx",
                args=["@modelcontextprotocol/server-everything", "streamableHttp"],
                env={"PORT": "3000"}, port=3000)

    In sandbox mode the caller is responsible for configuring the MCP server to
    listen on ``port`` (via ``args`` or ``env``).  ``get_mcp`` only maps the
    container port and connects to it. (In the future, this will be config logic)
    """
    if sandbox is not None or image is not None:
        if url is not None:
            raise ValueError("Cannot provide url together with image= or sandbox=")
        if command is None:
            raise ValueError(
                "command must be provided when using image= or sandbox= with get_mcp()"
            )
        owns_sandbox = sandbox is None
        if owns_sandbox:
            sandbox = get_sandbox(
                image=image or "python:3.12", env=env, ports={port: None}
            )
        sandbox_command = [command, *(args or [])]
        return MCPSession(
            sandbox=sandbox,
            sandbox_command=sandbox_command,
            sandbox_env=env,
            sandbox_port=port,
            on_close=(lambda _: sandbox.close()) if owns_sandbox else None,
            **kwargs,
        )
    return MCPSession(
        url=url, command=command, args=args, env=env, headers=headers, **kwargs
    )
