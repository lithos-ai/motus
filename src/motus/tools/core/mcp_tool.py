from __future__ import annotations

import asyncio
import contextlib
import inspect
import shlex
from collections.abc import Awaitable, Callable, Iterator
from typing import Self

import httpx
import mcp
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from .sandbox import Sandbox
from .tool import Tool


class MCPTool(Tool):
    def __init__(
        self,
        mcp_session: "MCPSession",
        tool: mcp.Tool,
        input_guardrails: list | None = None,
        output_guardrails: list | None = None,
    ):
        self._mcp_session = mcp_session
        self._mcp_name = tool.name  # original server-side name for RPC
        super().__init__(
            name=tool.name,
            description=tool.description,
            json_schema=tool.inputSchema,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
        )

    async def _invoke(self, **kwargs) -> str:
        """Call the MCP tool via RPC."""
        await self._mcp_session._ensure_connected()
        result = await self._mcp_session._session.call_tool(self._mcp_name, kwargs)
        return result.content[0].text if result.content else ""


class MCPSession:
    """Manages an MCP connection and exposes each remote tool as an MCPTool attribute.

    The constructor is synchronous and only stores connection parameters.
    The connection is established either lazily by the agent on first
    invocation, or explicitly by the user via ``async with``::

        # Lazy — agent connects before first tool call
        agent = Agent(tools=[MCPSession(url="http://...")])

        # Explicit — user manages the lifecycle
        async with MCPSession(url="http://...") as session:
            ...

    In both cases the connection runs on the AgentEngine event loop so
    that tool invocations never cross event-loop boundaries.

    Usage::

        session = MCPSession(url="http://...")
        agent = Agent(tools=tools(session, prefix="browser_"))

        # Or pick individual tools:
        agent = Agent(tools=[tool(input_guardrails=[no_rm])(session.bash)])
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        on_close: Callable[[MCPSession], Awaitable[None] | None] | None = None,
        sandbox: Sandbox | None = None,
        sandbox_command: list[str] | None = None,
        sandbox_env: dict[str, str] | None = None,
        sandbox_port: int | None = None,
        sandbox_path: str = "/mcp",
        **kwargs,
    ) -> None:
        # Validate mutually-exclusive modes.
        modes = sum([url is not None, command is not None, sandbox is not None])
        if modes == 0:
            raise ValueError("Must provide url, command, or sandbox")
        if modes > 1:
            raise ValueError(
                "url, command, and sandbox are mutually exclusive — pick one"
            )
        if sandbox is not None:
            if sandbox_command is None or sandbox_port is None:
                raise ValueError(
                    "sandbox_command and sandbox_port are required with sandbox="
                )
        if command is not None and (headers is not None or http_client is not None):
            raise ValueError(
                "headers and http_client only apply to HTTP (url=) connections"
            )
        self._url = url
        # headers is a convenience shortcut — build an httpx client if needed.
        self._owns_http_client = http_client is None and bool(headers)
        if self._owns_http_client:
            http_client = httpx.AsyncClient(headers=headers)
        self._http_client = http_client
        self._http_kwargs = kwargs if url else {}
        self._server_params = (
            StdioServerParameters(command=command, args=args or [], env=env, **kwargs)
            if command
            else None
        )
        self._sandbox: Sandbox | None = sandbox
        self._sandbox_command: list[str] | None = sandbox_command
        self._sandbox_env: dict[str, str] | None = sandbox_env
        self._sandbox_port: int | None = sandbox_port
        self._sandbox_path: str = sandbox_path
        self._on_close = on_close
        self._tools: dict[str, MCPTool] = {}
        self._session: ClientSession | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._lifecycle_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()

    # --- Mapping-like interface ---
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MCPSession):
            return NotImplemented
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        # Always truthy so ``if session`` checks pass before connection.
        return True

    def __getitem__(self, key: str) -> MCPTool:
        return self._tools[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    # --- Internal helpers ---

    def _populate_tools(self, mcp_tools: list[mcp.types.Tool]) -> None:
        reserved = dir(self)
        self._tools = {t.name: MCPTool(self, t) for t in mcp_tools}
        for name, mcp_tool in self._tools.items():
            attr_name = f"mcptool_{name}" if name in reserved else name
            setattr(self, attr_name, mcp_tool)

    # --- Connection on AgentEngine loop ---

    async def _ensure_connected(self) -> None:
        """Establish the persistent connection if not already connected."""
        if self._session is not None:
            return
        async with self._connect_lock:
            if self._session is not None:
                return
            await self._connect_coro()

    async def _connect_coro(self) -> None:
        """Coroutine that runs on the AgentEngine loop.

        Creates the lifecycle task, waits for the session to be ready,
        and populates tools.
        """
        ready: asyncio.Future[tuple[ClientSession, list[mcp.types.Tool]]] = (
            asyncio.get_running_loop().create_future()
        )
        self._shutdown_event = asyncio.Event()

        if self._sandbox is not None:
            # Sandbox HTTP mode: start MCP server inside the sandbox, then
            # connect over HTTP via the mapped host port.
            env_prefix = " ".join(
                f"{k}={shlex.quote(v)}" for k, v in (self._sandbox_env or {}).items()
            )
            cmd = " ".join(shlex.quote(c) for c in self._sandbox_command)
            full_cmd = f"{env_prefix} {cmd}".strip()
            await self._sandbox.exec("sh", "-c", f"{full_cmd} > /dev/null 2>&1 &")
            base_url = self._sandbox.endpoint(self._sandbox_port)
            url = f"{base_url}{self._sandbox_path}"
            await self._wait_for_http(url)
            self._lifecycle_task = asyncio.create_task(
                self._run_lifecycle(url, ready, self._shutdown_event)
            )
        elif self._url:
            self._lifecycle_task = asyncio.create_task(
                self._run_lifecycle(
                    self._url,
                    ready,
                    self._shutdown_event,
                    http_client=self._http_client,
                    **self._http_kwargs,
                )
            )
        else:
            self._lifecycle_task = asyncio.create_task(
                self._run_stdio_lifecycle(
                    self._server_params, ready, self._shutdown_event
                )
            )

        try:
            session, mcp_tools = await ready
        except BaseException:
            self._shutdown_event.set()
            with contextlib.suppress(BaseException):
                await self._lifecycle_task
            raise

        self._session = session
        self._populate_tools(mcp_tools)

    @staticmethod
    async def _wait_for_http(
        url: str,
        *,
        timeout: float = 30.0,
        interval: float = 0.5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Poll *url* until it accepts a TCP connection."""
        import time

        owns_client = http_client is None
        if owns_client:
            http_client = httpx.AsyncClient()
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    await http_client.get(url, timeout=2.0)
                    return
                except (httpx.ConnectError, httpx.ReadError, httpx.HTTPStatusError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"MCP server at {url} did not become ready "
                            f"within {timeout}s"
                        )
                    await asyncio.sleep(interval)
        finally:
            if owns_client:
                await http_client.aclose()

    # --- Lifecycle ---

    def close(self) -> None:
        """Sync close — safe from any thread.

        Signals the lifecycle task to shut down but does not wait for it.
        Use ``aclose()`` for graceful teardown that awaits transport cleanup.
        """
        if (
            getattr(self, "_shutdown_event", None) is not None
            and self._lifecycle_task is not None
        ):
            loop = self._lifecycle_task.get_loop()
            loop.call_soon_threadsafe(self._shutdown_event.set)
        if getattr(self, "_on_close", None) is not None:
            cb, self._on_close = self._on_close, None  # call once
            result = cb(self)
            # Discard unawaited coroutine if on_close is async
            if result is not None and inspect.iscoroutine(result):
                result.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    async def _aclose_impl(self) -> None:
        """Core async teardown."""
        if self._shutdown_event is not None and self._lifecycle_task is not None:
            self._shutdown_event.set()
            with contextlib.suppress(BaseException):
                await self._lifecycle_task
        if self._on_close is not None:
            cb, self._on_close = self._on_close, None
            result = cb(self)
            if result is not None and inspect.iscoroutine(result):
                await result
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
        self._session = None

    async def aclose(self) -> None:
        """Async close — routes to AgentEngine loop if needed."""
        if self._lifecycle_task is None:
            await self._aclose_impl()
            return
        loop = self._lifecycle_task.get_loop()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            await self._aclose_impl()
        else:
            future = asyncio.run_coroutine_threadsafe(self._aclose_impl(), loop)
            await asyncio.wrap_future(future)

    async def __aenter__(self) -> Self:
        if self._session is None:
            await MCPSession._connect_on_engine(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    # --- Lifecycle tasks (hold transport CMs open) ---

    # The two lifecycle tasks below keep the MCP transport context managers
    # open for the duration of the session.  The pattern is:
    #
    #   1. Open the transport (HTTP or stdio) — this enters the async-with.
    #   2. Handshake: initialize + list_tools, then signal the caller via
    #      ready_future so _connect_coro() can return.
    #   3. Park on shutdown_event.wait() — the transport stays alive as long
    #      as the event is unset.  All tool calls go through this session.
    #   4. When aclose() sets the event, wait() returns, the async-with
    #      blocks exit, and the transport tears down gracefully.

    @staticmethod
    async def _run_lifecycle(
        url: str,
        ready_future: asyncio.Future[tuple[ClientSession, list[mcp.types.Tool]]],
        shutdown_event: asyncio.Event,
        **kwargs,
    ) -> None:
        try:
            async with streamable_http_client(url, **kwargs) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    ready_future.set_result((session, result.tools))
                    await shutdown_event.wait()
        except BaseException as e:
            if not ready_future.done():
                ready_future.set_exception(e)

    @staticmethod
    async def _run_stdio_lifecycle(
        server_params: StdioServerParameters,
        ready_future: asyncio.Future[tuple[ClientSession, list[mcp.types.Tool]]],
        shutdown_event: asyncio.Event,
    ) -> None:
        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    ready_future.set_result((session, result.tools))
                    await shutdown_event.wait()
        except BaseException as e:
            if not ready_future.done():
                ready_future.set_exception(e)

    # --- Async factories (backward compat / standalone usage) ---

    @classmethod
    async def _connect_on_engine(cls, instance: "MCPSession") -> None:
        """Route connection to the AgentEngine loop with lock protection.

        Detects whether the caller is already on the AgentEngine loop to
        avoid deadlock from ``run_coroutine_threadsafe`` + ``wrap_future``.
        """
        from motus.runtime.agent_runtime import get_runtime

        rt = get_runtime()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is rt._loop:
            await instance._ensure_connected()
        else:
            future = asyncio.run_coroutine_threadsafe(
                instance._ensure_connected(), rt._loop
            )
            await asyncio.wrap_future(future)

    @classmethod
    async def connect(
        cls,
        url: str,
        on_close: Callable[["MCPSession"], Awaitable[None] | None] | None = None,
    ) -> Self:
        """Connect to an MCP server. The connection is established on the AgentEngine loop."""
        instance = cls(url=url, on_close=on_close)
        await cls._connect_on_engine(instance)
        return instance

    @classmethod
    async def connect_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        on_close: Callable[["MCPSession"], Awaitable[None] | None] | None = None,
    ) -> Self:
        """Connect a stdio MCP server. The connection is established on the AgentEngine loop."""
        instance = cls(command=command, args=args, env=env, on_close=on_close)
        await cls._connect_on_engine(instance)
        return instance
