from contextlib import AsyncExitStack, ExitStack
from typing import Self, Sequence, Union

from .sandbox import Sandbox
from .tool import Tools
from .tool_provider import MCPProvider, SandboxProvider


class CompositeToolProvider(MCPProvider, SandboxProvider):
    def __init__(
        self, providers: Sequence[Union[MCPProvider, SandboxProvider]]
    ) -> None:
        self.providers = providers
        self._active: bool = False
        self._exit_stack: ExitStack | None = None
        self._async_exit_stack: AsyncExitStack | None = None

    async def list_services(self) -> list[str]:
        if not self._active:
            raise RuntimeError(
                "CompositeToolProvider must be used as a context manager"
            )

        services: set[str] = set()
        for provider in self.providers:
            if isinstance(provider, MCPProvider):
                services.update(await provider.list_services())
        return list(services)

    async def get_tools(self, service_name: str) -> Tools | None:
        if not self._active:
            raise RuntimeError(
                "CompositeToolProvider must be used as a context manager"
            )

        for provider in self.providers:
            if isinstance(provider, MCPProvider):
                if (tools := await provider.get_tools(service_name)) is not None:
                    return tools

        raise KeyError(f"Service '{service_name}' not found")

    def get_sandbox(self, **kwargs) -> Sandbox:
        if not self._active:
            raise RuntimeError(
                "CompositeToolProvider must be used as a context manager"
            )

        for provider in self.providers:
            if isinstance(provider, SandboxProvider):
                if (sandbox := provider.get_sandbox(**kwargs)) is not None:
                    return sandbox

        raise RuntimeError("No sandbox provider available")

    # --- Context managers ---

    def __enter__(self) -> Self:
        stack = ExitStack()
        stack.__enter__()
        for provider in self.providers:
            stack.enter_context(provider)
        self._exit_stack = stack
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        self._active = False
        assert self._exit_stack is not None
        return self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Self:
        stack = AsyncExitStack()
        await stack.__aenter__()
        for provider in self.providers:
            await stack.enter_async_context(provider)
        self._async_exit_stack = stack
        self._active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        self._active = False
        assert self._async_exit_stack is not None
        return await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
