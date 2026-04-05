from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from .sandbox import Sandbox
    from .tool import Tools


class _ProviderLifecycle:
    """Mixin providing default lifecycle and context-manager methods."""

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()


class MCPProvider(_ProviderLifecycle, ABC):
    """Provides tools (MCP protocol concept)."""

    @abstractmethod
    async def list_services(self) -> list[str]: ...

    @abstractmethod
    async def get_tools(self, service_name: str) -> Tools | None: ...


class SandboxProvider(_ProviderLifecycle, ABC):
    """Provides execution sandboxes."""

    @abstractmethod
    def get_sandbox(
        self,
        *,
        image: str = "python:3.12",
        dockerfile: str | None = None,
        name: str | None = None,
        env: dict[str, str] | None = None,
        mounts: dict[str, str] | None = None,
        connect: str | None = None,
        ports: dict[int, int | None] | None = None,
    ) -> Sandbox | None: ...
