import asyncio
from abc import ABC, abstractmethod
from typing import Mapping, Self

from .decorators import tools


@tools(allowlist={"python", "sh"})
class Sandbox(ABC):
    """Execution environment with lifecycle management.

    Core primitive: exec(). Everything else is built on top.

    Entry points::

        acreate/create  — create a new sandbox (abstract, subclass must implement)
        connect         — attach to an existing sandbox (optional, default raises)

    Lifecycle patterns::

        # 1. Async
        async with await SandboxImpl.acreate(image) as sb: ...

        # 2. Sync
        with SandboxImpl.create(image) as sb: ...

        # 3. Manual
        sb = SandboxImpl.create(image); ...; sb.close()

        # 4. Connect to existing
        with SandboxImpl.connect(identifier) as sb: ...

    Dual identity — same instance works as both:
        - A Python class (developer calls methods directly)
        - A tool collection (pass to Agent, normalize_tools extracts methods)
    """

    # --- Entry points ---

    @classmethod
    @abstractmethod
    def create(
        cls,
        image: str = "python:3.12",
        *,
        ports: dict[int, int | None] | None = None,
        **kwargs,
    ) -> Self:
        """Sync factory. Subclasses must implement.

        Args:
            ports: Container-port to host-port mapping.
                ``{8080: None}`` means "map container port 8080 to a random
                host port".  ``None`` values are resolved after creation.
        """
        ...

    @classmethod
    async def acreate(
        cls,
        image: str = "python:3.12",
        *,
        ports: dict[int, int | None] | None = None,
        **kwargs,
    ) -> Self:
        """Async factory. Default wraps create() in a thread.
        Subclasses can override for native async implementation."""
        return await asyncio.to_thread(cls.create, image, ports=ports, **kwargs)

    @classmethod
    def connect(cls, identifier: str, **kwargs) -> Self:
        """Connect to an existing sandbox. Not all backends support this."""
        raise NotImplementedError(f"{cls.__name__} does not support connect()")

    # --- Abstract: core primitives ---

    @abstractmethod
    async def exec(
        self,
        *cmd: str,
        input: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Execute a command in the sandbox and return its output."""
        ...

    def endpoint(self, port: int) -> str:
        """Return the URL to reach a service listening on *port* inside the sandbox."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support port mapping"
        )

    # --- Convenience wrappers (built on exec) ---

    async def python(self, script: str) -> str:
        """Execute a Python script in the sandbox and return its output."""
        return await self.exec("python3", "-c", script)

    async def sh(self, command: str) -> str:
        """Execute a shell command in the sandbox and return its output."""
        return await self.exec("sh", "-c", command)

    @abstractmethod
    async def put(self, local_path: str, sandbox_path: str) -> None:
        """Copy a file from the host to the sandbox."""
        ...

    @abstractmethod
    async def get(self, sandbox_path: str, local_path: str) -> str:
        """Copy a file from the sandbox to the host.

        If local_path is a directory, the file is saved there with its
        original basename. Returns the resolved local file path.
        """
        ...

    # --- Lifecycle ---

    @abstractmethod
    def close(self) -> None:
        """Sync cleanup. Subclasses must implement."""
        ...

    async def aclose(self) -> None:
        """Async cleanup. Default wraps close() in a thread.
        Subclasses can override for native async implementation."""
        await asyncio.to_thread(self.close)

    # --- Context managers ---

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
