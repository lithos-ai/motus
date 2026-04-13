import os

from ...core import Sandbox, SandboxProvider
from .sandbox import CloudSandbox


class CloudSandboxToolProvider(SandboxProvider):
    """Cloud-based sandbox provider.

    Reads ``SANDBOX_URL`` and ``SANDBOX_TOKEN`` from the environment.
    Returns ``None`` from :meth:`get_sandbox` when the env vars are not set,
    allowing a :class:`CompositeToolProvider` to fall through to the next
    provider (e.g. Docker).

    When ``MOTUS_ON_CLOUD=1`` is set, the provider always returns a
    :class:`CloudSandbox` even if the URL/token vars are not yet available.
    The sandbox will resolve them from the environment at first use.
    """

    def __init__(self) -> None:
        self._sandbox: CloudSandbox | None = None

    def get_sandbox(
        self,
        *,
        image: str | None = None,
        dockerfile: str | None = None,
        name: str | None = None,
        env: dict[str, str] | None = None,
        mounts: dict[str, str] | None = None,
        connect: str | None = None,
        ports: dict[int, int | None] | None = None,
    ) -> Sandbox | None:
        url = os.environ.get("SANDBOX_URL")
        token = os.environ.get("SANDBOX_TOKEN")
        on_cloud = os.environ.get("MOTUS_ON_CLOUD") == "1"

        if not on_cloud and (not url or not token):
            return None

        if self._sandbox is None:
            self._sandbox = CloudSandbox(
                url=url or None,
                token=token or None,
            )
        return self._sandbox

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None

    async def aclose(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.aclose()
            self._sandbox = None
