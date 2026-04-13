import os
from typing import Mapping, Self

import httpx

from ...core import Sandbox


class CloudSandbox(Sandbox):
    """Cloud-based sandbox backed by the sandbox REST API.

    Connects to a pre-provisioned sandbox identified by URL and bearer token.
    Does NOT manage the sandbox lifecycle (create/delete/pause).

    When *url* or *token* are omitted the sandbox reads
    ``SANDBOX_URL`` / ``SANDBOX_TOKEN`` from the environment on every
    request.  This allows construction before the env vars are set (e.g.
    at module-import time under ``motus serve``).

    Usage::

        async with CloudSandbox(url="https://...", token="...") as sb:
            output = await sb.sh("echo hello")
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        token: str | None = None,
    ) -> None:
        self._url = url.rstrip("/") if url else None
        self._token = token
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    def _get_url(self) -> str:
        url = self._url or os.environ.get("SANDBOX_URL")
        if not url:
            raise RuntimeError(
                "Sandbox URL not available. Set the SANDBOX_URL environment "
                "variable or pass url= to CloudSandbox()."
            )
        return url.rstrip("/")

    def _get_auth_headers(self) -> dict[str, str]:
        token = self._token or os.environ.get("SANDBOX_TOKEN")
        if not token:
            raise RuntimeError(
                "Sandbox token not available. Set the SANDBOX_TOKEN environment "
                "variable or pass token= to CloudSandbox()."
            )
        return {"Authorization": f"Bearer {token}"}

    @classmethod
    def create(cls, image: str | None = None, **kwargs) -> Self:
        raise NotImplementedError(
            "CloudSandbox does not support create(). "
            "Use CloudSandboxToolProvider or pass url and token directly."
        )

    async def exec(
        self,
        *cmd: str,
        input: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        body: dict = {"command": list(cmd)}
        if cwd is not None:
            body["working_directory"] = cwd
        if env is not None:
            body["env"] = dict(env)
        if input is not None:
            body["stdin"] = input
        body["timeout"] = 300

        resp = await self._client.post(
            f"{self._get_url()}/exec",
            headers=self._get_auth_headers(),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

        return data.get("stdout", "") + data.get("stderr", "")

    async def put(self, local_path: str, sandbox_path: str) -> None:
        if not sandbox_path.startswith("/"):
            raise ValueError("Target path must be absolute")
        with open(local_path, "rb") as f:
            content = f.read()
        resp = await self._client.put(
            f"{self._get_url()}/files/{sandbox_path.lstrip('/')}",
            headers=self._get_auth_headers(),
            content=content,
        )
        resp.raise_for_status()

    async def get(self, sandbox_path: str, local_path: str) -> str:
        if not sandbox_path.startswith("/"):
            raise ValueError("Source path must be absolute")
        if os.path.isdir(local_path):
            local_path = os.path.join(local_path, os.path.basename(sandbox_path))

        resp = await self._client.get(
            f"{self._get_url()}/files/{sandbox_path.lstrip('/')}",
            headers=self._get_auth_headers(),
        )
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

        return local_path

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        await self._client.aclose()
