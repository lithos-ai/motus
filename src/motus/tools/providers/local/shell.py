from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from typing import Mapping, Self

from ...core import Sandbox


class LocalShell(Sandbox):
    """Local subprocess-based sandbox.

    Runs commands directly on the host via ``subprocess``.
    Useful when no container runtime is available or needed.

    Usage::

        with LocalShell() as sh:
            sh.sh("echo hi").af_result()

        # With explicit working directory
        with LocalShell(cwd="/tmp/work") as sh:
            sh.sh("pwd").af_result()

        # As agent tools
        agent = Agent(tools=[LocalShell()])
    """

    def __init__(self, *, cwd: str | None = None) -> None:
        self._cwd = cwd or os.getcwd()

    @classmethod
    def create(cls, image: str = "python:3.12", **kwargs) -> Self:
        """Create a LocalShell. The *image* parameter is ignored."""
        return cls(**{k: v for k, v in kwargs.items() if k == "cwd"})

    @classmethod
    def connect(cls, identifier: str, **kwargs) -> Self:
        """Treat *identifier* as a working directory path."""
        return cls(cwd=identifier, **kwargs)

    async def exec(
        self,
        *cmd: str,
        input: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Execute a command locally and return combined stdout+stderr."""
        merged_env = {**os.environ, **(env or {})}
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            input=input,
            capture_output=True,
            text=True,
            cwd=cwd or self._cwd,
            env=merged_env,
        )
        # Match DockerSandbox behaviour: non-zero exit returns output, no raise.
        if proc.returncode != 0:
            return (proc.stdout + proc.stderr).rstrip("\n")
        return proc.stdout

    async def put(self, local_path: str, sandbox_path: str) -> None:
        """Copy a file from *local_path* to *sandbox_path* (same filesystem)."""
        await asyncio.to_thread(shutil.copy2, local_path, sandbox_path)

    async def get(self, sandbox_path: str, local_path: str) -> str:
        """Copy a file from *sandbox_path* to *local_path*.

        If *local_path* is a directory the file keeps its original name.
        Returns the resolved destination path.
        """
        if os.path.isdir(local_path):
            local_path = os.path.join(local_path, os.path.basename(sandbox_path))
        await asyncio.to_thread(shutil.copy2, sandbox_path, local_path)
        return local_path

    def close(self) -> None:
        """No-op — nothing to tear down for local processes."""
