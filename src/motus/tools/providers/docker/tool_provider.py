import logging
import os
from typing import Set

import docker
from docker.errors import DockerException

from ...core import Sandbox, SandboxProvider
from .sandbox import SANDBOX_IMAGE, DockerSandbox


class DockerToolProvider(SandboxProvider):
    """Docker-based sandbox provider.

    Manages Docker containers for sandbox creation. Docker interaction is
    deferred until the first :meth:`get_sandbox` call so that constructing
    a ``CompositeToolProvider`` that includes this provider never crashes
    when Docker is unavailable.
    """

    def __init__(self) -> None:
        self.client: docker.DockerClient | None = None
        self.sandboxes: Set[DockerSandbox] = set()

    def _ensure_client(self) -> bool:
        """Lazily connect to Docker and ensure the sandbox image exists.

        Returns ``True`` if Docker is ready, ``False`` otherwise.
        """
        if self.client is not None:
            return True

        try:
            client = docker.from_env()
        except DockerException:
            logging.warning("Docker is not available — skipping DockerToolProvider")
            return False

        if not any(
            any(
                tag == SANDBOX_IMAGE or tag.startswith(SANDBOX_IMAGE + ":")
                for tag in image.tags
            )
            for image in client.images.list()
        ):
            logging.info(
                f"Sandbox image '{SANDBOX_IMAGE}' not found locally, building..."
            )
            dir = os.path.dirname(__file__)
            client.images.build(
                path=dir,
                dockerfile=dir + "/Dockerfile.sandbox",
                tag=SANDBOX_IMAGE,
            )

        self.client = client
        return True

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
    ) -> Sandbox | None:
        if not self._ensure_client():
            return None

        if connect is not None:
            sandbox = DockerSandbox.connect(connect)
        else:
            sandbox = DockerSandbox.create(
                image,
                ports=ports,
                dockerfile=dockerfile,
                name=name,
                env=env,
                mounts=mounts,
            )
        sandbox._on_close = self._record_sandbox_close
        self.sandboxes.add(sandbox)
        return sandbox

    def _record_sandbox_close(self, sandbox: DockerSandbox):
        self.sandboxes.discard(sandbox)

    def close(self) -> None:
        errors: list[Exception] = []
        for sandbox in list(self.sandboxes):
            try:
                sandbox.close()
            except Exception as e:
                errors.append(e)
        if errors:
            raise ExceptionGroup("Errors during cleanup", errors)

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for sandbox in list(self.sandboxes):
            try:
                await sandbox.aclose()
            except Exception as e:
                errors.append(e)
        if errors:
            raise ExceptionGroup("Errors during cleanup", errors)
