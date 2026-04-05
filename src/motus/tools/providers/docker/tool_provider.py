import logging
import os
from typing import Set

import docker

from ...core import Sandbox, SandboxProvider
from .sandbox import SANDBOX_IMAGE, DockerSandbox


class DockerToolProvider(SandboxProvider):
    """Docker-based sandbox provider.

    Manages Docker containers for sandbox creation.
    """

    def __init__(self) -> None:
        self.client = docker.from_env()

        if not any(
            any(
                tag == SANDBOX_IMAGE or tag.startswith(SANDBOX_IMAGE + ":")
                for tag in image.tags
            )
            for image in self.client.images.list()
        ):
            logging.info(
                f"Sandbox image '{SANDBOX_IMAGE}' not found locally, building..."
            )
            dir = os.path.dirname(__file__)
            self.client.images.build(
                path=dir,
                dockerfile=dir + "/Dockerfile.sandbox",
                tag=SANDBOX_IMAGE,
            )

        self.sandboxes: Set[DockerSandbox] = set()

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
