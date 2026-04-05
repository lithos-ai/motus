import asyncio
import os
import tarfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from socket import SocketIO
from struct import Struct
from threading import Thread
from typing import Mapping

import docker
from docker.api.client import APIClient
from docker.models.containers import Container

from ...core import Sandbox

SANDBOX_IMAGE = "ghcr.io/lithos-ai/sandbox"


class DockerSandbox(Sandbox):
    """Docker-based sandbox implementation.

    Usage::

        # Async — from image
        async with await DockerSandbox.acreate("python:3.12") as sb:
            output = await sb.python("print('hello')")

        # Sync — from image
        with DockerSandbox.create("python:3.12") as sb: ...

        # From Dockerfile
        sb = await DockerSandbox.acreate(dockerfile="./my-project")

        # Connect to existing container (won't destroy on close)
        with DockerSandbox.connect("my-dev-container") as sb: ...

        # Direct injection (testing / orchestration)
        sb = DockerSandbox(mock_container, owns=False)

        # As agent tools
        agent = Agent(tools=[sb])
    """

    def __init__(
        self,
        container: Container,
        *,
        owns: bool = True,
        on_close: Callable[["DockerSandbox"], None] | None = None,
    ) -> None:
        self.container = container
        self._owns = owns
        self._on_close = on_close

    @property
    def id(self) -> str:
        return self.container.id

    @property
    def name(self) -> str:
        return self.container.name

    # --- Factories ---

    @classmethod
    async def acreate(
        cls,
        image: str = "python:3.12",
        *,
        ports: dict[int, int | None] | None = None,
        dockerfile: str | Path | None = None,
        name: str | None = None,
        env: Mapping[str, str] | None = None,
        mounts: Mapping[str, str] | None = None,
    ) -> "DockerSandbox":
        """Async factory: create a new sandbox from an image or Dockerfile.

        Args:
            ports: Container-port to host-port mapping, e.g. ``{8080: None}``.
            mounts: Host-to-container path mapping, e.g.
                ``{"/local/project": "/workspace"}``
        """
        client = docker.from_env()

        if dockerfile is not None:
            img, _ = await asyncio.to_thread(client.images.build, path=str(dockerfile))
            image = img.id

        volumes = (
            {src: {"bind": dst, "mode": "rw"} for src, dst in mounts.items()}
            if mounts
            else None
        )
        docker_ports = {f"{p}/tcp": h for p, h in ports.items()} if ports else None
        container = await asyncio.to_thread(
            client.containers.run,
            image,
            stdin_open=True,
            detach=True,
            name=name,
            environment=env,
            volumes=volumes,
            ports=docker_ports,
        )
        try:
            await asyncio.to_thread(container.reload)
        except BaseException:
            await asyncio.to_thread(container.stop)
            await asyncio.to_thread(container.remove)
            raise
        return cls(container, owns=True)

    @classmethod
    def create(
        cls,
        image: str = "python:3.12",
        *,
        ports: dict[int, int | None] | None = None,
        dockerfile: str | Path | None = None,
        name: str | None = None,
        env: Mapping[str, str] | None = None,
        mounts: Mapping[str, str] | None = None,
    ) -> "DockerSandbox":
        """Sync factory: direct Docker SDK calls (no asyncio.run overhead).

        Args:
            ports: Container-port to host-port mapping, e.g. ``{8080: None}``.
            mounts: Host-to-container path mapping, e.g.
                ``{"/local/project": "/workspace"}``
        """
        client = docker.from_env()

        if dockerfile is not None:
            img, _ = client.images.build(path=str(dockerfile))
            image = img.id

        volumes = (
            {src: {"bind": dst, "mode": "rw"} for src, dst in mounts.items()}
            if mounts
            else None
        )
        docker_ports = {f"{p}/tcp": h for p, h in ports.items()} if ports else None
        container = client.containers.run(
            image,
            stdin_open=True,
            detach=True,
            name=name,
            environment=env,
            volumes=volumes,
            ports=docker_ports,
        )
        try:
            container.reload()
        except BaseException:
            container.stop()
            container.remove()
            raise
        return cls(container, owns=True)

    # --- Connect ---

    @classmethod
    def connect(cls, name_or_id: str) -> "DockerSandbox":
        """Connect to an existing running container. Won't destroy on close."""
        client = docker.from_env()
        container = client.containers.get(name_or_id)
        if container.status != "running":
            raise RuntimeError(
                f"Container '{name_or_id}' is {container.status}. "
                f"Use create() for a fresh sandbox."
            )
        return cls(container, owns=False)

    # --- Port mapping ---

    def endpoint(self, port: int) -> str:
        """Return the URL to reach a service listening on *port* inside the sandbox."""
        self.container.reload()
        mapping = self.container.ports.get(f"{port}/tcp")
        if not mapping:
            raise RuntimeError(f"Port {port} is not mapped")
        host_port = mapping[0]["HostPort"]
        return f"http://localhost:{host_port}"

    # --- Core primitives ---

    async def exec(
        self,
        *cmd: str,
        input: str | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        """Execute a command in the sandbox and return its output."""
        api: APIClient = self.container.client.api
        e = api.exec_create(
            self.container.id, cmd, stdin=True, environment=env, workdir=cwd
        )
        sock: SocketIO = api.exec_start(e["Id"], socket=True)
        if input is not None:
            thread = Thread(target=sock.write, args=[input.encode()])
            thread.start()
        header = Struct(">BxxxI")

        def read():
            output = bytearray()
            while True:
                hdr = sock.read(header.size)
                if not hdr:
                    break
                _t, length = header.unpack(hdr)
                while length > 0:
                    chunk = sock.read(length)
                    output += chunk
                    length -= len(chunk)
            if input is not None:
                thread.join()
            return output

        try:
            output = await asyncio.to_thread(read)
        finally:
            sock.close()
        return output.decode()

    async def put(self, local_path: str, sandbox_path: str) -> None:
        """Copy a file from the host to the sandbox."""
        if not sandbox_path.startswith("/"):
            raise ValueError("Target path must be absolute")
        data = BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            tar.add(local_path, arcname=sandbox_path.lstrip("/"))
        data.seek(0)
        success = await asyncio.to_thread(
            self.container.put_archive, path="/", data=data
        )
        if not success:
            raise RuntimeError(f"Failed to upload {local_path} to {sandbox_path}")

    async def get(self, sandbox_path: str, local_path: str) -> str:
        """Copy a file from the sandbox to the host.

        If local_path is a directory, the file is saved there with its
        original basename. Returns the resolved local file path.
        """
        if not sandbox_path.startswith("/"):
            raise ValueError("Source path must be absolute")
        if os.path.isdir(local_path):
            local_path = os.path.join(local_path, os.path.basename(sandbox_path))
        bits, _stat = self.container.get_archive(sandbox_path)
        data = BytesIO()
        for chunk in bits:
            data.write(chunk)
        data.seek(0)
        with (
            tarfile.open(fileobj=data, mode="r") as tar,
            tar.extractfile(os.path.basename(sandbox_path)) as remote_f,
            open(local_path, "wb") as local_f,
        ):
            local_f.write(remote_f.read())
        return local_path

    # --- Lifecycle ---

    async def aclose(self) -> None:
        if self._owns:
            await asyncio.to_thread(self.container.stop)
            await asyncio.to_thread(self.container.remove)
        if self._on_close:
            self._on_close(self)

    def close(self) -> None:
        if self._owns:
            self.container.stop()
            self.container.remove()
        if self._on_close:
            self._on_close(self)
