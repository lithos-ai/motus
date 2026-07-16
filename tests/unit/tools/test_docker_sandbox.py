import threading
import unittest
from socket import SHUT_WR
from struct import Struct
from unittest.mock import MagicMock

from motus.tools.providers.docker.sandbox import DockerSandbox


class _WriteSocket:
    def __init__(self, parent: "_ExecSocket") -> None:
        self._parent = parent

    def sendall(self, data: bytes) -> None:
        self._parent.stdin.extend(data)

    def shutdown(self, how: int) -> None:
        if how != SHUT_WR:
            raise AssertionError(f"unexpected shutdown mode: {how}")
        self._parent.stdin_closed.set()


class _ExecSocket:
    def __init__(self, output: bytes) -> None:
        header = Struct(">BxxxI")
        self._output = bytearray(header.pack(1, len(output)) + output)
        self.stdin = bytearray()
        self.stdin_closed = threading.Event()
        self._sock = _WriteSocket(self)
        self.closed = False

    def write(self, data: memoryview) -> int:
        raise AssertionError("docker-py's response SocketIO is read-only")

    def read(self, length: int) -> bytes:
        if not self.stdin_closed.wait(timeout=1):
            raise TimeoutError("Docker exec stdin did not receive EOF")
        chunk = bytes(self._output[:length])
        del self._output[:length]
        return chunk

    def close(self) -> None:
        self.closed = True


class TestDockerSandboxExec(unittest.IsolatedAsyncioTestCase):
    async def test_input_is_fully_written_and_half_closed(self):
        sock = _ExecSocket(b"command output")
        api = MagicMock()
        api.exec_create.return_value = {"Id": "exec-id"}
        api.exec_start.return_value = sock
        container = MagicMock()
        container.id = "container-id"
        container.client.api = api

        sandbox = DockerSandbox(container, owns=False)
        output = await sandbox.exec("tee", "/tmp/file", input="abcdefg")

        self.assertEqual(output, "command output")
        self.assertEqual(bytes(sock.stdin), b"abcdefg")
        self.assertTrue(sock.stdin_closed.is_set())
        self.assertTrue(sock.closed)
