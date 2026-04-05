import os
import tempfile
import unittest

import docker

from motus.tools import DEFAULT_TOOL_PROVIDER


class TestSandboxFiles(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        try:
            client = docker.from_env()
            client.images.get("ghcr.io/lithos-ai/sandbox")
        except Exception as exc:
            raise unittest.SkipTest(
                "Sandbox image ghcr.io/lithos-ai/sandbox is not available"
            ) from exc

    async def test_put_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "source.txt")
            target = os.path.join(tmpdir, "target.txt")
            payload = "hello from host\n"
            with open(source, "wb") as f:
                f.write(payload.encode())

            with DEFAULT_TOOL_PROVIDER() as tool_provider:
                with tool_provider.get_sandbox() as sandbox:
                    await sandbox.put(source, "/tmp/source.txt")
                    output = await sandbox.exec("sh", "-c", "cat /tmp/source.txt")
                    self.assertEqual(output, payload)
                    await sandbox.get("/tmp/source.txt", target)

            with open(target, "rb") as f:
                self.assertEqual(f.read().decode(), payload)
