import unittest
import warnings

import docker

from motus.tools import DEFAULT_TOOL_PROVIDER


class TestSandboxTools(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        try:
            client = docker.from_env()
            client.images.get("ghcr.io/lithos-ai/sandbox")
        except Exception as exc:
            raise unittest.SkipTest(
                "Sandbox image ghcr.io/lithos-ai/sandbox is not available"
            ) from exc

    async def test_exec_helpers_run(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with DEFAULT_TOOL_PROVIDER() as tool_provider:
                with tool_provider.get_sandbox() as sandbox:
                    sh_output = await sandbox.sh("echo hi")
                    python_output = await sandbox.python("print(1)")

        self.assertEqual(sh_output, "hi\n")
        self.assertEqual(python_output, "1\n")
