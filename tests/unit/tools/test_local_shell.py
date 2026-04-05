import os
import tempfile
import unittest

from motus.tools.providers.local import LocalShell


class TestLocalShell(unittest.IsolatedAsyncioTestCase):
    """Tests for LocalShell — local subprocess-based Sandbox."""

    def setUp(self):
        self.shell = LocalShell()

    def tearDown(self):
        self.shell.close()

    # --- sh / exec basics ---

    async def test_sh_echo(self):
        out = await self.shell.sh("echo hello")
        self.assertEqual(out, "hello\n")

    async def test_exec_echo(self):
        out = await self.shell.exec("echo", "world")
        self.assertEqual(out, "world\n")

    async def test_python(self):
        out = await self.shell.python("print(1+2)")
        self.assertEqual(out.strip(), "3")

    # --- cwd ---

    async def test_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            real_td = os.path.realpath(td)
            shell = LocalShell(cwd=real_td)
            out = await shell.sh("pwd")
            self.assertEqual(out.strip(), real_td)

    # --- non-zero exit code (no raise) ---

    async def test_nonzero_exit_returns_output(self):
        out = await self.shell.sh("echo fail >&2; exit 1")
        self.assertIn("fail", out)

    # --- input piping ---

    async def test_exec_with_input(self):
        out = await self.shell.exec("cat", input="hello from stdin")
        self.assertEqual(out, "hello from stdin")

    # --- put + get ---

    async def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.txt")
            dst = os.path.join(td, "dst.txt")
            with open(src, "w") as f:
                f.write("test content")

            await self.shell.put(src, dst)
            self.assertTrue(os.path.exists(dst))
            with open(dst) as f:
                self.assertEqual(f.read(), "test content")

    async def test_get_into_directory(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "file.txt")
            dst_dir = os.path.join(td, "out")
            os.makedirs(dst_dir)
            with open(src, "w") as f:
                f.write("data")

            result = await self.shell.get(src, dst_dir)
            self.assertEqual(result, os.path.join(dst_dir, "file.txt"))
            with open(result) as f:
                self.assertEqual(f.read(), "data")

    # --- context manager ---

    def test_context_manager(self):
        with LocalShell() as sh:
            self.assertIsInstance(sh, LocalShell)

    # --- factories ---

    def test_create_ignores_image(self):
        sh = LocalShell.create("node:20")
        self.assertIsInstance(sh, LocalShell)
        sh.close()

    def test_connect_uses_cwd(self):
        sh = LocalShell.connect("/tmp")
        self.assertEqual(sh._cwd, "/tmp")
        sh.close()

    # --- env ---

    async def test_exec_with_env(self):
        out = await self.shell.exec("sh", "-c", "echo $MY_VAR", env={"MY_VAR": "42"})
        self.assertEqual(out.strip(), "42")
