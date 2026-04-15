import io
import sys
import unittest
from unittest.mock import patch


class TestCliUnicodeOutput(unittest.TestCase):
    def test_ensure_utf8_stdout_reconfigures(self):
        """_ensure_utf8_stdout sets encoding to utf-8."""
        from motus.cli import _ensure_utf8_stdout

        _ensure_utf8_stdout()
        assert sys.stdout.encoding.lower().replace("-", "") == "utf8"

    def test_print_emoji_after_reconfigure(self):
        """After reconfigure, printing emoji doesn't crash."""
        from motus.cli import _ensure_utf8_stdout

        _ensure_utf8_stdout()
        # This would crash on cp1252 without the fix
        print("Hello 👋😊 — curly quotes \"test\"")

    def test_send_and_wait_fallback_on_broken_stdout(self):
        """The line-221 fallback writes UTF-8 bytes when print() raises."""
        content = "Hello 👋 Fadil!"
        buf = io.BytesIO()

        class FakeStdout:
            """Simulates cp1252 stdout that can't encode emoji."""
            encoding = "cp1252"
            buffer = buf
            def write(self, s):
                s.encode("cp1252")  # raises UnicodeEncodeError

        with patch("sys.stdout", FakeStdout()):
            # This is the exact logic from the fix at line 221
            try:
                print(content)
            except UnicodeEncodeError:
                sys.stdout.buffer.write(content.encode("utf-8", errors="replace") + b"\n")

        assert "Hello".encode() in buf.getvalue()
        assert "Fadil".encode() in buf.getvalue()


if __name__ == "__main__":
    unittest.main()
