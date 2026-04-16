import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
