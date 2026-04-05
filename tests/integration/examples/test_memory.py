"""
Tests for memory examples — session restore and compaction.

Both demos use mock clients and need no API keys.

Usage:
    pytest tests/integration/examples/test_memory.py -v
"""

import subprocess
import sys

import pytest

DEMOS = [
    ("memory_restore", ["memory_restore"]),
    ("session_restore", ["session_restore"]),
]


@pytest.mark.parametrize("name,args", DEMOS, ids=[d[0] for d in DEMOS])
def test_memory_demo(name, args):
    """Run each memory demo as a subprocess and verify it exits cleanly."""
    result = subprocess.run(
        [sys.executable, "examples/memory.py", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{name} failed (exit {result.returncode}):\n{result.stderr}"
    )
