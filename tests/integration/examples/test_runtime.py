"""
Tests for runtime examples — hooks, resilient tasks, task graphs, guardrails.

All demos are self-contained with assertions and need no API keys.

Usage:
    pytest tests/integration/examples/test_runtime.py -v
"""

import subprocess
import sys

import pytest

EXAMPLES = [
    ("hooks", "examples/runtime/hooks_demo.py"),
    ("resilient_tasks", "examples/runtime/resilient_tasks.py"),
    ("task_graph", "examples/runtime/task_graph_demo.py"),
    ("guardrails", "examples/runtime/guardrails/guardrails_demo.py"),
]


@pytest.mark.parametrize("name,script", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_runtime_example(name, script):
    """Run each runtime example as a subprocess and verify it exits cleanly."""
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{name} failed (exit {result.returncode}):\n{result.stderr}"
    )
