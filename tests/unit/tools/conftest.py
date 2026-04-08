"""Reset the global motus runtime between tests.

FunctionTool and AgentTool trigger auto-initialization of the global
runtime singleton.  unittest.IsolatedAsyncioTestCase creates (and closes)
a fresh event loop per test, leaving the runtime holding a stale loop
reference.  Shutting down the runtime after each test prevents
``RuntimeError: Event loop is closed`` cascading to later tests.
"""

import pytest

import motus.runtime.agent_runtime as _rt


@pytest.fixture(autouse=True)
def _reset_runtime():
    yield
    if _rt._runtime is not None:
        try:
            _rt.shutdown()
        except Exception:
            pass
        _rt._runtime = None
