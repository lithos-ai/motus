"""Subprocess regression test: process must exit promptly when executor threads
are blocked on AgentFuture.af_result() — the real deadlock pattern from production."""

import subprocess
import sys
import textwrap
import time

# Reproduces the real deadlock: a sync @agent_task blocks on .af_result() of an
# async task that never completes. Poisoning the futures during shutdown
# unblocks the thread so _python_exit().join() succeeds.
CHILD_SCRIPT = textwrap.dedent("""\
    import asyncio
    import logging
    import os

    logging.disable(logging.DEBUG)
    os.environ["MOTUS_QUIET_SYNC"] = "1"

    from motus.runtime.agent_task import agent_task
    from motus.runtime.agent_runtime import shutdown

    @agent_task
    async def never_complete():
        # Simulates an LLM call that hangs — runs on event loop, never resolves
        await asyncio.Event().wait()
        return "done"

    @agent_task
    def blocking_sync():
        # Simulates ReActAgent._run blocking on tool_result.af_result()
        return never_complete().af_result()

    async def run():
        future = blocking_sync()
        try:
            await asyncio.wait_for(future, timeout=1.0)
        except asyncio.TimeoutError:
            pass

    asyncio.run(run())
    shutdown()
""")


class TestExecutorExit:
    def test_process_exits_after_timeout(self):
        """Process must exit in <5s when threads are blocked on resolve()."""
        start = time.time()
        result = subprocess.run(
            [sys.executable, "-c", CHILD_SCRIPT],
            timeout=15,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start

        assert elapsed < 5, (
            f"Child process took {elapsed:.1f}s to exit (expected <5s).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
