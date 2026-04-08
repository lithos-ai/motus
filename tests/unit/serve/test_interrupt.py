from dataclasses import is_dataclass

import pytest


def test_interrupt_message_dataclass():
    from motus.serve.interrupt import InterruptMessage

    assert is_dataclass(InterruptMessage)
    msg = InterruptMessage(interrupt_id="abc", payload={"type": "test"})
    assert msg.interrupt_id == "abc"
    assert msg.payload == {"type": "test"}


def test_resume_message_dataclass():
    from motus.serve.interrupt import ResumeMessage

    assert is_dataclass(ResumeMessage)
    msg = ResumeMessage(interrupt_id="abc", value={"approved": True})
    assert msg.interrupt_id == "abc"
    assert msg.value == {"approved": True}


def test_max_message_bytes_defined():
    from motus.serve.interrupt import MAX_MESSAGE_BYTES

    assert MAX_MESSAGE_BYTES == 16 * 1024


def test_interrupt_raises_outside_worker():
    """Calling interrupt() outside a worker subprocess (no _init_interrupt_channel)
    must raise RuntimeError."""
    import asyncio

    from motus.serve.interrupt import interrupt

    async def run():
        await interrupt({"type": "test"})

    with pytest.raises(RuntimeError, match="outside motus serve worker"):
        asyncio.run(run())


def test_init_interrupt_channel_asserts_single_loop_mode():
    """_init_interrupt_channel must assert that AgentRuntime is in single-loop
    mode bound to the current asyncio loop."""
    import asyncio
    import multiprocessing as mp

    import motus.serve.interrupt as interrupt_mod

    # Reset module state between tests
    interrupt_mod._conn = None
    interrupt_mod._loop = None
    interrupt_mod._reader_thread = None
    interrupt_mod._pending = {}

    # Reset agent_runtime so get_runtime() auto-initializes in single-loop mode
    import motus.runtime.agent_runtime as ar_mod

    ar_mod._runtime = None

    parent_conn, child_conn = mp.Pipe(duplex=True)

    async def run():
        # First call to get_runtime() inside a running loop should initialize
        # single-loop mode; _init_interrupt_channel's assertion should pass.
        interrupt_mod._init_interrupt_channel(child_conn)
        assert interrupt_mod._conn is child_conn
        assert interrupt_mod._loop is asyncio.get_running_loop()
        assert interrupt_mod._reader_thread is not None
        assert interrupt_mod._reader_thread.is_alive()
        assert interrupt_mod._reader_thread.daemon is True

    asyncio.run(run())

    # Clean up: CRITICAL — reset the runtime singleton so subsequent tests
    # don't inherit a runtime bound to the now-closed asyncio.run() loop.
    # Without this, any later test calling get_runtime() gets a singleton
    # whose .loop is closed → RuntimeError: Event loop is closed.
    ar_mod._runtime = None

    # Also reset interrupt module state (reader thread, conn, etc.)
    interrupt_mod._conn = None
    interrupt_mod._loop = None
    interrupt_mod._reader_thread = None
    interrupt_mod._pending = {}

    parent_conn.close()
    child_conn.close()
