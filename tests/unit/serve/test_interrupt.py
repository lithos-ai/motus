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


def test_init_interrupt_channel_binds_current_loop():
    """_init_interrupt_channel binds to the running loop and starts the reader thread."""
    import asyncio
    import multiprocessing as mp

    import motus.serve.interrupt as interrupt_mod

    # Reset module state between tests
    interrupt_mod._conn = None
    interrupt_mod._loop = None
    interrupt_mod._reader_thread = None
    interrupt_mod._pending = {}

    parent_conn, child_conn = mp.Pipe(duplex=True)

    async def run():
        interrupt_mod._init_interrupt_channel(child_conn)
        assert interrupt_mod._conn is child_conn
        assert interrupt_mod._loop is asyncio.get_running_loop()
        assert interrupt_mod._reader_thread is not None
        assert interrupt_mod._reader_thread.is_alive()
        assert interrupt_mod._reader_thread.daemon is True

    asyncio.run(run())

    # Reset interrupt module state (reader thread, conn, etc.)
    interrupt_mod._conn = None
    interrupt_mod._loop = None
    interrupt_mod._reader_thread = None
    interrupt_mod._pending = {}

    parent_conn.close()
    child_conn.close()
