"""Tests for the Ray-style init() / shutdown() runtime lifecycle."""

import threading
import time

import pytest

from motus.runtime.agent_future import resolve
from motus.runtime.agent_runtime import (
    AgentRuntime,
    GraphScheduler,
    get_runtime,
    init,
    is_initialized,
    shutdown,
)
from motus.runtime.agent_task import agent_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_clean():
    """Shut down any existing runtime so each test starts from a clean state."""
    shutdown()


# ---------------------------------------------------------------------------
# Lifecycle basics
# ---------------------------------------------------------------------------


class TestInitShutdown:
    def setup_method(self):
        _ensure_clean()

    def teardown_method(self):
        _ensure_clean()

    def test_not_initialized_after_shutdown(self):
        assert not is_initialized()

    def test_init_creates_runtime(self):
        rt = init()
        assert isinstance(rt, AgentRuntime)
        assert is_initialized()

    def test_shutdown_clears_runtime(self):
        init()
        assert is_initialized()
        shutdown()
        assert not is_initialized()

    def test_double_init_raises(self):
        init()
        with pytest.raises(RuntimeError, match="already initialized"):
            init()

    def test_shutdown_then_reinit(self):
        rt1 = init()
        shutdown()
        rt2 = init()
        assert rt2 is not rt1
        assert is_initialized()

    def test_shutdown_when_not_initialized_is_noop(self):
        # Should not raise
        shutdown()
        shutdown()

    def test_init_with_tracing_enabled(self, monkeypatch):
        monkeypatch.setenv("MOTUS_TRACING", "1")
        rt = init()
        assert rt.scheduler.tracer.config.is_collecting
        assert rt.scheduler.tracer.config.export_enabled

    def test_init_with_tracing_disabled(self, monkeypatch):
        monkeypatch.setenv("MOTUS_COLLECTION_LEVEL", "disabled")
        rt = init()
        assert not rt.scheduler.tracer.config.is_collecting


# ---------------------------------------------------------------------------
# Auto-init via get_runtime
# ---------------------------------------------------------------------------


class TestAutoInit:
    def setup_method(self):
        _ensure_clean()

    def teardown_method(self):
        _ensure_clean()

    def test_get_runtime_auto_initializes(self):
        assert not is_initialized()
        rt = get_runtime()
        assert rt is not None
        assert is_initialized()

    def test_auto_init_via_agent_task(self):
        assert not is_initialized()

        @agent_task
        async def hello():
            return "world"

        result = resolve(hello())
        assert result == "world"
        assert is_initialized()


# ---------------------------------------------------------------------------
# No import side effects
# ---------------------------------------------------------------------------


class TestNoImportSideEffects:
    def setup_method(self):
        _ensure_clean()

    def teardown_method(self):
        _ensure_clean()

    def test_runtime_is_none_after_shutdown(self):
        import motus.runtime.agent_runtime as mod

        assert mod._runtime is None

    def test_no_agent_engine_thread_before_init(self):
        agent_threads = [t for t in threading.enumerate() if t.name == "AgentEngine"]
        # After a clean shutdown there should be no AgentEngine thread
        # (it's a daemon thread that stops when the loop stops).
        # Give it a moment to wind down.
        import time

        time.sleep(0.1)
        agent_threads = [t for t in threading.enumerate() if t.name == "AgentEngine"]
        assert len(agent_threads) == 0


# ---------------------------------------------------------------------------
# Top-level motus.init / motus.shutdown
# ---------------------------------------------------------------------------


class TestTopLevelAPI:
    def setup_method(self):
        _ensure_clean()

    def teardown_method(self):
        _ensure_clean()

    def test_motus_namespace(self):
        import motus

        assert not motus.is_initialized()
        motus.init()
        assert motus.is_initialized()
        motus.shutdown()
        assert not motus.is_initialized()


# ---------------------------------------------------------------------------
# Executor shutdown
# ---------------------------------------------------------------------------


class TestExecutorShutdown:
    def setup_method(self):
        _ensure_clean()

    def teardown_method(self):
        _ensure_clean()

    def test_shutdown_with_running_executor_task(self):
        """Shutdown must return promptly when an executor thread is blocked on resolve()."""
        import asyncio

        init()

        @agent_task
        async def never_complete():
            await asyncio.Event().wait()
            return "done"

        @agent_task
        def blocking_sync():
            return never_complete().af_result()

        # Fire-and-forget: executor thread will block on resolve()
        blocking_sync()

        # Give the executor thread a moment to start and block
        time.sleep(0.3)

        start = time.time()
        shutdown()
        elapsed = time.time() - start

        assert elapsed < 5, (
            f"shutdown() took {elapsed:.1f}s — executor thread blocked exit"
        )


# ---------------------------------------------------------------------------
# GC: completed tasks and futures are released
# ---------------------------------------------------------------------------


class TestSchedulerGC:
    """Verify that completed tasks/futures are removed from scheduler dicts."""

    def setup_method(self):
        _ensure_clean()

    def teardown_method(self):
        _ensure_clean()

    def _scheduler(self) -> GraphScheduler:
        return get_runtime().scheduler

    def test_simple_task_gc(self):
        """A single completed task should leave all scheduler dicts empty."""

        @agent_task
        async def add_one(x):
            return x + 1

        result = resolve(add_one(41))
        assert result == 42

        sched = self._scheduler()
        assert len(sched.tasks) == 0, f"tasks not cleaned: {sched.tasks}"
        assert len(sched.future_to_dependents) == 0, (
            f"future_to_dependents not cleaned: {sched.future_to_dependents}"
        )
        assert len(sched.agent_futures) == 0, (
            f"agent_futures not cleaned: {sched.agent_futures}"
        )

    def test_chain_gc(self):
        """A chain A -> B -> C should leave all dicts empty after completion."""

        @agent_task
        async def step_a():
            return 1

        @agent_task
        async def step_b(x):
            return x + 10

        @agent_task
        async def step_c(x):
            return x + 100

        a = step_a()
        b = step_b(a)
        c = step_c(b)

        assert resolve(c) == 111

        sched = self._scheduler()
        assert len(sched.tasks) == 0, f"tasks not cleaned: {sched.tasks}"
        assert len(sched.future_to_dependents) == 0, (
            f"future_to_dependents not cleaned: {sched.future_to_dependents}"
        )
        assert len(sched.agent_futures) == 0, (
            f"agent_futures not cleaned: {sched.agent_futures}"
        )

    def test_error_gc(self):
        """Failed tasks should also be cleaned up."""

        @agent_task
        async def fail_task():
            raise ValueError("boom")

        fut = fail_task()
        with pytest.raises(ValueError, match="boom"):
            resolve(fut)

        sched = self._scheduler()
        assert len(sched.tasks) == 0, f"tasks not cleaned: {sched.tasks}"
        assert len(sched.future_to_dependents) == 0, (
            f"future_to_dependents not cleaned: {sched.future_to_dependents}"
        )
        assert len(sched.agent_futures) == 0, (
            f"agent_futures not cleaned: {sched.agent_futures}"
        )

    def test_fan_out_gc(self):
        """Multiple tasks depending on the same future should all be cleaned."""

        @agent_task
        async def root():
            return 5

        @agent_task
        async def branch(x):
            return x * 2

        r = root()
        b1 = branch(r)
        b2 = branch(r)

        assert resolve(b1) == 10
        assert resolve(b2) == 10

        sched = self._scheduler()
        assert len(sched.tasks) == 0, f"tasks not cleaned: {sched.tasks}"
        assert len(sched.future_to_dependents) == 0, (
            f"future_to_dependents not cleaned: {sched.future_to_dependents}"
        )
        assert len(sched.agent_futures) == 0, (
            f"agent_futures not cleaned: {sched.agent_futures}"
        )
