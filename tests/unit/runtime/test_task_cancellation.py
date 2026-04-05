"""Tests for task cancellation in the Motus runtime."""

import asyncio
import sys
import threading
import time

import pytest

from motus.runtime.agent_runtime import get_runtime
from motus.runtime.agent_task import agent_task
from motus.runtime.hooks import HookManager
from motus.runtime.task_instance import TaskCancelledError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@agent_task
async def returns(value):
    return value


@agent_task
async def slow_async(duration=1.0):
    await asyncio.sleep(duration)
    return "completed"


@agent_task
def slow_sync(duration=1.0):
    time.sleep(duration)
    return "completed"


@agent_task
async def identity(x):
    return x


@agent_task(num_returns=2)
async def split_two(a, b):
    await asyncio.sleep(0.5)
    return a * 10, b * 10


class Counter:
    """Mutable counter that survives _deep_unwrap."""

    def __init__(self):
        self.n = 0


@agent_task(retries=3, retry_delay=0.3)
async def retries_with_delay(counter: Counter):
    counter.n += 1
    raise ConnectionError(f"flake #{counter.n}")


# ---------------------------------------------------------------------------
# Cancel PENDING tasks
# ---------------------------------------------------------------------------


class TestCancelPendingTask:
    """Cancel a task that hasn't started executing yet."""

    def test_cancel_pending_returns_true(self):
        blocker = slow_async(10.0)
        dep = identity(blocker)  # PENDING — waiting on blocker
        assert dep.af_cancel() is True
        # Clean up blocker
        blocker.af_cancel()

    def test_cancel_pending_raises_cancelled_error(self):
        blocker = slow_async(10.0)
        dep = identity(blocker)
        dep.af_cancel()
        with pytest.raises(TaskCancelledError):
            dep.af_result(timeout=2)
        blocker.af_cancel()

    def test_cancel_pending_reports_cancelled(self):
        blocker = slow_async(10.0)
        dep = identity(blocker)
        dep.af_cancel()
        time.sleep(0.1)  # Allow event loop to process
        assert dep.af_cancelled() is True
        blocker.af_cancel()


# ---------------------------------------------------------------------------
# Cancel RUNNING async tasks
# ---------------------------------------------------------------------------


class TestCancelRunningAsyncTask:
    """Cancel a task that is currently executing as a coroutine."""

    def test_cancel_running_async(self):
        f = slow_async(10.0)
        time.sleep(0.1)  # Let it start running
        assert f.af_cancel() is True
        with pytest.raises(TaskCancelledError):
            f.af_result(timeout=2)

    def test_cancel_running_async_reports_cancelled(self):
        f = slow_async(10.0)
        time.sleep(0.1)
        f.af_cancel()
        time.sleep(0.2)
        assert f.af_cancelled() is True


# ---------------------------------------------------------------------------
# Cancel RUNNING sync tasks (best-effort)
# ---------------------------------------------------------------------------


class TestCancelRunningSyncTask:
    """Cancel a sync task running in the executor."""

    def test_cancel_sync_unblocks_af_result(self):
        """Even though the thread continues, .af_result() should raise immediately."""
        f = slow_sync(10.0)
        time.sleep(0.1)  # Let it start in the executor
        f.af_cancel()
        start = time.monotonic()
        with pytest.raises(TaskCancelledError):
            f.af_result(timeout=5)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0  # Should NOT wait for the 10s sleep


# ---------------------------------------------------------------------------
# Cancellation cascade through dependency graph
# ---------------------------------------------------------------------------


class TestCancelCascade:
    """Cancellation propagates to all downstream dependents."""

    def test_cascade_to_single_dependent(self):
        root = slow_async(10.0)
        dep = root + 1  # Deferred op depends on root
        root.af_cancel()
        with pytest.raises(TaskCancelledError):
            dep.af_result(timeout=2)

    def test_cascade_through_chain(self):
        root = slow_async(10.0)
        a = root + 1
        b = a * 2
        c = b - 3
        root.af_cancel()
        with pytest.raises(TaskCancelledError):
            c.af_result(timeout=2)

    def test_cascade_fan_out(self):
        root = slow_async(10.0)
        d1 = root + 1
        d2 = root + 2
        d3 = root + 3
        root.af_cancel()
        for dep in [d1, d2, d3]:
            with pytest.raises(TaskCancelledError):
                dep.af_result(timeout=2)


# ---------------------------------------------------------------------------
# Multi-return cancellation
# ---------------------------------------------------------------------------


class TestCancelMultiReturn:
    """Cancelling any result future cancels the whole multi-return task."""

    def test_cancel_first_future_cancels_second(self):
        r1, r2 = split_two(1, 2)
        time.sleep(0.05)
        r1.af_cancel()
        with pytest.raises(TaskCancelledError):
            r2.af_result(timeout=2)

    def test_cancel_second_future_cancels_first(self):
        r1, r2 = split_two(1, 2)
        time.sleep(0.05)
        r2.af_cancel()
        with pytest.raises(TaskCancelledError):
            r1.af_result(timeout=2)


# ---------------------------------------------------------------------------
# Cancellation + retry interaction
# ---------------------------------------------------------------------------


class TestCancelAndRetry:
    """Cancellation during retry should not retry further."""

    def test_cancel_during_retry_delay(self):
        counter = Counter()
        f = retries_with_delay(counter)
        time.sleep(0.2)  # Let first attempt fail and enter retry delay
        f.af_cancel()
        with pytest.raises(TaskCancelledError):
            f.af_result(timeout=2)
        # Should not have exhausted all retries
        assert counter.n <= 2


# ---------------------------------------------------------------------------
# Cancel already-done futures
# ---------------------------------------------------------------------------


class TestCancelAlreadyDone:
    """Cancelling a completed or failed future is a no-op."""

    def test_cancel_completed_returns_false(self):
        f = returns(42)
        assert f.af_result(timeout=5) == 42
        assert f.af_cancel() is False

    def test_cancel_failed_returns_false(self):
        @agent_task
        async def fail():
            raise ValueError("boom")

        f = fail()
        with pytest.raises(ValueError):
            f.af_result(timeout=5)
        assert f.af_cancel() is False


# ---------------------------------------------------------------------------
# .af_cancelled() method
# ---------------------------------------------------------------------------


class TestCancelledMethod:
    """The .af_cancelled() property correctly reports cancellation state."""

    def test_not_cancelled_initially(self):
        f = slow_async(10.0)
        assert f.af_cancelled() is False
        f.af_cancel()
        time.sleep(0.1)
        assert f.af_cancelled() is True

    def test_not_cancelled_after_success(self):
        f = returns(42)
        f.af_result(timeout=5)
        assert f.af_cancelled() is False

    def test_not_cancelled_after_error(self):
        @agent_task
        async def fail():
            raise ValueError("boom")

        f = fail()
        with pytest.raises(ValueError):
            f.af_result(timeout=5)
        assert f.af_cancelled() is False

    def test_cancelled_after_cancel(self):
        f = slow_async(10.0)
        time.sleep(0.1)
        f.af_cancel()
        time.sleep(0.2)
        assert f.af_cancelled() is True


# ---------------------------------------------------------------------------
# Hook system
# ---------------------------------------------------------------------------


class TestCancelTaskHook:
    """The hook system emits task_cancelled events."""

    def setup_method(self):
        """Swap the global hooks singleton with a fresh HookManager."""
        self._hooks_mod = sys.modules["motus.runtime.hooks"]
        self._rt_mod = sys.modules["motus.runtime.agent_runtime"]
        self._orig_hooks = self._hooks_mod.hooks
        self._orig_rt_hooks = self._rt_mod.hooks
        fresh = HookManager()
        self._hooks_mod.hooks = fresh
        self._rt_mod.hooks = fresh

    def teardown_method(self):
        """Restore the original hooks singleton."""
        self._hooks_mod.hooks = self._orig_hooks
        self._rt_mod.hooks = self._orig_rt_hooks

    def test_task_cancelled_hook_fires(self):
        events = []
        self._hooks_mod.hooks.register("task_cancelled", lambda e: events.append(e))

        f = slow_async(10.0)
        time.sleep(0.1)
        f.af_cancel()
        time.sleep(0.3)  # Allow hook to fire

        cancelled_events = [e for e in events if e.event_type == "task_cancelled"]
        assert len(cancelled_events) >= 1
        assert isinstance(cancelled_events[0].error, TaskCancelledError)


# ---------------------------------------------------------------------------
# Scheduler cleanup
# ---------------------------------------------------------------------------


class TestCancelSchedulerCleanup:
    """Cancelled tasks are cleaned up from scheduler state."""

    def test_cancelled_task_removed_from_scheduler(self):
        f = slow_async(10.0)
        time.sleep(0.1)
        task_count_before = len(get_runtime().scheduler.tasks)
        f.af_cancel()
        time.sleep(0.3)

        sched = get_runtime().scheduler
        # The task should have been removed
        assert len(sched.tasks) <= task_count_before - 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestCancelThreadSafety:
    """af_cancel() can be called safely from any thread."""

    def test_cancel_from_another_thread(self):
        f = slow_async(10.0)
        time.sleep(0.1)

        result_box = [None]

        def cancel_in_thread():
            result_box[0] = f.af_cancel()

        t = threading.Thread(target=cancel_in_thread)
        t.start()
        t.join(timeout=5)

        assert result_box[0] is True
        with pytest.raises(TaskCancelledError):
            f.af_result(timeout=2)
