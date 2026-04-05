"""Tests for TaskPolicy, AgentTaskDefinition, retries, timeouts, stack stitching, and multi-return."""

import asyncio
import os
import time

import pytest

from motus.runtime.agent_future import AgentFuture
from motus.runtime.agent_task import AgentTaskDefinition, agent_task
from motus.runtime.task_instance import DEFAULT_POLICY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Counter:
    """Mutable counter that survives _deep_unwrap (not a dict/list/tuple)."""

    def __init__(self):
        self.n = 0


@agent_task
async def returns(value):
    return value


@agent_task(retries=3)
async def retries_three(counter: Counter):
    """Increment counter and fail until counter reaches 3."""
    counter.n += 1
    if counter.n < 3:
        raise ConnectionError(f"flake #{counter.n}")
    return "ok"


@agent_task(retries=1)
async def always_fails_with_retry():
    raise ValueError("permanent")


@agent_task
async def always_fails_no_retry():
    raise ValueError("immediate")


@agent_task(timeout=0.2)
async def slow_task():
    await asyncio.sleep(10)
    return "unreachable"


@agent_task
async def fast_task():
    await asyncio.sleep(0.05)
    return "done"


@agent_task
async def fails_for_stack_test():
    raise RuntimeError("boom")


# Sync helpers (run_in_executor path)
@agent_task
def sync_returns(value):
    return value


@agent_task(retries=3)
def sync_retries(counter: Counter):
    counter.n += 1
    if counter.n < 2:
        raise ConnectionError(f"sync flake #{counter.n}")
    return "sync ok"


@agent_task(timeout=0.3)
def sync_slow():
    time.sleep(10)
    return "unreachable"


# Multi-level chain helpers
@agent_task
def inner_fail():
    raise RuntimeError("deep boom")


@agent_task
def outer_call():
    return inner_fail().af_result()


# ---------------------------------------------------------------------------
# TestAgentTaskDefinition
# ---------------------------------------------------------------------------


class TestAgentTaskDefinition:
    def test_returns_agent_task_definition(self):
        assert isinstance(returns, AgentTaskDefinition)

    def test_preserves_name(self):
        assert returns.__name__ == "returns"
        assert retries_three.__name__ == "retries_three"

    def test_policy_returns_new_definition(self):
        original = returns
        overridden = returns.policy(retries=5, timeout=10)

        assert isinstance(overridden, AgentTaskDefinition)
        assert overridden is not original
        # Original unchanged
        assert original._policy == DEFAULT_POLICY
        # Override applied
        assert overridden._policy.retries == 5
        assert overridden._policy.timeout == 10

    def test_descriptor_on_class_method(self):
        class Worker:
            @agent_task
            def compute(self, x):
                return x * 2

        w = Worker()
        f = w.compute(21)
        assert f.af_result(timeout=5) == 42

    def test_descriptor_policy_on_method(self):
        class Worker:
            @agent_task(retries=1)
            def compute(self, x):
                return x + 1

        w = Worker()
        # .policy() on bound method returns _BoundAgentTask
        overridden = w.compute.policy(retries=0)
        f = overridden(10)
        assert f.af_result(timeout=5) == 11


# ---------------------------------------------------------------------------
# TestTaskRetries
# ---------------------------------------------------------------------------


class TestTaskRetries:
    def test_retry_succeeds_within_limit(self):
        counter = Counter()
        f = retries_three(counter)
        result = f.af_result(timeout=10)
        assert result == "ok"
        assert counter.n == 3  # failed twice, succeeded on 3rd

    def test_retry_exhausted_raises(self):
        with pytest.raises(ValueError, match="permanent"):
            # retries=1: tries once, retries once → 2 total attempts, all fail
            always_fails_with_retry().af_result(timeout=10)

    def test_no_retry_by_default(self):
        with pytest.raises(ValueError, match="immediate"):
            always_fails_no_retry().af_result(timeout=5)

    def test_retry_delay(self):
        counter = Counter()

        @agent_task(retries=2, retry_delay=0.15)
        async def delayed_retry(c: Counter):
            c.n += 1
            if c.n < 3:
                raise ConnectionError("flake")
            return "ok"

        start = time.monotonic()
        result = delayed_retry(counter).af_result(timeout=10)
        elapsed = time.monotonic() - start
        assert result == "ok"
        assert counter.n == 3
        # 2 retries × 0.15s delay = at least 0.3s
        assert elapsed >= 0.25


# ---------------------------------------------------------------------------
# TestTaskTimeout
# ---------------------------------------------------------------------------


class TestTaskTimeout:
    def test_timeout_fires(self):
        with pytest.raises(TimeoutError):
            slow_task().af_result(timeout=5)

    def test_no_timeout_by_default(self):
        result = fast_task().af_result(timeout=5)
        assert result == "done"


# ---------------------------------------------------------------------------
# TestPolicyOverride
# ---------------------------------------------------------------------------


class TestPolicyOverride:
    def test_override_retries(self):
        counter = Counter()

        @agent_task(retries=0)
        async def fragile(c: Counter):
            c.n += 1
            if c.n < 3:
                raise ConnectionError("flake")
            return "recovered"

        # Default: retries=0 → fails immediately
        with pytest.raises(ConnectionError):
            fragile(counter).af_result(timeout=5)

        assert counter.n == 1

        # Override to retries=3 → succeeds
        counter.n = 0
        result = fragile.policy(retries=3)(counter).af_result(timeout=10)
        assert result == "recovered"
        assert counter.n == 3

    def test_override_timeout(self):
        @agent_task(timeout=10)
        async def generous():
            await asyncio.sleep(0.5)
            return "ok"

        # Override to very short timeout
        with pytest.raises(TimeoutError):
            generous.policy(timeout=0.05)().af_result(timeout=5)


# ---------------------------------------------------------------------------
# TestCreationStackStitching
# ---------------------------------------------------------------------------


class TestSyncFunctions:
    """Sync (non-async) functions go through run_in_executor — different code path."""

    def test_sync_returns(self):
        assert sync_returns(42).af_result(timeout=5) == 42

    def test_sync_retries(self):
        counter = Counter()
        result = sync_retries(counter).af_result(timeout=10)
        assert result == "sync ok"
        assert counter.n == 2  # failed once, succeeded on 2nd

    def test_sync_timeout(self):
        with pytest.raises(TimeoutError):
            sync_slow().af_result(timeout=5)


# ---------------------------------------------------------------------------
# TestCreationStackStitching
# ---------------------------------------------------------------------------


class TestCreationStackStitching:
    def test_single_unified_note(self):
        try:
            fails_for_stack_test().af_result(timeout=5)
            pytest.fail("Should have raised")
        except RuntimeError as e:
            assert hasattr(e, "__notes__")
            # Exactly one note — no fragmentation
            assert len(e.__notes__) == 1
            assert "Task creation chain" in e.__notes__[0]

    def test_notes_contain_caller_location(self):
        try:
            fails_for_stack_test().af_result(timeout=5)
            pytest.fail("Should have raised")
        except RuntimeError as e:
            note = e.__notes__[0]
            # The note should reference THIS test file, not motus internals
            assert "test_task_policy.py" in note
            # Should NOT contain motus runtime or stdlib frames
            assert os.path.join("motus", "runtime") not in note
            assert "threading.py" not in note
            assert os.sep + "concurrent" + os.sep not in note
            assert os.sep + "asyncio" + os.sep not in note

    def test_error_site_in_chain(self):
        """The chain should end at the actual raise line, not just the creation site."""
        try:
            fails_for_stack_test().af_result(timeout=5)
            pytest.fail("Should have raised")
        except RuntimeError as e:
            note = e.__notes__[0]
            # Should contain the function that raised
            assert "fails_for_stack_test" in note
            assert (
                "boom" in note
                or "RuntimeError" in note
                or 'raise RuntimeError("boom")' in note
            )

    def test_multi_level_chain(self):
        """When tasks are nested, the chain should include all levels in one note."""
        try:
            outer_call().af_result(timeout=5)
            pytest.fail("Should have raised")
        except RuntimeError as e:
            assert len(e.__notes__) == 1
            note = e.__notes__[0]
            # Both creation sites should appear
            assert "outer_call" in note
            assert "inner_fail" in note
            # Outermost should appear before innermost
            outer_pos = note.index("outer_call")
            inner_pos = note.index("inner_fail")
            assert outer_pos < inner_pos


# ---------------------------------------------------------------------------
# TestMultiReturn
# ---------------------------------------------------------------------------


@agent_task(num_returns=2)
def split_two(a, b):
    return a * 10, b * 10


@agent_task(num_returns=3)
async def split_three_async(x):
    return x, x + 1, x + 2


@agent_task(num_returns=2)
def bad_count():
    return 1, 2, 3  # 3 elements but num_returns=2


@agent_task(num_returns=2)
def returns_scalar():
    return 42  # scalar but num_returns=2


@agent_task(num_returns=2)
def fails_multi():
    raise RuntimeError("multi boom")


@agent_task
def identity(x):
    return x


@agent_task(num_returns=2)
def returns_nested_futures():
    """Return AgentFutures as elements — triggers RESOLVE path."""
    f1 = identity(100)
    f2 = identity(200)
    return f1, f2


class TestMultiReturn:
    def test_basic_multi_return(self):
        r1, r2 = split_two(3, 7)
        assert isinstance(r1, AgentFuture)
        assert isinstance(r2, AgentFuture)
        assert r1.af_result(timeout=5) == 30
        assert r2.af_result(timeout=5) == 70

    def test_basic_multi_return_async(self):
        r1, r2, r3 = split_three_async(10)
        assert r1.af_result(timeout=5) == 10
        assert r2.af_result(timeout=5) == 11
        assert r3.af_result(timeout=5) == 12

    def test_multi_return_wrong_count(self):
        r1, r2 = bad_count()
        with pytest.raises(ValueError, match="num_returns=2"):
            r1.af_result(timeout=5)
        with pytest.raises(ValueError, match="num_returns=2"):
            r2.af_result(timeout=5)

    def test_multi_return_not_tuple(self):
        r1, r2 = returns_scalar()
        with pytest.raises(ValueError, match="num_returns=2"):
            r1.af_result(timeout=5)

    def test_multi_return_error_propagates_to_all(self):
        r1, r2 = fails_multi()
        with pytest.raises(RuntimeError, match="multi boom"):
            r1.af_result(timeout=5)
        with pytest.raises(RuntimeError, match="multi boom"):
            r2.af_result(timeout=5)

    def test_multi_return_fine_grained_deps(self):
        r1, r2 = split_two(1, 2)

        @agent_task
        def add_one(x):
            return x + 1

        # Each downstream depends on only one of the multi-return futures
        d1 = add_one(r1)
        d2 = add_one(r2)
        assert d1.af_result(timeout=5) == 11  # 1*10 + 1
        assert d2.af_result(timeout=5) == 21  # 2*10 + 1

    def test_multi_return_with_nested_futures(self):
        r1, r2 = returns_nested_futures()
        assert r1.af_result(timeout=5) == 100
        assert r2.af_result(timeout=5) == 200

    def test_single_return_unchanged(self):
        result = identity(42)
        assert isinstance(result, AgentFuture)
        assert not isinstance(result, tuple)
        assert result.af_result(timeout=5) == 42

    def test_policy_override_num_returns(self):
        @agent_task
        def basic(a, b):
            return a, b

        # Override to num_returns=2 via .policy()
        r1, r2 = basic.policy(num_returns=2)(10, 20)
        assert r1.af_result(timeout=5) == 10
        assert r2.af_result(timeout=5) == 20
