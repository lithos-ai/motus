"""Tests for AgentFuture magic method overloading."""

import asyncio
import warnings

import pytest

from motus.runtime.agent_future import resolve
from motus.runtime.agent_task import agent_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@agent_task
async def returns(value):
    """Task that returns a given value."""
    return value


@agent_task
async def returns_delayed(value, delay=0.05):
    """Task that returns a value after a short delay."""
    await asyncio.sleep(delay)
    return value


# ---------------------------------------------------------------------------
# Non-blocking arithmetic
# ---------------------------------------------------------------------------


class TestNonBlockingArithmetic:
    def test_add(self):
        f = returns(10)
        result = resolve(f + 5)
        assert result == 15

    def test_sub(self):
        f = returns(10)
        result = resolve(f - 3)
        assert result == 7

    def test_mul(self):
        f = returns(4)
        result = resolve(f * 3)
        assert result == 12

    def test_truediv(self):
        f = returns(10)
        result = resolve(f / 4)
        assert result == 2.5

    def test_floordiv(self):
        f = returns(10)
        result = resolve(f // 3)
        assert result == 3

    def test_mod(self):
        f = returns(10)
        result = resolve(f % 3)
        assert result == 1

    def test_radd(self):
        f = returns(10)
        result = resolve(5 + f)
        assert result == 15

    def test_rsub(self):
        f = returns(3)
        result = resolve(10 - f)
        assert result == 7

    def test_rmul(self):
        f = returns(4)
        result = resolve(3 * f)
        assert result == 12

    def test_rtruediv(self):
        f = returns(4)
        result = resolve(10 / f)
        assert result == 2.5

    def test_rfloordiv(self):
        f = returns(3)
        result = resolve(10 // f)
        assert result == 3

    def test_rmod(self):
        f = returns(3)
        result = resolve(10 % f)
        assert result == 1

    def test_neg(self):
        f = returns(5)
        result = resolve(-f)
        assert result == -5

    def test_pos(self):
        f = returns(-5)
        result = resolve(+f)
        assert result == -5

    def test_abs(self):
        f = returns(-5)
        result = resolve(abs(f))
        assert result == 5

    def test_chained_arithmetic(self):
        f = returns(10)
        result = resolve((f + 5) * 2)
        assert result == 30

    def test_chained_mixed(self):
        f = returns(100)
        result = resolve((f - 10) // 3)
        assert result == 30


# ---------------------------------------------------------------------------
# Non-blocking item access
# ---------------------------------------------------------------------------


class TestNonBlockingItemAccess:
    def test_getitem_list(self):
        f = returns([10, 20, 30])
        result = resolve(f[1])
        assert result == 20

    def test_getitem_dict(self):
        f = returns({"key": "value"})
        result = resolve(f["key"])
        assert result == "value"

    def test_getitem_chained(self):
        f = returns({"data": [1, 2, 3]})
        result = resolve(f["data"][2])
        assert result == 3

    def test_getattr(self):
        f = returns("hello")
        result = resolve(f.upper())
        assert result == "HELLO"

    def test_getattr_chained(self):
        f = returns("  hello  ")
        result = resolve(f.strip().upper())
        assert result == "HELLO"


# ---------------------------------------------------------------------------
# Non-blocking comparisons
# ---------------------------------------------------------------------------


class TestNonBlockingComparisons:
    def test_gt(self):
        f = returns(10)
        assert resolve(f > 5) is True

    def test_gt_false(self):
        f = returns(3)
        assert resolve(f > 5) is False

    def test_lt(self):
        f = returns(3)
        assert resolve(f < 5) is True

    def test_ge(self):
        f = returns(5)
        assert resolve(f >= 5) is True

    def test_le(self):
        f = returns(5)
        assert resolve(f <= 5) is True

    def test_comparison_in_if(self, monkeypatch):
        """Comparison + bool: if future > 5 triggers __bool__(__gt__)."""
        monkeypatch.setattr("motus.runtime.agent_future._QUIET_SYNC", True)
        f = returns(10)
        if f > 5:
            passed = True
        else:
            passed = False
        assert passed is True

    def test_comparison_in_if_false(self, monkeypatch):
        monkeypatch.setattr("motus.runtime.agent_future._QUIET_SYNC", True)
        f = returns(3)
        if f > 5:
            passed = True
        else:
            passed = False
        assert passed is False


# ---------------------------------------------------------------------------
# Blocking methods (sync barriers)
# ---------------------------------------------------------------------------


class TestBlockingMethods:
    @pytest.fixture(autouse=True)
    def quiet_sync(self, monkeypatch):
        monkeypatch.setattr("motus.runtime.agent_future._QUIET_SYNC", True)

    def test_bool_true(self):
        f = returns(42)
        assert bool(f) is True

    def test_bool_false(self):
        f = returns(0)
        assert bool(f) is False

    def test_int(self):
        f = returns(3.7)
        assert int(f) == 3

    def test_float(self):
        f = returns(5)
        assert float(f) == 5.0

    def test_str(self):
        f = returns(42)
        assert str(f) == "42"

    def test_len(self):
        f = returns([1, 2, 3])
        assert len(f) == 3

    def test_iter(self):
        f = returns([1, 2, 3])
        assert list(f) == [1, 2, 3]

    def test_contains(self):
        f = returns([1, 2, 3])
        assert 2 in f
        assert 99 not in f

    def test_eq(self):
        f = returns(42)
        assert f == 42

    def test_ne(self):
        f = returns(42)
        assert f != 99

    def test_hash(self):
        f = returns(42)
        assert hash(f) == hash(42)

    def test_if_future(self):
        """Plain `if future:` triggers __bool__."""
        f = returns("nonempty")
        if f:
            passed = True
        else:
            passed = False
        assert passed is True


# ---------------------------------------------------------------------------
# Sync barrier warnings
# ---------------------------------------------------------------------------


class TestSyncBarrierWarnings:
    def test_bool_emits_warning(self):
        f = returns(1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bool(f)
        assert any("__bool__" in str(x.message) for x in w)

    def test_str_emits_warning(self):
        f = returns(1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            str(f)
        assert any("__str__" in str(x.message) for x in w)

    def test_eq_emits_warning(self):
        f = returns(42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = f == 42
        assert any("__eq__" in str(x.message) for x in w)

    def test_quiet_sync_suppresses(self, monkeypatch):
        monkeypatch.setattr("motus.runtime.agent_future._QUIET_SYNC", True)
        f = returns(1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bool(f)
        sync_warnings = [x for x in w if "implicitly blocked" in str(x.message)]
        assert len(sync_warnings) == 0


# ---------------------------------------------------------------------------
# Async context detection
# ---------------------------------------------------------------------------


class TestAsyncContextDetection:
    def test_non_runtime_async_context_warns(self):
        """Calling af_result() from a non-runtime async context emits a warning."""

        async def run():
            f = returns_delayed(42, delay=0.1)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = f.af_result()
            return result, w

        result, w = asyncio.run(run())
        assert result == 42
        assert any("async context" in str(x.message) for x in w)

    def test_runtime_loop_raises(self):
        """Calling af_result() from the runtime event loop raises RuntimeError."""
        import pytest

        @agent_task
        async def blocker():
            f = returns_delayed(42, delay=0.1)
            f.af_result()  # should raise — we ARE on the runtime loop

        with pytest.raises(RuntimeError, match="runtime event loop"):
            resolve(blocker(), timeout=5)


# ---------------------------------------------------------------------------
# Exception propagation through dependents
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_exception_propagates_to_dependent(self):
        """When a task fails, the exception propagates to dependent futures."""

        @agent_task
        async def delayed_fail():
            await asyncio.sleep(0.1)
            raise ValueError("boom")

        f = delayed_fail()
        # Create a dependent before the failure happens
        g = f + 1

        with pytest.raises(ValueError, match="boom"):
            resolve(g, timeout=5)

    def test_exception_propagates_through_chain(self):
        """Exception propagates through a chain of dependents."""

        @agent_task
        async def delayed_fail():
            await asyncio.sleep(0.1)
            raise ValueError("chain-boom")

        f = delayed_fail()
        g = f + 1
        h = g * 2

        with pytest.raises(ValueError, match="chain-boom"):
            resolve(h, timeout=5)


# ---------------------------------------------------------------------------
# Callable deferral
# ---------------------------------------------------------------------------


class TestCallDeferral:
    def test_call(self):
        f = returns(str.upper)
        result = resolve(f("hello"))
        assert result == "HELLO"

    def test_getattr_then_call(self):
        f = returns("hello world")
        result = resolve(f.split(" "))
        assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# Await support
# ---------------------------------------------------------------------------


class TestAwaitSupport:
    def test_await_in_agent_task(self):
        @agent_task
        async def outer():
            inner = returns_delayed(99)
            return await inner

        assert resolve(outer()) == 99

    def test_await_chained(self):
        @agent_task
        async def outer():
            f = returns_delayed([1, 2, 3])
            val = await f
            return sum(val)

        assert resolve(outer()) == 6
