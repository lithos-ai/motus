from __future__ import annotations

import asyncio
import logging
import operator
import os
import warnings
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Generic, Sequence, TypeVar, Union, overload

from .types import MAGIC_TASK, AgentFutureId

T = TypeVar("T")

logger = logging.getLogger("AgentFuture")

_QUIET_SYNC = os.environ.get("MOTUS_QUIET_SYNC", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _call_op(fn, *args, **kwargs):
    """Helper for __call__ deferral — calls resolved value with given args."""
    return fn(*args, **kwargs)


class AgentFuture(Generic[T]):
    def __init__(
        self,
        agent_future_id: int,
        triggered_task=None,
        future: ConcurrentFuture[T] = None,
    ):
        self._agent_future_id = AgentFutureId(agent_future_id)
        self._triggered_task = triggered_task
        self._future: ConcurrentFuture[T] = future or ConcurrentFuture()

    # ===================================================================
    # Public API — all user-facing methods carry the ``af_`` prefix to
    # avoid shadowing same-named attributes on the wrapped result value
    # ===================================================================

    def af_result(self, timeout: float = None) -> T:
        """Block until the result is available and return it."""
        return self._wait_for_result(timeout=timeout)

    def af_done(self) -> bool:
        """Return ``True`` if the future is resolved (success, error, or cancelled)."""
        return self._future.done()

    def af_cancel(self) -> bool:
        """Cancel the task that produces this future.

        If the future is already resolved, returns ``False``.  Otherwise,
        cancels the owning task and all downstream dependents, then
        returns ``True``.

        This method is **thread-safe** and may be called from any thread,
        including from within an ``@agent_task``.
        """
        if self._future.done():
            return False

        from .agent_runtime import get_runtime

        rt = get_runtime()
        rt.cancel_future(self)
        return True

    def af_cancelled(self) -> bool:
        """Return ``True`` if this future was cancelled."""
        from .task_instance import TaskCancelledError

        if not self._future.done():
            return False
        try:
            self._future.result()
            return False
        except TaskCancelledError:
            return True
        except Exception:
            return False

    def __await__(self):  # -> Generator[Any, None, T]
        return asyncio.wrap_future(self._future).__await__()

    # ===================================================================
    # Internal API — used by the scheduler / runtime only.
    # ===================================================================

    def _set_result(self, result: T) -> None:
        self._future.set_result(result)

    def _set_error(self, error: Exception) -> None:
        self._future.set_exception(error)

    def _set_exception(self, error: Exception) -> None:
        self._future.set_exception(error)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _wait_for_result(self, timeout: float = None, op_name: str | None = None) -> T:
        from .agent_runtime import _runtime

        if self._future.done():
            return self._future.result()

        # Detect async context — blocking the runtime loop would deadlock.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and _runtime is not None and loop is _runtime._loop:
            # On the runtime's own event loop — guaranteed deadlock.
            label = f"AgentFuture.{op_name}()" if op_name else "resolve()"
            raise RuntimeError(
                f"{label} called from the runtime event loop — "
                f"this may lead to a deadlock. Use 'await future' instead."
            )

        if loop is not None:
            # Other async context (e.g. pytest IsolatedAsyncioTestCase).
            # Not a deadlock, but freezes the caller's loop during the wait.
            label = f"AgentFuture.{op_name}()" if op_name else "resolve()"
            warnings.warn(
                f"{label} called from an async context — "
                f"this may block the current event loop. "
                f"Use 'await future' instead.",
                stacklevel=3,
            )

        return self._future.result(timeout=timeout)

    def _sync_barrier_warn(self, op_name: str):
        """Emit a warning when an implicit sync barrier is hit."""
        if not _QUIET_SYNC:
            warnings.warn(
                f"AgentFuture.{op_name}() implicitly blocked to resolve value. "
                f"Set MOTUS_QUIET_SYNC=1 to suppress.",
                stacklevel=3,
            )

    def _defer_op(self, func, *args, **kwargs) -> AgentFuture[Any]:
        """Register a new task that depends on self, return a new AgentFuture."""
        from .agent_task import register_agent_task

        return register_agent_task(func, self, *args, task_type=MAGIC_TASK, **kwargs)

    # -----------------------------------------------------------------------
    # Debug / introspection (non-blocking)
    # -----------------------------------------------------------------------

    def __repr__(self):
        return (
            f"AgentFuture(id={self._agent_future_id.id}, "
            f"task={self._triggered_task!r}, done={self._future.done()})"
        )

    # ===================================================================
    # Blocking methods (sync barriers) — resolve then return concrete value
    # ===================================================================

    def __bool__(self):
        self._sync_barrier_warn("__bool__")
        return bool(self._wait_for_result(op_name="__bool__"))

    def __str__(self):
        self._sync_barrier_warn("__str__")
        return str(self._wait_for_result(op_name="__str__"))

    def __len__(self):
        self._sync_barrier_warn("__len__")
        return len(self._wait_for_result(op_name="__len__"))

    def __iter__(self):
        self._sync_barrier_warn("__iter__")
        return iter(self._wait_for_result(op_name="__iter__"))

    def __int__(self):
        self._sync_barrier_warn("__int__")
        return int(self._wait_for_result(op_name="__int__"))

    def __float__(self):
        self._sync_barrier_warn("__float__")
        return float(self._wait_for_result(op_name="__float__"))

    def __eq__(self, other):
        self._sync_barrier_warn("__eq__")
        return self._wait_for_result(op_name="__eq__") == other

    def __ne__(self, other):
        self._sync_barrier_warn("__ne__")
        return self._wait_for_result(op_name="__ne__") != other

    def __hash__(self):
        self._sync_barrier_warn("__hash__")
        return hash(self._wait_for_result(op_name="__hash__"))

    def __contains__(self, item):
        self._sync_barrier_warn("__contains__")
        return item in self._wait_for_result(op_name="__contains__")

    # ===================================================================
    # Non-blocking methods (graph extensions) — return a new AgentFuture
    # ===================================================================

    def __getattr__(self, name: str) -> AgentFuture[Any]:
        return self._defer_op(getattr, name)

    def __call__(self, *args, **kwargs) -> AgentFuture[Any]:
        return self._defer_op(_call_op, *args, **kwargs)

    def __getitem__(self, key) -> AgentFuture[Any]:
        return self._defer_op(operator.getitem, key)

    # --- Binary arithmetic ---

    def __add__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.add, other)

    def __sub__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.sub, other)

    def __mul__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.mul, other)

    def __truediv__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.truediv, other)

    def __floordiv__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.floordiv, other)

    def __mod__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.mod, other)

    # --- Reverse arithmetic (e.g. 5 + future) ---

    def __radd__(self, other) -> AgentFuture[Any]:
        return self._defer_op(lambda x, y: y + x, other)

    def __rsub__(self, other) -> AgentFuture[Any]:
        return self._defer_op(lambda x, y: y - x, other)

    def __rmul__(self, other) -> AgentFuture[Any]:
        return self._defer_op(lambda x, y: y * x, other)

    def __rtruediv__(self, other) -> AgentFuture[Any]:
        return self._defer_op(lambda x, y: y / x, other)

    def __rfloordiv__(self, other) -> AgentFuture[Any]:
        return self._defer_op(lambda x, y: y // x, other)

    def __rmod__(self, other) -> AgentFuture[Any]:
        return self._defer_op(lambda x, y: y % x, other)

    # --- Ordering comparisons (non-blocking, __bool__ handles sync) ---

    def __gt__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.gt, other)

    def __lt__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.lt, other)

    def __ge__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.ge, other)

    def __le__(self, other) -> AgentFuture[Any]:
        return self._defer_op(operator.le, other)

    # --- Unary operators ---

    def __neg__(self) -> AgentFuture[Any]:
        return self._defer_op(operator.neg)

    def __pos__(self) -> AgentFuture[Any]:
        return self._defer_op(operator.pos)

    def __abs__(self) -> AgentFuture[Any]:
        return self._defer_op(abs)


# ═══════════════════════════════════════════════════════════════════════════
# Standalone functions — the preferred public API.
# ═══════════════════════════════════════════════════════════════════════════


@overload
def resolve(future: AgentFuture[T], *, timeout: float = ...) -> T: ...


@overload
def resolve(
    future: Sequence[AgentFuture[Any]], *, timeout: float = ...
) -> list[Any]: ...


def resolve(
    future: Union[AgentFuture[T], Sequence[AgentFuture[Any]]],
    *,
    timeout: float = None,
) -> Union[T, list[Any]]:
    """Block until one or more futures are resolved and return their values.

    Accepts a single ``AgentFuture`` or a sequence of them.  When given a
    sequence, all futures are waited on and a list of results is returned
    (order preserved).

    Examples::

        val = resolve(my_task(1, 2))
        a, b, c = resolve([task_a(), task_b(), task_c()])
    """
    if isinstance(future, AgentFuture):
        return future.af_result(timeout=timeout)
    return [f.af_result(timeout=timeout) for f in future]


def cancel(future: AgentFuture) -> bool:
    """Cancel the task producing *future* and all its downstream dependents.

    Returns ``True`` if cancellation was requested, ``False`` if the future
    was already resolved.
    """
    return future.af_cancel()


def cancelled(future: AgentFuture) -> bool:
    """Return ``True`` if *future* was cancelled."""
    return future.af_cancelled()
