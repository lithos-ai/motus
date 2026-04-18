from __future__ import annotations

import dataclasses
import logging
import traceback
import types
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar, overload

from ..runtime.agent_runtime import get_runtime
from ..runtime.hooks import HookCallback, hooks
from ..runtime.task_instance import DEFAULT_POLICY, TaskPolicy, capture_user_stack
from ..runtime.types import TASK, TaskType

if TYPE_CHECKING:
    from ..runtime.agent_future import AgentFuture

logger = logging.getLogger("AgentTask")

R = TypeVar("R")


def _register_task_hooks(
    task_name: str,
    on_start: HookCallback | list[HookCallback] | None,
    on_end: HookCallback | list[HookCallback] | None,
    on_error: HookCallback | list[HookCallback] | None,
) -> None:
    for event_type, cbs in (
        ("task_start", on_start),
        ("task_end", on_end),
        ("task_error", on_error),
    ):
        if cbs is None:
            continue
        if callable(cbs):
            cbs = [cbs]
        for cb in cbs:
            hooks.register_name_hook(task_name, event_type, cb)


class AgentTaskDefinition(Generic[R]):
    """A callable task definition returned by ``@agent_task``.

    Holds the original function, default policy, and exposes ``.policy()``
    for runtime configuration overrides.

    Implements the descriptor protocol (``__get__``) so that it works
    correctly when used to decorate class methods.
    """

    __slots__ = ("_fn", "_policy", "_num_returns", "_task_type", "__name__")

    def __init__(
        self,
        fn: Callable[..., R],
        *,
        policy: TaskPolicy = DEFAULT_POLICY,
        num_returns: int = 1,
        task_type: TaskType = TASK,
    ) -> None:
        self._fn = fn
        self._policy = policy
        self._num_returns = num_returns
        self._task_type = task_type
        self.__name__ = fn.__name__

    def __call__(self, *args, **kwargs) -> AgentFuture[R]:
        creation_stack = capture_user_stack()
        return register_agent_task(
            self._fn,
            *args,
            policy=self._policy,
            num_returns=self._num_returns,
            creation_stack=creation_stack,
            task_type=self._task_type,
            **kwargs,
        )

    def policy(
        self, *, num_returns: int | None = None, **kwargs
    ) -> AgentTaskDefinition[R]:
        """Return a copy with updated policy fields.

        Example::

            future = fetch_data.policy(retries=5, timeout=10)("http://api...")
        """
        new_policy = dataclasses.replace(self._policy, **kwargs)
        defn = AgentTaskDefinition.__new__(AgentTaskDefinition)
        defn._fn = self._fn
        defn._policy = new_policy
        defn._num_returns = (
            num_returns if num_returns is not None else self._num_returns
        )
        defn._task_type = self._task_type
        defn.__name__ = self.__name__
        return defn

    def unwrap_task(self) -> Callable:
        """Return the raw underlying function, bypassing the agent_task wrapper."""
        return self._fn

    # -- Descriptor protocol ---------------------------------------------------

    def __get__(self, obj, objtype=None):
        """Bind *obj* as the first argument when accessed as a method."""
        if obj is None:
            return self
        return _BoundAgentTask(self, obj)


class _BoundAgentTask(Generic[R]):
    """An AgentTaskDefinition bound to an instance (like a bound method)."""

    __slots__ = ("_definition", "_instance", "__name__")

    def __init__(self, definition: AgentTaskDefinition[R], instance: object) -> None:
        self._definition = definition
        self._instance = instance
        self.__name__ = definition.__name__

    def __call__(self, *args, **kwargs) -> AgentFuture[R]:
        return self._definition(self._instance, *args, **kwargs)

    def unwrap_task(self) -> Callable:
        """Return the raw function bound to the instance (a bound method).

        Preserves ismethod(), inspect.signature(), get_type_hints(),
        and iscoroutinefunction() — unlike functools.partial.
        """
        return types.MethodType(self._definition._fn, self._instance)

    def policy(self, **kwargs) -> _BoundAgentTask[R]:
        new_defn = self._definition.policy(**kwargs)
        return _BoundAgentTask(new_defn, self._instance)


class _AgentTaskDecorator(Protocol):
    """Return type for ``@agent_task(...)`` with parentheses."""

    @overload
    def __call__(
        self, func: Callable[..., Coroutine[Any, Any, R]], /
    ) -> AgentTaskDefinition[R]: ...
    @overload
    def __call__(self, func: Callable[..., R], /) -> AgentTaskDefinition[R]: ...
    def __call__(self, func): ...


# -- @agent_task  (no parentheses, async) --
@overload
def agent_task(
    func: Callable[..., Coroutine[Any, Any, R]], /
) -> AgentTaskDefinition[R]: ...


# -- @agent_task  (no parentheses, sync) --
@overload
def agent_task(func: Callable[..., R], /) -> AgentTaskDefinition[R]: ...


# -- @agent_task(retries=3, ...)  (with parentheses) --
@overload
def agent_task(
    func: None = None,
    *,
    task_type: TaskType = ...,
    retries: int = ...,
    timeout: float | None = ...,
    retry_delay: float = ...,
    num_returns: int = ...,
    on_start: HookCallback | list[HookCallback] | None = ...,
    on_end: HookCallback | list[HookCallback] | None = ...,
    on_error: HookCallback | list[HookCallback] | None = ...,
) -> _AgentTaskDecorator: ...


def agent_task(
    func=None,
    *,
    task_type: TaskType = TASK,
    retries: int = 0,
    timeout: float | None = None,
    retry_delay: float = 0.0,
    num_returns: int = 1,
    on_start: HookCallback | list[HookCallback] | None = None,
    on_end: HookCallback | list[HookCallback] | None = None,
    on_error: HookCallback | list[HookCallback] | None = None,
):
    """
    Decorator to register a task with the background loop.

    See ``examples/omni/gpuos_omni_demo.py`` for parallel task graphs with
    retries and timeouts, and ``apps/deep_research/researcher.py`` for
    sequential pipelines.

    Supports multiple forms::

        @agent_task
        async def my_task(...): ...

        @agent_task(on_start=cb1, on_end=[cb2, cb3])
        async def my_task(...): ...

        @agent_task(retries=3, timeout=10)
        def flaky_task(...): ...

        @agent_task(num_returns=2)
        def split(data):
            return part_a, part_b

        @agent_task(task_type="tool_call")
        async def __call__(self, args): ...
    """
    policy = TaskPolicy(retries=retries, timeout=timeout, retry_delay=retry_delay)

    def decorator(fn):
        logger.debug(f"Registering agent task: {fn.__name__}")

        _register_task_hooks(
            getattr(fn, "__qualname__", fn.__name__), on_start, on_end, on_error
        )

        return AgentTaskDefinition(
            fn, policy=policy, num_returns=num_returns, task_type=task_type
        )

    # @agent_task  (no parentheses)
    if func is not None:
        return decorator(func)
    # @agent_task(retries=3, on_start=..., ...)
    return decorator


def register_agent_task(
    func: Callable,
    *args,
    policy: TaskPolicy = DEFAULT_POLICY,
    num_returns: int = 1,
    creation_stack: traceback.StackSummary | None = None,
    on_start: HookCallback | list[HookCallback] | None = None,
    on_end: HookCallback | list[HookCallback] | None = None,
    on_error: HookCallback | list[HookCallback] | None = None,
    parent_stack: tuple[int, ...] | None = None,
    task_type: TaskType = TASK,
    **kwargs,
) -> AgentFuture | tuple[AgentFuture, ...]:
    """
    Register and submit a task directly (no decorator wrapper).

    Examples::

        # Direct registration without decorators
        future = register_agent_task(my_task, arg1, arg2, on_end=cb)
    """
    logger.debug(f"Registering agent task directly: {func.__name__}")

    _register_task_hooks(
        getattr(func, "__qualname__", func.__name__), on_start, on_end, on_error
    )

    # If it hasn't been inited yet, automatically initialize it
    rt = get_runtime()
    if parent_stack is None:
        from .tracing import get_stack

        parent_stack = get_stack()
    return rt.submit_task_registration(
        func,
        args,
        kwargs,
        parent_stack,
        policy=policy,
        num_returns=num_returns,
        creation_stack=creation_stack,
        task_type=task_type,
    )
