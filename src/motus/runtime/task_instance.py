from __future__ import annotations

import asyncio
import enum
import logging
import os
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .types import TASK, TaskType

if TYPE_CHECKING:
    from .agent_future import AgentFuture
    from .types import AgentTaskId

logger = logging.getLogger("AgentRuntime")


class TaskKind(enum.Enum):
    COMPUTE = "compute"  # Has a func to execute
    RESOLVE = "resolve"  # Pass-through: unwraps nested futures and sets result


class TaskStatus(enum.Enum):
    PENDING = "pending"  # Waiting for dependencies
    READY = "ready"  # All dependencies resolved, waiting to execute
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with an error
    CANCELLED = "cancelled"  # Cancelled by the user or via dependency cancellation


class TaskCancelledError(Exception):
    """Raised when a task is cancelled or depends on a cancelled task.

    This is a regular ``Exception`` (not ``BaseException``) so it can be
    carried by ``concurrent.futures.Future.set_exception()`` and propagates
    correctly through the dependency graph.
    """

    def __init__(self, task_name: str = "", task_id: int | None = None):
        self.task_name = task_name
        self.task_id = task_id
        msg = f"Task '{task_name}' was cancelled"
        if task_id is not None:
            msg += f" (id={task_id})"
        super().__init__(msg)


@dataclass(frozen=True)
class TaskPolicy:
    """Static configuration for task execution behaviour."""

    retries: int = 0
    timeout: float | None = None
    retry_delay: float = 0.0


DEFAULT_POLICY = TaskPolicy()


class TaskInstance:
    """The unified, stateful unit of work managed by the GraphScheduler."""

    __slots__ = (
        "id",
        "kind",
        "name",
        "func",
        "args",
        "kwargs",
        "result_futures",
        "num_returns",
        "status",
        "parent_stack",
        "policy",
        "creation_stack",
        "retry_count",
        "task_type",
        # Dependencies: keyed by id() to avoid triggering AgentFuture.__hash__/__eq__
        "_prerequisites",
        # Hook emission info: func, args, kwargs, task_id for task_end/task_error
        "hook_data",
        # Reference to the asyncio.Task running this task (for cancellation)
        "_asyncio_task",
    )

    def __init__(
        self,
        task_id: AgentTaskId,
        kind: TaskKind,
        result_futures: list[AgentFuture],
        *,
        name: str = "",
        func: Callable | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        parent_stack: tuple[int, ...] = (),
        policy: TaskPolicy = DEFAULT_POLICY,
        creation_stack: traceback.StackSummary | None = None,
        hook_data: dict | None = None,
        task_type: TaskType = TASK,
    ) -> None:
        self.id = task_id
        self.kind = kind
        self.name = name or (func.__name__ if func else "resolve")
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.result_futures = result_futures
        self.num_returns = len(self.result_futures)
        self.status = TaskStatus.PENDING
        self.parent_stack = parent_stack
        self.policy = policy
        self.creation_stack = creation_stack
        self.retry_count = 0
        self.task_type = task_type
        self.hook_data = hook_data
        self._asyncio_task: asyncio.Task | None = None

        # Scan all AgentFuture dependencies from args/kwargs
        deps = _scan_deps(self.args) + _scan_deps(self.kwargs)
        self._prerequisites: dict[int, AgentFuture] = {id(af): af for af in deps}

        if not self._prerequisites:
            self.status = TaskStatus.READY

    # -- Dependency tracking ------------------------------------------------

    @property
    def prerequisite_count(self) -> int:
        return len(self._prerequisites)

    def is_ready(self) -> bool:
        return len(self._prerequisites) == 0

    def dependency_resolved(self, finished_future: AgentFuture) -> bool:
        """Remove a resolved dependency. Return True if all dependencies are now met."""
        self._prerequisites.pop(id(finished_future), None)
        if self.is_ready():
            self.status = TaskStatus.READY
            return True
        return False

    # -- Argument unwrapping ------------------------------------------------

    def unwrap_args(self) -> tuple[tuple, dict]:
        """Unwrap all AgentFuture values in args/kwargs to their resolved results."""
        real_args = _deep_unwrap(self.args)
        real_kwargs = _deep_unwrap(self.kwargs)
        return real_args, real_kwargs

    def unwrap_value(self) -> Any:
        """For RESOLVE tasks: unwrap the result structure (stored in args[0])."""
        return _deep_unwrap(self.args[0])


# ---------------------------------------------------------------------------
# Module-level helpers (no dependency on TaskInstance for reuse)
# ---------------------------------------------------------------------------


def _scan_deps(structure: Any) -> list[AgentFuture]:
    """Recursively find all unresolved AgentFutures in a nested structure."""
    from .agent_future import AgentFuture

    deps: list[AgentFuture] = []
    if isinstance(structure, AgentFuture):
        if not structure.af_done():
            deps.append(structure)
    elif isinstance(structure, (list, tuple)):
        for item in structure:
            deps.extend(_scan_deps(item))
    elif isinstance(structure, dict):
        for value in structure.values():
            deps.extend(_scan_deps(value))
    return deps


def _deep_unwrap(structure: Any) -> Any:
    """Recursively replace AgentFuture values with their resolved results."""
    from .agent_future import AgentFuture

    if isinstance(structure, AgentFuture):
        return structure.af_result()
    elif isinstance(structure, list):
        return [_deep_unwrap(item) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(_deep_unwrap(item) for item in structure)
    elif isinstance(structure, dict):
        return {k: _deep_unwrap(v) for k, v in structure.items()}
    return structure


# ---------------------------------------------------------------------------
# Stack stitching helpers
# ---------------------------------------------------------------------------


# Path fragments used to identify non-user frames (motus internals + stdlib).
_FILTER_FRAGMENTS = (
    os.path.join("motus", "runtime") + os.sep,
    os.sep + "threading.py",
    os.sep + "concurrent" + os.sep,
    os.sep + "asyncio" + os.sep,
    os.sep + "importlib" + os.sep,
)


def _is_user_frame(filename: str) -> bool:
    """Return True if *filename* looks like user code (not stdlib/motus)."""
    return not any(frag in filename for frag in _FILTER_FRAGMENTS)


def capture_user_stack() -> traceback.StackSummary:
    """Capture the call stack, filtering out non-user frames.

    Returns only frames from user code so that error messages can point
    back to the line where a task was *created*, not where it was executed.
    """
    raw = traceback.extract_stack()
    # Drop the last frame (this function itself)
    raw = raw[:-1]
    return traceback.StackSummary.from_list(
        frame for frame in raw if _is_user_frame(frame.filename)
    )


_CHAIN_SEPARATOR = "=" * 60
_CHAIN_HEADER = _CHAIN_SEPARATOR + "\nTask creation chain (most recent call last):"


def stitch_creation_chain(
    exc: BaseException, stack: traceback.StackSummary | None
) -> None:
    """Accumulate *stack* into a unified creation-chain note on *exc*.

    Each task boundary calls this once.  Frames are prepended so that the
    outermost call site appears first (matching Python's traceback convention).
    The result is a **single** ``__notes__`` entry that reads like a normal
    traceback — easy to scan, no fragmentation.

    On the first call (innermost failure), user-code frames from the actual
    exception traceback are captured and appended at the tail of the chain,
    so the note ends at the exact line that raised.
    """
    if not stack:
        return

    # Accumulate raw frames on a private attribute.
    if not hasattr(exc, "_motus_creation_chain"):
        exc._motus_creation_chain = []
        # First call — capture the actual error site (user frames only).
        if exc.__traceback__:
            tb_frames = traceback.extract_tb(exc.__traceback__)
            exc._motus_error_frames = [
                f for f in tb_frames if _is_user_frame(f.filename)
            ]
        else:
            exc._motus_error_frames = []
    exc._motus_creation_chain = list(stack) + exc._motus_creation_chain

    # (Re)build the unified note: creation chain + error site at the tail.
    all_frames = exc._motus_creation_chain + getattr(exc, "_motus_error_frames", [])
    lines = [_CHAIN_HEADER]
    for frame in all_frames:
        lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        if frame.line:
            lines.append(f"    {frame.line}")
    lines.append(_CHAIN_SEPARATOR)
    note = "\n".join(lines)

    # Replace existing chain note or insert a new one.
    if not hasattr(exc, "__notes__"):
        exc.__notes__ = []
    exc.__notes__ = [n for n in exc.__notes__ if not n.startswith(_CHAIN_HEADER)]
    exc.__notes__.insert(0, note)
