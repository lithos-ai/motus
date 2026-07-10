"""Runtime-specific tracing helpers.

These symbols encode motus's *task-tree* concept (task IDs, parent task IDs,
the current-task ContextVar). They're separate from :mod:`motus.tracing`
because apps that don't use the motus runtime — framework integrations,
standalone OTel-decorated code, tests — have no notion of a motus task and
shouldn't inherit these attributes.

Producer: :mod:`motus.runtime.agent_runtime`'s ``wrapper()`` pushes the
current task id onto ``_current_task_id`` and stamps ``ATTR_TASK_ID`` /
``ATTR_PARENT_TASK_ID`` onto the span.

Consumer: :mod:`motus.tracing.span_convert` reads these attributes (with
``gen_ai.*`` fallbacks) to rebuild the task tree in the viewer.
"""

from __future__ import annotations

from contextvars import ContextVar

ATTR_TASK_ID = "motus.task_id_int"
ATTR_PARENT_TASK_ID = "motus.parent_task_id"

_current_task_id: ContextVar[int] = ContextVar("motus.current_task_id", default=-1)


def get_current_task_id() -> int:
    """Return the currently-running motus task id, or ``-1`` outside one."""
    return _current_task_id.get()


def get_stack() -> tuple[int, ...]:
    """Return the current task id as a single-element tuple (``()`` if none).

    Legacy shape: the runtime used to maintain an explicit ``tuple[int, ...]``
    stack. Parent propagation now rides the OTel context, so only the
    immediate parent matters — callers still read ``stack[-1]``.
    """
    tid = _current_task_id.get()
    return (tid,) if tid >= 0 else ()
