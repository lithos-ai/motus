"""
Example usage:

from motus.runtime.hooks import register_hook, register_task_hook, HookEvent

# Global hook — fires for every agent_task
def on_task_end(event: HookEvent) -> None:
    print(f"[task_end] {event.name} result={event.result}")

register_hook("task_end", on_task_end)

# Per-name hook — fires only for "web_search"
from motus.runtime.hooks import task_hook

@task_hook("web_search", "task_end")
def on_web_search_end(event: HookEvent) -> None:
    print(f"[web_search_end] result={event.result}")

# Per-type hook — fires for ALL tool calls
from motus.runtime.hooks import type_hook, TOOL_CALL

@type_hook(TOOL_CALL, "task_end")
def on_any_tool_end(event: HookEvent) -> None:
    print(f"[tool_end] {event.name} result={event.result}")
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Literal

from .types import AGENT_CALL, MODEL_CALL, TOOL_CALL, AgentTaskId, TaskType

logger = logging.getLogger("HookManager")

HookType = Literal["task_start", "task_end", "task_error", "task_cancelled"]


@dataclass(frozen=True)
class HookEvent:
    event_type: HookType
    name: str | None = None
    task_type: TaskType | None = None
    args: tuple | None = None
    kwargs: dict | None = None
    result: Any | None = None
    error: Exception | None = None
    task_id: AgentTaskId | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


HookCallback = Callable[[HookEvent], Any | Awaitable[Any]]


def _append_callbacks(
    target: list[HookCallback],
    callback: HookCallback | List[HookCallback],
    prepend: bool,
) -> None:
    if callable(callback):
        callback = [callback]
    if prepend:
        target[:0] = callback
    else:
        target.extend(callback)


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[HookType, list[HookCallback]] = {}
        # Matched against event.name (e.g., "web_search", "model_serve_task")
        self._name_hooks: Dict[str, Dict[HookType, list[HookCallback]]] = {}
        # Matched against event.task_type (e.g., "tool_call", "model_call")
        self._type_hooks: Dict[str, Dict[HookType, list[HookCallback]]] = {}

    # ── Global hooks ──────────────────────────────────────────────

    def register(
        self, event_type: HookType, callback: HookCallback, prepend: bool = False
    ) -> None:
        hooks = self._hooks.setdefault(event_type, [])
        if prepend:
            hooks.insert(0, callback)
        else:
            hooks.append(callback)

    def list_hooks(self, event_type: HookType) -> Iterable[HookCallback]:
        return list(self._hooks.get(event_type, []))

    # ── Per-name hooks (matched by event.name) ────────────────────

    def register_name_hook(
        self,
        name: str | Callable,
        event_type: HookType,
        callback: HookCallback | List[HookCallback],
        prepend: bool = False,
    ) -> None:
        if callable(name):
            name = getattr(name, "__qualname__", name.__name__)
        per_type = self._name_hooks.setdefault(name, {})
        target = per_type.setdefault(event_type, [])
        _append_callbacks(target, callback, prepend)

    def list_name_hooks(
        self, name: str, event_type: HookType
    ) -> Iterable[HookCallback]:
        return list(self._name_hooks.get(name, {}).get(event_type, []))

    # ── Per-type hooks (matched by event.task_type) ───────────────

    def register_type_hook(
        self,
        task_type: str,
        event_type: HookType,
        callback: HookCallback | List[HookCallback],
        prepend: bool = False,
    ) -> None:
        per_type = self._type_hooks.setdefault(task_type, {})
        target = per_type.setdefault(event_type, [])
        _append_callbacks(target, callback, prepend)

    def list_type_hooks(
        self, task_type: str, event_type: HookType
    ) -> Iterable[HookCallback]:
        return list(self._type_hooks.get(task_type, {}).get(event_type, []))

    # ── Deregistration ─────────────────────────────────────────────

    def deregister(self, event_type: HookType, callback: HookCallback) -> None:
        """Remove a specific callback from a global hook."""
        hooks_list = self._hooks.get(event_type, [])
        try:
            hooks_list.remove(callback)
        except ValueError:
            pass

    # ── Emit ──────────────────────────────────────────────────────

    async def emit(self, event: HookEvent) -> None:
        # 1. Global hooks
        callbacks = list(self.list_hooks(event.event_type))
        # 2. Name hooks (e.g., "web_search", "model_serve_task")
        if event.name:
            callbacks += list(self.list_name_hooks(event.name, event.event_type))
        # 3. Type hooks (e.g., "tool_call", "model_call")
        if event.task_type:
            callbacks += list(self.list_type_hooks(event.task_type, event.event_type))
        if not callbacks:
            return
        for callback in callbacks:
            try:
                result = callback(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Hook callback failed: %s", callback)


hooks = HookManager()


async def emit_on_task(
    event_type: HookType,
    func: Callable,
    args: tuple | None = None,
    kwargs: dict | None = None,
    result: Any | None = None,
    error: Exception | None = None,
    task_id: AgentTaskId | None = None,
    metadata: Dict[str, Any] | None = None,
    task_type: TaskType | None = None,
) -> Awaitable[None]:
    if metadata is None:
        metadata = {}
    # Determine task name for per-name hook matching.
    # For class methods (qualname contains '.'), args[0] is the instance (self).
    # If it has a .name attribute, use that (e.g., "web_search" for FunctionTool,
    # agent name for AgentBase). For standalone functions, use the qualname.
    hook_name: str | None = None
    qualname = getattr(func, "__qualname__", func.__name__)
    if args and "." in qualname and getattr(args[0], "name", None):
        hook_name = args[0].name
    if hook_name is None:
        hook_name = qualname

    await hooks.emit(
        HookEvent(
            event_type=event_type,
            name=hook_name,
            task_type=task_type,
            args=args,
            kwargs=kwargs,
            result=result,
            error=error,
            task_id=task_id,
            metadata=metadata,
        )
    )


# ── Global registration ───────────────────────────────────────────


def register_hook(
    event_type: HookType, callback: HookCallback, prepend: bool = False
) -> None:
    """Register a global hook. See ``apps/deep_research/researcher.py``."""
    hooks.register(event_type, callback, prepend=prepend)


def global_hook(event_type: HookType, prepend: bool = False):
    """Decorator to register a global hook callback for an event type."""

    def decorator(callback: HookCallback):
        register_hook(event_type, callback, prepend=prepend)
        return callback

    return decorator


# ── Per-name registration (matched by event.name) ────────────────


def register_task_hook(
    task_name: str | Callable,
    event_type: HookType,
    callback: HookCallback,
    prepend: bool = False,
) -> None:
    hooks.register_name_hook(task_name, event_type, callback, prepend=prepend)


def task_hook(task_name: str | Callable, event_type: HookType, prepend: bool = False):
    """Decorator to register a hook matched by task **name**.

    The name is the specific function/tool name (e.g., ``"web_search"``,
    ``"model_serve_task"``).
    """

    def decorator(callback: HookCallback):
        register_task_hook(task_name, event_type, callback, prepend=prepend)
        return callback

    return decorator


# ── Per-type registration (matched by event.task_type) ───────────


def register_type_hook(
    task_type: str,
    event_type: HookType,
    callback: HookCallback,
    prepend: bool = False,
) -> None:
    hooks.register_type_hook(task_type, event_type, callback, prepend=prepend)


def type_hook(task_type: str, event_type: HookType, prepend: bool = False):
    """Decorator to register a hook matched by task **type**.

    The type is the category (e.g., ``"tool_call"``, ``"model_call"``).
    """

    def decorator(callback: HookCallback):
        register_type_hook(task_type, event_type, callback, prepend=prepend)
        return callback

    return decorator


# ── Convenience registration for built-in task types ──────────────


def register_model_hook(
    event_type: HookType, callback: HookCallback, prepend: bool = False
) -> None:
    """Register a hook that fires for all model calls."""
    register_type_hook(MODEL_CALL, event_type, callback, prepend=prepend)


def register_tool_hook(
    event_type: HookType, callback: HookCallback, prepend: bool = False
) -> None:
    """Register a hook that fires for all tool calls."""
    register_type_hook(TOOL_CALL, event_type, callback, prepend=prepend)


def register_agent_hook(
    event_type: HookType, callback: HookCallback, prepend: bool = False
) -> None:
    """Register a hook that fires for all agent calls."""
    register_type_hook(AGENT_CALL, event_type, callback, prepend=prepend)


# ── Convenience decorators for built-in task types ────────────────


def model_task_hook(event_type: HookType, prepend: bool = False):
    """Decorator to register a hook that fires for all model calls."""
    return type_hook(MODEL_CALL, event_type, prepend=prepend)


def tool_task_hook(event_type: HookType, prepend: bool = False):
    """Decorator to register a hook that fires for all tool calls."""
    return type_hook(TOOL_CALL, event_type, prepend=prepend)


def agent_task_hook(event_type: HookType, prepend: bool = False):
    """Decorator to register a hook that fires for all agent calls."""
    return type_hook(AGENT_CALL, event_type, prepend=prepend)
