"""
Guardrail system for Motus agents and tools.

**Agent guardrails** receive ``(value: str)`` or ``(value: str, agent)``
and return ``str | None``.  The ``agent`` argument is only passed when
the guardrail declares it (by name or by accepting two positional params).

When an agent uses ``response_format`` (structured output), output guardrails
declare fields from the BaseModel — just like tool input guardrails::

    def validate_score(score: float, agent):
        if score < 0 or score > 1:
            raise OutputGuardrailTripped("Score out of range")

    def redact_raw(raw_data: str) -> dict:
        return {"raw_data": "[REDACTED]"}  # partial update

**Tool input guardrails** declare parameters matching the tool's signature —
the system extracts only the declared params via ``inspect.signature``::

    def reject_drop(query: str):
        if "DROP" in query.upper():
            raise ToolInputGuardrailTripped("DROP forbidden")

    def redact_token(token: str) -> dict:
        return {"token": "[REDACTED]"}  # partial update

Return ``None`` → no change. Return ``dict`` → partial update kwargs.
Raise → block execution.

**Tool output guardrails** receive the typed return value (before encoding)::

    def redact_password(result: str) -> str:
        return result.replace("hunter2", "***")

Return ``None`` → no change. Return a value → replace. Raise → block.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from asyncio import iscoroutinefunction
from typing import Any, Callable

logger = logging.getLogger("Guardrails")


# ── Exceptions ─────────────────────────────────────────────────────


class GuardrailTripped(Exception):
    """Base exception for all guardrail trips."""

    def __init__(self, message: str = "Guardrail tripped"):
        self.message = message
        super().__init__(message)


class InputGuardrailTripped(GuardrailTripped):
    """Raised when an input guardrail blocks execution."""


class OutputGuardrailTripped(GuardrailTripped):
    """Raised when an output guardrail blocks execution."""


class ToolInputGuardrailTripped(GuardrailTripped):
    """Raised when a tool input guardrail blocks execution."""


class ToolOutputGuardrailTripped(GuardrailTripped):
    """Raised when a tool output guardrail blocks execution."""


# ── Execution engine ───────────────────────────────────────────────


async def _execute_guardrail(fn: Callable, value: str, agent: Any = None) -> str | None:
    """Execute a single guardrail function."""
    pass_agent = agent is not None and "agent" in inspect.signature(fn).parameters
    args = (value, agent) if pass_agent else (value,)
    if iscoroutinefunction(fn):
        return await fn(*args)
    else:
        return await asyncio.to_thread(fn, *args)


async def run_guardrails(
    guardrails: list[Callable],
    value: str,
    *,
    agent: Any = None,
    parallel: bool = False,
) -> str:
    """Run a list of guardrails and return the (possibly modified) value.

    Agent guardrails receive ``(value, agent)``.
    Tool guardrails receive ``(value,)`` only (no *agent*).

    Args:
        guardrails: Ordered list of guardrail functions.
        value: The current value to validate/transform.
        agent: The agent instance.  Passed as the second positional
            argument to each guardrail when not ``None``.
        parallel: If ``True``, all guardrails run concurrently on
            the *original* value — only tripwires take effect.
            If ``False`` (default), guardrails run sequentially
            and each one sees the output of the previous one.

    Returns:
        The (possibly modified) value after all guardrails pass.

    Raises:
        GuardrailTripped: If any guardrail triggers a tripwire.
    """
    if parallel:
        await asyncio.gather(*[_execute_guardrail(g, value, agent) for g in guardrails])
        return value

    for g in guardrails:
        result = await _execute_guardrail(g, value, agent)
        if result is not None:
            value = result
    return value


# ── Tool guardrails (typed signatures) ───────────────────────────


async def _execute_tool_input_guardrail(fn: Callable, kwargs: dict) -> dict | None:
    """Execute one tool input guardrail.

    The guardrail receives only the kwargs it declares.
    Returns ``dict`` (partial update) or ``None`` (no change).
    Skipped (returns ``None``) when required parameters are missing from kwargs.
    """
    sig = inspect.signature(fn)
    matched = {k: kwargs[k] for k in sig.parameters if k in kwargs}
    # Skip if any required param is missing
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and name not in matched:
            return None
    if iscoroutinefunction(fn):
        result = await fn(**matched)
    else:
        result = await asyncio.to_thread(fn, **matched)
    return result


async def _execute_tool_output_guardrail(fn: Callable, value: Any) -> Any:
    """Execute one tool output guardrail on the typed result."""
    if iscoroutinefunction(fn):
        return await fn(value)
    else:
        return await asyncio.to_thread(fn, value)


async def run_tool_input_guardrails(guardrails: list[Callable], kwargs: dict) -> dict:
    """Run tool input guardrails sequentially, merging ``dict`` updates.

    Called from async context inside ``FunctionTool.__call__``.
    """
    for g in guardrails:
        result = await _execute_tool_input_guardrail(g, kwargs)
        if result is not None:
            kwargs = {**kwargs, **result}
    return kwargs


async def run_tool_output_guardrails(guardrails: list[Callable], value: Any) -> Any:
    """Run tool output guardrails sequentially on the typed result.

    Called from async context inside ``FunctionTool.__call__``.
    """
    for g in guardrails:
        result = await _execute_tool_output_guardrail(g, value)
        if result is not None:
            value = result
    return value


# ── Structured output guardrails (BaseModel agent results) ───────


async def _execute_structured_guardrail(
    fn: Callable, kwargs: dict, agent: Any = None
) -> dict | None:
    """Execute one guardrail on structured (BaseModel) output.

    The guardrail declares fields it cares about from the model.
    If it also declares an ``agent`` parameter, the agent is passed in.
    Returns ``dict`` (partial update) or ``None`` (no change).
    Skipped (returns ``None``) when required field parameters are missing.
    """
    sig = inspect.signature(fn)
    matched = {k: kwargs[k] for k in sig.parameters if k in kwargs and k != "agent"}
    # Check required params (excluding 'agent' which comes from the agent instance)
    for name, param in sig.parameters.items():
        if name == "agent":
            continue
        if param.default is inspect.Parameter.empty and name not in matched:
            return None
    if "agent" in sig.parameters and agent is not None:
        matched["agent"] = agent
    if iscoroutinefunction(fn):
        result = await fn(**matched)
    else:
        result = await asyncio.to_thread(fn, **matched)
    return result


async def run_structured_output_guardrails(
    guardrails: list[Callable], kwargs: dict, *, agent: Any = None
) -> dict:
    """Run output guardrails on a structured (BaseModel) result.

    Each guardrail declares fields it cares about.  Returns ``None``
    for no change, ``dict`` for partial update, or raises to block.
    """
    for g in guardrails:
        result = await _execute_structured_guardrail(g, kwargs, agent)
        if result is not None:
            kwargs = {**kwargs, **result}
    return kwargs
