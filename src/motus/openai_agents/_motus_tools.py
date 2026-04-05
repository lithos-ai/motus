"""Wrap OAI SDK tool invocations so motus sits in the execution path.

Currently a transparent pass-through. Tracing is handled by OAI SDK's
TracingProcessor (see _motus_tracing.py).

Future hooks: tool-level caching, rate limiting, sandboxing, audit logging, etc.
"""

from __future__ import annotations

import functools


def _wrap_tools_for_motus(agent) -> None:
    """Wrap agent (and handoff targets) tool invocations for future interception."""
    visited: set[int] = set()
    _wrap_recursive(agent, visited)


def _wrap_recursive(agent, visited: set[int]) -> None:
    if id(agent) in visited:
        return
    visited.add(id(agent))

    for tool in getattr(agent, "tools", []):
        if not hasattr(tool, "on_invoke_tool"):
            continue
        original = tool.on_invoke_tool
        if getattr(original, "_motus_wrapped", False):
            continue  # already wrapped

        @functools.wraps(original)
        async def wrapped(ctx, input_str, _orig=original):
            # --- future hook: pre-tool ---
            # e.g. input validation, caching, sandboxing
            result = await _orig(ctx, input_str)
            # --- future hook: post-tool ---
            # e.g. result filtering, audit logging
            return result

        wrapped._motus_wrapped = True
        tool.on_invoke_tool = wrapped

    for h in getattr(agent, "handoffs", []):
        target = getattr(h, "agent", None) or (h if hasattr(h, "tools") else None)
        if target is not None:
            _wrap_recursive(target, visited)
