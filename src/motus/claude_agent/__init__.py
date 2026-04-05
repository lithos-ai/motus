"""Claude Agent SDK compatibility layer for motus.

Re-exports the real claude-agent-sdk package. motus intercepts at two levels:
  - Hooks: tracing hooks injected into ClaudeAgentOptions for tool/agent lifecycle
  - Messages: AssistantMessage/ResultMessage observed for model call + cost tracing

Usage::

    from motus.claude_agent import query, ClaudeSDKClient, ClaudeAgentOptions
"""

from __future__ import annotations

import atexit
import logging

# Re-export everything from real Claude Agent SDK
try:
    from claude_agent_sdk import *  # noqa: F401,F403
    from claude_agent_sdk import (
        ClaudeAgentOptions as _OrigOptions,
    )
    from claude_agent_sdk import (
        ClaudeSDKClient as _OrigClient,
    )
    from claude_agent_sdk import (
        query as _orig_query,
    )
except ImportError as exc:
    raise ImportError(
        "motus.claude_agent requires the Claude Agent SDK. "
        "Install it with: uv pip install motus[claude-agent]"
    ) from exc

from ._tracing_hooks import inject_tracing_hooks

logger = logging.getLogger("AgentTracer")

_tracer = None


def register_tracing() -> None:
    """Create a standalone TraceManager for Claude Agent SDK tracing (idempotent)."""
    global _tracer
    if _tracer is not None:
        return

    try:
        from motus.runtime.tracing import TraceManager

        _tracer = TraceManager()
        atexit.register(_auto_export)
        logger.debug("Registered motus tracing for Claude Agent SDK")
    except Exception as e:
        logger.warning(f"Could not register motus tracing: {e}")


def get_tracer():
    """Get the TraceManager instance (call register_tracing() first)."""
    return _tracer


def _auto_export():
    """Auto-export traces on process exit (registered via atexit)."""
    if _tracer and _tracer.task_meta and _tracer.config.export_enabled:
        try:
            _tracer.export_trace()
        except Exception:
            # Don't crash during interpreter shutdown (e.g. KeyboardInterrupt
            # leaves the live server thread in a bad state)
            pass


# ── Streaming prompt conversion ──


async def _as_message_stream(prompt: str, done):
    """Convert string prompt to single-message AsyncIterable.

    Bypasses SDK's blocking wait_for_result_and_end_input() path
    (which delays all messages until session end when hooks are present)
    so hooks and consumer messages arrive interleaved in real-time.

    Stays alive until `done` event is set (on ResultMessage), then returns
    so stream_input() can close stdin naturally.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
    }
    # Keep stream alive until session completes.
    # When ResultMessage arrives, done is set, stream returns,
    # stream_input() proceeds to wait_for_result_and_end_input()
    # which closes stdin (ResultMessage already received, so no blocking).
    await done.wait()


# ── query() wrapper — async generator ──


async def query(*, prompt, options=None, transport=None):
    """Drop-in replacement for claude_agent_sdk.query() with motus tracing.

    Injects tracing hooks into options and observes each yielded message
    to record model calls and session summaries.
    """
    from claude_agent_sdk import ResultMessage as _ResultMessage

    if options is None:
        options = _OrigOptions()
    hook_state = inject_tracing_hooks(options, _tracer)

    # When tracing injects hooks and prompt is a string, convert to
    # AsyncIterable to avoid SDK's blocking wait_for_result_and_end_input().
    effective_prompt = prompt
    done_event = None
    if isinstance(prompt, str) and _tracer and _tracer.config.is_collecting:
        import anyio

        done_event = anyio.Event()
        effective_prompt = _as_message_stream(prompt, done_event)

    async for message in _orig_query(
        prompt=effective_prompt, options=options, transport=transport
    ):
        if _tracer:
            hook_state.observe_message(message, _tracer)
        if done_event and isinstance(message, _ResultMessage):
            done_event.set()
        yield message


# ── ClaudeSDKClient subclass ──


class ClaudeSDKClient(_OrigClient):  # type: ignore[misc]
    """Drop-in replacement for claude_agent_sdk.ClaudeSDKClient with motus tracing."""

    def __init__(self, options=None, transport=None):
        if options is None:
            options = _OrigOptions()
        self._hook_state = inject_tracing_hooks(options, _tracer)
        super().__init__(options=options, transport=transport)

    async def receive_messages(self):
        async for message in super().receive_messages():
            if _tracer:
                self._hook_state.observe_message(message, _tracer)
            yield message

    async def receive_response(self):
        async for message in super().receive_response():
            if _tracer:
                self._hook_state.observe_message(message, _tracer)
            yield message


# Auto-register tracing on import
register_tracing()
