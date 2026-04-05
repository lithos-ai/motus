"""Motus-enhanced Google ADK LlmAgent.

Subclasses the Google ADK ``Agent`` and adds a ``run_turn`` method that
satisfies the motus serve agent contract::

    async def run_turn(message: ChatMessage, state: list[ChatMessage])
        -> tuple[ChatMessage, list[ChatMessage]]

This allows a Google ADK agent to be served directly via
``motus serve start``.
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING

try:
    from google.adk.agents.llm_agent import Agent as _ADKAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
except ImportError as exc:
    raise ImportError(
        "motus.google_adk requires the Google ADK. "
        "Install it with: uv pip install motus[google-adk]"
    ) from exc

if TYPE_CHECKING:
    from motus.models import ChatMessage

logger = logging.getLogger("AgentTracer")

_APP_NAME = "motus"
_USER_ID = "motus_user"
_SESSION_ID = "motus_session"

_tracer = None


def register_tracing() -> None:
    """Register MotusSpanProcessor with Google ADK's OTEL providers (idempotent).

    Creates a standalone TraceManager and installs a MotusSpanProcessor so
    that ADK's OTEL spans (agent invocations, LLM calls, tool executions) are
    ingested into the motus trace viewer / analytics pipeline.
    """
    global _tracer
    if _tracer is not None:
        return

    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider

        from motus.google_adk._motus_tracing import MotusSpanProcessor
        from motus.runtime.tracing import TraceManager

        _tracer = TraceManager()
        processor = MotusSpanProcessor(_tracer)

        # Add our processor to the existing TracerProvider if one is already set
        # (ADK or another library may have configured it). Otherwise, create one.
        provider = otel_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.add_span_processor(processor)
        else:
            # Proxy wrapper — set up a new TracerProvider with our processor
            from google.adk.telemetry.setup import OTelHooks, maybe_set_otel_providers

            maybe_set_otel_providers([OTelHooks(span_processors=[processor])])

        atexit.register(_auto_export)
        logger.debug("Registered MotusSpanProcessor with Google ADK OTEL")
    except Exception as e:
        logger.debug(f"Could not register MotusSpanProcessor: {e}")


def get_tracer():
    """Get the TraceManager instance (call register_tracing() first)."""
    return _tracer


def _auto_export():
    """Auto-export traces on process exit (registered via atexit)."""
    if _tracer and _tracer.task_meta and _tracer.config.export_enabled:
        _tracer.export_trace()


class Agent(_ADKAgent):
    """Google ADK Agent with a built-in ``run_turn`` method for motus serve."""

    async def run_turn(
        self, message: ChatMessage, state: list[ChatMessage]
    ) -> tuple[ChatMessage, list[ChatMessage]]:
        """Run a single conversational turn through the ADK agent.

        Satisfies the serve agent contract:
        ``(ChatMessage, list[ChatMessage]) -> (ChatMessage, list[ChatMessage])``
        """
        from google.adk.events import Event

        from motus.models import ChatMessage as _CM

        register_tracing()

        runner = InMemoryRunner(self, app_name=_APP_NAME)
        session = await runner.session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=_SESSION_ID,
        )

        # Replay prior conversation history into the session so the model
        # sees the full context of prior turns.
        for msg in state:
            author = "user" if msg.role == "user" else self.name
            role = "user" if msg.role == "user" else "model"
            await runner.session_service.append_event(
                session,
                Event(
                    author=author,
                    content=genai_types.Content(
                        role=role,
                        parts=[genai_types.Part(text=msg.content or "")],
                    ),
                ),
            )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=message.content or "")],
        )

        response_parts: list[str] = []
        async for event in runner.run_async(
            user_id=_USER_ID,
            session_id=_SESSION_ID,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)

        await runner.close()

        response_text = "".join(response_parts) or "(no response)"
        response = _CM.assistant_message(content=response_text)

        if _tracer is not None:
            _tracer.close()

        return response, state + [message, response]
