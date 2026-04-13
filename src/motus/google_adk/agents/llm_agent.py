"""Motus-enhanced Google ADK LlmAgent.

Subclasses the Google ADK ``Agent`` and adds a ``run_turn`` method that
satisfies the motus serve agent contract::

    async def run_turn(message: ChatMessage, state: list[ChatMessage])
        -> tuple[ChatMessage, list[ChatMessage]]

This allows a Google ADK agent to be served directly via
``motus serve start``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace

try:
    from google.adk.agents.llm_agent import Agent as _ADKAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
except ImportError as exc:
    raise ImportError(
        "motus.google_adk requires the Google ADK. "
        "Install it with: uv pip install lithosai-motus[google-adk]"
    ) from exc

if TYPE_CHECKING:
    from motus.models import ChatMessage

logger = logging.getLogger("AgentTracer")

_APP_NAME = "motus"
_USER_ID = "motus_user"
_SESSION_ID = "motus_session"

_otel_registered = False


def _get_tracer() -> trace.Tracer:
    """Get the OTel tracer from motus tracing setup."""
    from motus.runtime.tracing.agent_tracer import get_tracer

    return get_tracer()


def _ensure_tracing() -> None:
    """Register MotusSpanProcessor with the OTel TracerProvider (once per process).

    ADK emits OTel spans on the global TracerProvider. Our MotusSpanProcessor
    re-emits them on the motus tracer with motus.* attributes so they flow
    through our configured SpanProcessors.
    """
    global _otel_registered
    if _otel_registered:
        return

    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider

        from motus.google_adk._motus_tracing import MotusSpanProcessor

        processor = MotusSpanProcessor()

        # Add our processor to the existing TracerProvider if one is already set
        # (ADK or another library may have configured it). Otherwise, create one.
        provider = otel_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.add_span_processor(processor)
        else:
            # Proxy wrapper -- set up a new TracerProvider with our processor
            from google.adk.telemetry.setup import OTelHooks, maybe_set_otel_providers

            maybe_set_otel_providers([OTelHooks(span_processors=[processor])])

        _otel_registered = True
        logger.debug("Registered MotusSpanProcessor with Google ADK OTel")
    except Exception as e:
        logger.debug(f"Could not register MotusSpanProcessor: {e}")


def get_tracer() -> trace.Tracer:
    """Public accessor for the OTel tracer."""
    return _get_tracer()


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

        _ensure_tracing()

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

        return response, state + [message, response]
