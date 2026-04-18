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


class Agent(_ADKAgent):
    """Google ADK Agent with a built-in ``run_turn`` method for motus serve.

    ADK emits spans on the global TracerProvider, which motus already owns
    (``setup_tracing`` runs at ``motus`` import time). ADK's gen_ai.* spans
    flow through our SpanProcessors and are rendered via ``span_convert``'s
    gen_ai.* fallbacks — no bridge processor needed.
    """

    async def run_turn(
        self, message: ChatMessage, state: list[ChatMessage]
    ) -> tuple[ChatMessage, list[ChatMessage]]:
        """Run a single conversational turn through the ADK agent.

        Satisfies the serve agent contract:
        ``(ChatMessage, list[ChatMessage]) -> (ChatMessage, list[ChatMessage])``
        """
        from google.adk.events import Event

        from motus.models import ChatMessage as _CM

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
