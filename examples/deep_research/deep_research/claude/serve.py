"""
Serve entry point for Deep Research (Claude Agent SDK).

Usage:
    motus serve start deep_research.claude.serve:agent
"""

import logging

from motus.models import ChatMessage

from .agent import deep_research

logger = logging.getLogger(__name__)


async def agent(
    message: ChatMessage, state: list[ChatMessage]
) -> tuple[ChatMessage, list[ChatMessage]]:
    question = message.content or ""
    if not question.strip():
        resp = ChatMessage.assistant_message(
            content="Please provide a research question."
        )
        return resp, state + [message, resp]
    try:
        report = await deep_research(question)
    except Exception:
        logger.exception("Research failed")
        report = "Research encountered an internal error. Please try again."
    resp = ChatMessage.assistant_message(content=report)
    return resp, state + [message, resp]
