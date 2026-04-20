# =============================================================================
# MotusClaw — always-on scaffold agent
# =============================================================================
#
# Minimal Motus-native take on an OpenClaw-style always-on agent:
# a ReActAgent with web search + shell + file tools that registers a
# recurring scheduled notification on the first user turn. Subsequent
# turns (interactive and scheduled) are handled by the ReAct loop.
#
# Multi-user isolation, inbound auth, and tracing are provided by the
# existing platform:
#   * agent_router enforces Authorization: Bearer <LITHOSAI_API_KEY>
#     per project (PR #120, ProjectApiKeysTable)
#   * model + search traffic egresses via the platform proxies
#     (OPENROUTER_BASE_URL / BRAVE_BASE_URL wired by ecr_trigger)
#   * tracing is the custom Motus subsystem exporting to the LithosAI
#     cloud via LITHOSAI_API_URL + LITHOSAI_API_KEY when MOTUS_TRACING=1
#
# -----------------------------------------------------------------------------
# Cloud deployment
# -----------------------------------------------------------------------------
#
#   motus deploy <project-id> examples.serving.motusclaw:agent
#
#   # Create a session and kick off the schedule:
#   curl -X POST https://<project>.agent.<domain>/sessions \
#     -H "Authorization: Bearer $LITHOSAI_API_KEY"
#   curl -X POST https://<project>.agent.<domain>/sessions/<id>/messages \
#     -H "Authorization: Bearer $LITHOSAI_API_KEY" \
#     -H 'Content-Type: application/json' \
#     -d '{"content":"Find recent LithosAI news and schedule a check-in."}'
#
# -----------------------------------------------------------------------------
# Local testing (LocalScheduler, no cloud dependency)
# -----------------------------------------------------------------------------
#
#   export OPENROUTER_API_KEY=...
#   export BRAVE_API_KEY=...          # optional; search tool only
#   export MOTUSCLAW_SCHEDULER=local  # use LocalScheduler instead of CloudScheduler
#   uv run motus serve start examples.serving.motusclaw:agent --port 8000
#
# =============================================================================

import logging
import os

from motus.agent import ReActAgent
from motus.models import ChatMessage, OpenRouterChatClient
from motus.tools.builtins import builtin_tools
from motus.tools.providers.brave import WebSearchTool
from motus.utils import CloudScheduler, LocalScheduler, Scheduler

logger = logging.getLogger("motusclaw")

SYSTEM_PROMPT = """\
You are MotusClaw, an always-on research assistant.
Use the web search tool for current information, and the shell and file
tools to capture, summarize, and persist findings under the working
directory. When a scheduled check-in arrives, briefly report progress
on any open threads from the conversation.
"""

_HEARTBEAT_EXPRESSION = os.environ.get(
    "MOTUSCLAW_HEARTBEAT_EXPRESSION", "rate(2 minutes)"
)
_HEARTBEAT_MESSAGE = os.environ.get(
    "MOTUSCLAW_HEARTBEAT_MESSAGE",
    "Scheduled check-in — any updates on your open tasks?",
)


def _build_tools() -> list:
    tools = [*builtin_tools()]
    brave_key = os.environ.get("BRAVE_API_KEY")
    if brave_key:
        tools.append(WebSearchTool(api_key=brave_key))
    else:
        logger.info("BRAVE_API_KEY unset; MotusClaw running without web search")
    return tools


_react = ReActAgent(
    client=OpenRouterChatClient(),
    model_name=os.environ.get("MOTUSCLAW_MODEL", "anthropic/claude-sonnet-4"),
    system_prompt=SYSTEM_PROMPT,
    tools=_build_tools(),
)

_scheduler: Scheduler | None = None


def _get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        if os.environ.get("MOTUSCLAW_SCHEDULER", "cloud").lower() == "local":
            _scheduler = LocalScheduler()
        else:
            _scheduler = CloudScheduler()
    return _scheduler


async def agent(message: ChatMessage, state: list[ChatMessage]):
    is_first_turn = not any(m.role == "assistant" for m in state)

    if is_first_turn:
        try:
            notification_id = await _get_scheduler().schedule(
                "motusclaw-heartbeat",
                schedule_type="rate",
                expression=_HEARTBEAT_EXPRESSION,
                message=_HEARTBEAT_MESSAGE,
            )
            logger.info("Registered heartbeat notification_id=%s", notification_id)
        except Exception as e:
            # Scheduler failures must not block the first-turn response.
            logger.warning("Failed to register heartbeat: %s", e)

    return await _react.run_turn(message, state)
