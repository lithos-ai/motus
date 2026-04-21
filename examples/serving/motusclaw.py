# =============================================================================
# MotusClaw — always-on scaffold agent
# =============================================================================
#
# Minimal Motus-native take on an OpenClaw-style always-on agent:
# a ReActAgent with web search + shell + file tools, plus scheduler tools
# the agent calls when it wants to register, cancel, or list periodic
# or one-shot notifications. Scheduled fires arrive back into the same
# session as user-role messages via the platform notification service.
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
#   # Create a session and ask the agent to schedule something:
#   curl -X POST https://<project>.agent.<domain>/sessions \
#     -H "Authorization: Bearer $LITHOSAI_API_KEY"
#   curl -X POST https://<project>.agent.<domain>/sessions/<id>/messages \
#     -H "Authorization: Bearer $LITHOSAI_API_KEY" \
#     -H 'Content-Type: application/json' \
#     -d '{"content":"Remind me to check on my tasks every 3 minutes."}'
#
# =============================================================================

import logging
import os

from motus.agent import ReActAgent
from motus.models import ChatMessage, OpenRouterChatClient
from motus.tools import tool
from motus.tools.builtins import builtin_tools
from motus.tools.providers.brave import WebSearchTool
from motus.utils import CloudScheduler, LocalScheduler, Scheduler

logger = logging.getLogger("motusclaw")


def _make_scheduler() -> Scheduler:
    if os.environ.get("MOTUSCLAW_SCHEDULER", "cloud").lower() == "local":
        return LocalScheduler()
    return CloudScheduler()


@tool
async def schedule_notification(
    name: str,
    expression: str,
    message: str,
    schedule_type: str = "rate",
) -> str:
    """Register a notification that will be delivered into this session later.

    Use this when the user asks you to remind them, follow up, or check in
    periodically. The scheduled message arrives as a user-role turn, so you
    should respond to it as if the user just sent it.

    Args:
        name: Short identifier for this schedule (e.g. "daily-digest").
            Must be unique per session.
        expression: EventBridge schedule expression. For recurring:
            "rate(N minutes|hours|days)" (e.g. "rate(5 minutes)").
            For a one-off at a specific time: use schedule_type="at" and pass
            an ISO-8601 timestamp like "at(2026-04-21T14:30:00)".
            For cron: "cron(0 9 * * ? *)" with schedule_type="cron".
        message: Content the user-role message will carry when the schedule
            fires. Write it so your future self understands what to do.
        schedule_type: "rate" (default), "cron", or "at".

    Returns:
        The notification_id. Keep it if you might want to cancel later.
    """
    notification_id = await _make_scheduler().schedule(
        name,
        schedule_type=schedule_type,
        expression=expression,
        message=message,
    )
    return notification_id


@tool
async def cancel_notification(notification_id: str) -> str:
    """Cancel a previously scheduled notification by its notification_id."""
    await _make_scheduler().remove(notification_id)
    return f"cancelled {notification_id}"


@tool
async def list_notifications() -> list[dict]:
    """List all notifications currently scheduled for this session/project."""
    scheduler = _make_scheduler()
    if isinstance(scheduler, CloudScheduler):
        return await scheduler.list()
    return []


SYSTEM_PROMPT = """\
You are MotusClaw, an always-on research and follow-up assistant.

You have three kinds of tools:
- Web search for current information.
- Shell and file tools to capture, summarize, and persist findings under the
  working directory.
- Scheduling tools (`schedule_notification`, `cancel_notification`,
  `list_notifications`) to remind yourself or the user to do something later.

When the user asks to be reminded, followed up with, or checked in on
periodically, use `schedule_notification` with an appropriate rate or cron
expression and write the `message` so that future-you knows what context to
reference when the schedule fires. Scheduled fires come back as user-role
messages — treat them as the user asking you to carry out the work now.
"""


def _build_tools() -> list:
    tools: list = [*builtin_tools(), schedule_notification, cancel_notification, list_notifications]
    brave_key = os.environ.get("BRAVE_API_KEY")
    if brave_key:
        tools.append(WebSearchTool(api_key=brave_key))
    else:
        logger.info("BRAVE_API_KEY unset; MotusClaw running without web search")
    return tools


agent = ReActAgent(
    client=OpenRouterChatClient(),
    model_name=os.environ.get("MOTUSCLAW_MODEL", "anthropic/claude-sonnet-4"),
    system_prompt=SYSTEM_PROMPT,
    tools=_build_tools(),
)
