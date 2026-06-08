# =============================================================================
# Scheduled Agent Example
# =============================================================================
#
# Demonstrates the Motus Scheduler — on the first turn, the agent registers
# a recurring notification that delivers a message into the session every
# minute.  Subsequent turns (including scheduled ones) are handled normally.
#
# On the platform (CloudScheduler), notifications are persisted by the
# notification service and delivered via EventBridge Scheduler.
#
# Locally (LocalScheduler), a background task fires the handler in-process.
#
# -----------------------------------------------------------------------------
# Cloud deployment
# -----------------------------------------------------------------------------
#
#   motus deploy <project-id> examples.serving.scheduled_agent:agent
#
#   # Create a session and send the first message to start the schedule:
#   curl -X POST https://<project>.agent.<domain>/sessions \
#     -H "Authorization: Bearer <api-key>"
#   curl -X POST https://<project>.agent.<domain>/sessions/<id>/messages \
#     -H "Authorization: Bearer <api-key>" \
#     -d '{"content": "Start the schedule"}'
#
# =============================================================================

import os

from motus.models import ChatMessage
from motus.utils import CloudScheduler

_scheduler = None
_scheduled_sessions: set[str] = set()


def _get_scheduler() -> CloudScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = CloudScheduler()
    return _scheduler


async def agent(message: ChatMessage, state: list[ChatMessage]):
    session_id = os.environ.get("MOTUS_SESSION_ID", "")

    # On the first user message, register a recurring notification.
    is_first_turn = not any(m.role == "assistant" for m in state)

    if is_first_turn and session_id not in _scheduled_sessions:
        scheduler = _get_scheduler()
        notification_id = await scheduler.schedule(
            "heartbeat",
            schedule_type="rate",
            expression="rate(1 minute)",
            message="Scheduled heartbeat — the agent is alive.",
        )
        _scheduled_sessions.add(session_id)
        response = ChatMessage.assistant_message(
            content=f"Schedule registered (notification_id={notification_id}). "
            f"You'll receive a heartbeat every minute.",
        )
    else:
        response = ChatMessage.assistant_message(
            content=f"Received: {message.content}",
        )

    return response, state + [message, response]
