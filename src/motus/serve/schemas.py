"""Pydantic models for the serve REST API."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

from motus.models import ChatMessage


class SessionStatus(str, Enum):
    idle = "idle"
    running = "running"
    interrupted = "interrupted"
    error = "error"


class InterruptInfo(BaseModel):
    """Information about a pending interrupt, returned to clients via GET /sessions/{id}."""

    interrupt_id: str
    type: str  # "tool_approval" | "user_input" (extensible)
    payload: dict


class ResumeRequest(BaseModel):
    """POST /sessions/{id}/resume body."""

    interrupt_id: str
    value: Any  # shape depends on interrupt type


class CreateSessionRequest(BaseModel):
    state: list[ChatMessage] = []


class SessionResponse(BaseModel):
    session_id: str
    status: SessionStatus
    response: ChatMessage | None = None
    error: str | None = None
    interrupts: list[InterruptInfo] | None = None


class SessionSummary(BaseModel):
    session_id: str
    total_messages: int
    status: SessionStatus


class WebhookSpec(BaseModel):
    url: str
    token: str | None = None
    include_messages: bool = False


class TraceMetrics(BaseModel):
    trace_id: str | None = None
    total_duration: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    has_error: bool = False


class WebhookPayload(BaseModel):
    session_id: str
    status: SessionStatus
    response: ChatMessage | None = None
    error: str | None = None
    messages: list[ChatMessage] | None = None
    trace_metrics: TraceMetrics | None = None


class MessageRequest(ChatMessage):
    """Inherits all ChatMessage fields so callers can inject arbitrary messages."""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    webhook: WebhookSpec | None = None


class MessageResponse(BaseModel):
    session_id: str
    status: SessionStatus


class HealthResponse(BaseModel):
    status: str
    max_workers: int
    running_workers: int
    total_sessions: int


class JudgeRequest(BaseModel):
    """POST /eval/judge body — platform-triggered post-session eval.

    The developer only supplies the ``criteria``; the judge system prompt
    and user message template (input/output injection) are platform-owned.
    """

    input: str
    output: str
    model: str = "claude-haiku-4-5"
    criteria: str


class JudgeResponse(BaseModel):
    """Response from POST /eval/judge."""

    score: float
    passed: bool
    reason: str = ""


class JudgeError(BaseModel):
    """Returned when the judge fails to produce a score."""

    error: str
