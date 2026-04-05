"""Pydantic models for the serve REST API."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from motus.models import ChatMessage


class SessionStatus(str, Enum):
    idle = "idle"
    running = "running"
    error = "error"


class CreateSessionRequest(BaseModel):
    state: list[ChatMessage] = []


class SessionResponse(BaseModel):
    session_id: str
    status: SessionStatus
    response: ChatMessage | None = None
    error: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    total_messages: int
    status: SessionStatus


class WebhookSpec(BaseModel):
    url: str
    token: str | None = None
    include_messages: bool = False


class WebhookPayload(BaseModel):
    session_id: str
    status: SessionStatus
    response: ChatMessage | None = None
    error: str | None = None
    messages: list[ChatMessage] | None = None


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
