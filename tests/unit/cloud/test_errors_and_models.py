"""Error hierarchy + response model contracts."""

import pytest

from motus.cloud import (
    AgentError,
    AmbiguousInterrupt,
    AuthError,
    BackendUnavailable,
    ChatResult,
    Interrupt,
    InterruptNotFound,
    MotusClientError,
    ProtocolError,
    ServerBusy,
    SessionClosed,
    SessionConflict,
    SessionNotFound,
    SessionTimeout,
    SessionUnsupported,
)
from motus.serve.schemas import InterruptInfo, SessionStatus

ALL_ERRORS = [
    AuthError,
    SessionNotFound,
    InterruptNotFound,
    SessionConflict,
    ServerBusy,
    BackendUnavailable,
    AgentError,
    ProtocolError,
    SessionClosed,
    AmbiguousInterrupt,
    SessionUnsupported,
]


@pytest.mark.parametrize("cls", ALL_ERRORS)
def test_every_error_subclasses_motus_client_error(cls):
    assert issubclass(cls, MotusClientError)


def test_session_timeout_carries_reconnection_fields():
    exc = SessionTimeout("deadline", session_id="abc", elapsed=2.5)
    assert exc.session_id == "abc"
    assert exc.elapsed == 2.5
    assert exc.last_snapshot is None


def test_agent_error_carries_session_id():
    exc = AgentError("boom", session_id="xyz")
    assert exc.session_id == "xyz"


def test_interrupt_from_info_roundtrips():
    info = InterruptInfo(
        interrupt_id="i1",
        type="tool_approval",
        payload={"tool_name": "search"},
    )
    i = Interrupt.from_info(info)
    assert i.id == "i1"
    assert i.type == "tool_approval"
    assert i.payload == {"tool_name": "search"}


def test_chat_result_resume_without_interrupts_raises():
    r = ChatResult(
        message=None, interrupts=[], session_id="s", status=SessionStatus.idle
    )
    with pytest.raises(AmbiguousInterrupt):
        r.resume("v")


def test_chat_result_resume_with_two_interrupts_raises():
    r = ChatResult(
        message=None,
        interrupts=[Interrupt("a", "t", {}), Interrupt("b", "t", {})],
        session_id="s",
        status=SessionStatus.interrupted,
    )
    with pytest.raises(AmbiguousInterrupt):
        r.resume("v")


def test_chat_result_resume_without_client_raises():
    r = ChatResult(
        message=None,
        interrupts=[Interrupt("a", "t", {})],
        session_id="s",
        status=SessionStatus.interrupted,
    )
    with pytest.raises(AmbiguousInterrupt):
        r.resume("v")


def test_chat_result_resume_delegates_to_resumer():
    calls = []

    def resumer(value):
        calls.append(value)
        return "delegated"

    r = ChatResult(
        message=None,
        interrupts=[Interrupt("iid-1", "tool_approval", {})],
        session_id="sid-1",
        status=SessionStatus.interrupted,
        _resumer=resumer,
    )
    assert r.resume("blue") == "delegated"
    assert calls == ["blue"]
