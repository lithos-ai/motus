"""Pinned Session handle (sync)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

from motus.serve.schemas import SessionStatus

from ._models import ChatResult, Interrupt
from ._transport import sync_resume_and_poll, sync_send_and_poll
from .errors import AgentError, SessionClosed

if TYPE_CHECKING:
    from .client import Client

logger = logging.getLogger("motus.cloud")


class Session:
    """Pinned session. Ownership determines whether close() issues DELETE.

    - Created via ``client.session()`` with no ``session_id``: owned = True.
    - Created via ``client.session(session_id=<existing>)``: owned = False,
      close() never DELETEs regardless of ``keep``.
    - ``keep=True`` on an owned session suppresses DELETE on close (logs info).
    """

    def __init__(
        self,
        client: "Client",
        session_id: str,
        *,
        owned: bool,
        keep: bool,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._owned = owned
        self._keep = keep
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def owned(self) -> bool:
        return self._owned

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------- lifecycle -------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned and not self._keep:
            try:
                self._client.delete_session(self._session_id)
            except Exception as e:  # noqa: BLE001 — never mask caller context on close
                logger.debug("Session.close DELETE failed: %r", e)
        elif self._owned and self._keep:
            logger.info("Session kept alive: session_id=%s", self._session_id)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------- turns -------------------------

    def chat(
        self,
        content: str,
        *,
        turn_timeout: float | None = None,
        role: str = "user",
        user_params: Mapping[str, Any] | None = None,
        webhook: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        **message_fields: Any,
    ) -> ChatResult:
        if self._closed:
            raise SessionClosed(f"session {self._session_id} is closed")
        self._client._guard_event_loop()
        body: dict[str, Any] = {"role": role, "content": content}
        if user_params:
            body["user_params"] = dict(user_params)
        if webhook:
            body["webhook"] = dict(webhook)
        body.update(message_fields)
        snapshot = sync_send_and_poll(
            self._client._http,
            self._client._base_url,
            self._session_id,
            body,
            turn_timeout=turn_timeout
            if turn_timeout is not None
            else self._client._turn_timeout,
            server_wait_slice=self._client._server_wait_slice,
            read_retry_budget=self._client._read_retry_budget,
            headers=self._client._headers(extra_headers),
        )
        if snapshot.status == SessionStatus.error:
            raise AgentError(
                snapshot.error or "agent error", session_id=self._session_id
            )
        interrupts = [Interrupt.from_info(i) for i in (snapshot.interrupts or [])]
        result = ChatResult(
            message=snapshot.response,
            interrupts=interrupts,
            session_id=self._session_id,
            status=snapshot.status,
            snapshot=snapshot,
        )
        result._client = self._client
        return result

    def resume(
        self,
        interrupt_id: str,
        value: Any,
        *,
        turn_timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ChatResult:
        if self._closed:
            raise SessionClosed(f"session {self._session_id} is closed")
        self._client._guard_event_loop()
        snapshot = sync_resume_and_poll(
            self._client._http,
            self._client._base_url,
            self._session_id,
            interrupt_id,
            value,
            turn_timeout=turn_timeout
            if turn_timeout is not None
            else self._client._turn_timeout,
            server_wait_slice=self._client._server_wait_slice,
            read_retry_budget=self._client._read_retry_budget,
            headers=self._client._headers(extra_headers),
        )
        if snapshot.status == SessionStatus.error:
            raise AgentError(
                snapshot.error or "agent error", session_id=self._session_id
            )
        interrupts = [Interrupt.from_info(i) for i in (snapshot.interrupts or [])]
        result = ChatResult(
            message=snapshot.response,
            interrupts=interrupts,
            session_id=self._session_id,
            status=snapshot.status,
            snapshot=snapshot,
        )
        result._client = self._client
        return result


__all__ = ["Session"]
