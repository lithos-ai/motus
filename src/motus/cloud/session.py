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
    - ``extra_headers`` is applied to every request issued through this session.
    """

    def __init__(
        self,
        client: "Client",
        session_id: str,
        *,
        owned: bool,
        keep: bool,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._owned = owned
        self._keep = keep
        self._session_headers: dict[str, str] = dict(extra_headers or {})
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

    def close(self) -> None:
        """Close the session.

        For owned sessions without ``keep=True``, issues a DELETE. A 404 response
        is treated as success (the session is already gone). Other errors — 5xx,
        connection failures, auth failures — propagate as ``MotusClientError``
        and leave ``closed`` False so the caller can retry.
        """
        if self._closed:
            return
        if self._owned and not self._keep:
            # delete_session already silences 404; everything else propagates.
            self._client.delete_session(
                self._session_id,
                extra_headers=self._session_headers or None,
            )
        elif self._owned and self._keep:
            logger.info("Session kept alive: session_id=%s", self._session_id)
        self._closed = True

    def keep_alive(self) -> None:
        """Switch the session into keep mode so ``close()`` does not DELETE."""
        self._keep = True

    def __enter__(self) -> "Session":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        # If the body already raised, do best-effort cleanup and never mask the
        # original exception with a secondary cleanup failure. Explicit
        # ``session.close()`` still propagates — only the context-manager exit
        # path swallows cleanup errors when there is already an exception in
        # flight.
        if exc_type is not None:
            try:
                self.close()
            except Exception as secondary:  # noqa: BLE001
                logger.debug(
                    "Session cleanup during exception propagation failed: %r",
                    secondary,
                )
        else:
            self.close()

    def _merged_headers(
        self, per_call: Mapping[str, str] | None
    ) -> dict[str, str] | None:
        if not self._session_headers and not per_call:
            return None
        merged: dict[str, str] = dict(self._session_headers)
        if per_call:
            merged.update(per_call)
        return merged

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
            headers=self._client._headers(self._merged_headers(extra_headers)),
        )
        if snapshot.status == SessionStatus.error:
            raise AgentError(
                snapshot.error or "agent error", session_id=self._session_id
            )
        interrupts = [Interrupt.from_info(i) for i in (snapshot.interrupts or [])]
        return self._make_result(snapshot, interrupts, per_call_headers=extra_headers)

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
            headers=self._client._headers(self._merged_headers(extra_headers)),
        )
        if snapshot.status == SessionStatus.error:
            raise AgentError(
                snapshot.error or "agent error", session_id=self._session_id
            )
        interrupts = [Interrupt.from_info(i) for i in (snapshot.interrupts or [])]
        return self._make_result(snapshot, interrupts, per_call_headers=extra_headers)

    def _make_result(
        self,
        snapshot,
        interrupts: list[Interrupt],
        *,
        per_call_headers: Mapping[str, str] | None,
    ) -> ChatResult:
        # Route resume through Session.resume(extra_headers=per_call_headers) so
        # both session-scoped and per-call headers survive to the /resume POST.
        def _resumer(value: Any) -> ChatResult:
            return self.resume(interrupts[0].id, value, extra_headers=per_call_headers)

        return ChatResult(
            message=snapshot.response,
            interrupts=interrupts,
            session_id=self._session_id,
            status=snapshot.status,
            snapshot=snapshot,
            _resumer=_resumer if interrupts else None,
        )


__all__ = ["Session"]
