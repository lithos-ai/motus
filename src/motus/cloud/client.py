"""Sync Client for motus.cloud."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Mapping

import httpx

from motus.serve.schemas import SessionResponse, SessionStatus

from ._models import ChatResult, Interrupt
from ._transport import (
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_READ_RETRY_BUDGET,
    DEFAULT_SERVER_WAIT_SLICE,
    build_headers,
    parse_session,
    resolve_api_key,
    sync_request,
    sync_resume_and_poll,
    sync_send_and_poll,
    validate_base_url,
)
from .errors import AgentError, MotusClientError

if TYPE_CHECKING:
    from motus.models import ChatMessage

    from .session import Session

logger = logging.getLogger("motus.cloud")


class Client:
    """Sync client for invoking Motus agents over the session/message REST protocol."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        turn_timeout: float | None = None,
        http_timeout: httpx.Timeout | None = None,
        server_wait_slice: float = DEFAULT_SERVER_WAIT_SLICE,
        read_retry_budget: int = DEFAULT_READ_RETRY_BUDGET,
        extra_headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if transport is not None and http_client is not None:
            raise ValueError("pass either transport= or http_client=, not both")
        self._base_url = validate_base_url(base_url)
        self._api_key = resolve_api_key(api_key)
        self._turn_timeout = turn_timeout
        self._http_timeout = http_timeout or DEFAULT_HTTP_TIMEOUT
        self._server_wait_slice = server_wait_slice
        self._read_retry_budget = read_retry_budget
        self._extra_headers: dict[str, str] = dict(extra_headers or {})
        if http_client is not None:
            self._http = http_client
            self._owns_http = False
        else:
            self._http = httpx.Client(timeout=self._http_timeout, transport=transport)
            self._owns_http = True

    # ------------------------- lifecycle -------------------------

    def close(self) -> None:
        if self._owns_http and self._http is not None:
            self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------- internals -------------------------

    def _guard_event_loop(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "motus.cloud.Client cannot be used from inside a running event loop; "
            "use motus.cloud.AsyncClient instead"
        )

    def _headers(self, per_call: Mapping[str, str] | None = None) -> dict[str, str]:
        return build_headers(self._api_key, self._extra_headers, per_call)

    # ------------------------- low-level -------------------------

    def health(self, *, extra_headers: Mapping[str, str] | None = None) -> dict:
        self._guard_event_loop()
        r = sync_request(
            self._http,
            "GET",
            f"{self._base_url}/health",
            headers=self._headers(extra_headers),
        )
        return r.json()

    def create_session(
        self,
        *,
        session_id: str | None = None,
        initial_state: list["ChatMessage"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> SessionResponse:
        self._guard_event_loop()
        body: dict[str, Any] | None = None
        if initial_state:
            body = {
                "state": [
                    m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
                    for m in initial_state
                ]
            }
        if session_id is None:
            r = sync_request(
                self._http,
                "POST",
                f"{self._base_url}/sessions",
                headers=self._headers(extra_headers),
                json=body,
            )
        else:
            self._check_uuid(session_id)
            r = sync_request(
                self._http,
                "PUT",
                f"{self._base_url}/sessions/{session_id}",
                headers=self._headers(extra_headers),
                json=body,
            )
        return parse_session(r.json())

    def get_session(
        self,
        session_id: str,
        *,
        wait: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> SessionResponse:
        self._guard_event_loop()
        params: dict[str, Any] = {}
        if wait:
            params["wait"] = "true"
        if timeout is not None:
            params["timeout"] = str(timeout)
        r = sync_request(
            self._http,
            "GET",
            f"{self._base_url}/sessions/{session_id}",
            headers=self._headers(extra_headers),
            params=params or None,
        )
        return parse_session(r.json())

    def list_sessions(
        self, *, extra_headers: Mapping[str, str] | None = None
    ) -> list[dict]:
        self._guard_event_loop()
        r = sync_request(
            self._http,
            "GET",
            f"{self._base_url}/sessions",
            headers=self._headers(extra_headers),
        )
        return r.json()

    def delete_session(
        self,
        session_id: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._guard_event_loop()
        try:
            sync_request(
                self._http,
                "DELETE",
                f"{self._base_url}/sessions/{session_id}",
                headers=self._headers(extra_headers),
            )
        except MotusClientError as e:
            # Idempotent: 404 is silent (already swept/deleted elsewhere).
            from .errors import SessionNotFound

            if isinstance(e, SessionNotFound):
                return
            raise

    def get_messages(
        self,
        session_id: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> list[dict]:
        self._guard_event_loop()
        r = sync_request(
            self._http,
            "GET",
            f"{self._base_url}/sessions/{session_id}/messages",
            headers=self._headers(extra_headers),
        )
        return r.json()

    def send_message(
        self,
        session_id: str,
        content: str | None = None,
        *,
        role: str = "user",
        user_params: Mapping[str, Any] | None = None,
        webhook: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        **message_fields: Any,
    ) -> dict:
        """POST /sessions/{id}/messages. Returns the 202 MessageResponse body verbatim.

        Extra ``**message_fields`` are forwarded into the request body to preserve
        full MessageRequest fidelity (e.g., tool_calls, name, tool_call_id, base64_image).
        """
        self._guard_event_loop()
        body: dict[str, Any] = {"role": role}
        if content is not None:
            body["content"] = content
        if user_params:
            body["user_params"] = dict(user_params)
        if webhook:
            body["webhook"] = dict(webhook)
        body.update(message_fields)
        r = sync_request(
            self._http,
            "POST",
            f"{self._base_url}/sessions/{session_id}/messages",
            headers=self._headers(extra_headers),
            json=body,
        )
        return r.json()

    # ------------------------- high-level -------------------------

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
        """Create an ephemeral session, send, poll, return ChatResult.

        Cleanup: DELETE only on clean idle terminal; interrupted/error/timeout
        leave the server session alive for resume/inspection.
        """
        self._guard_event_loop()
        session_id = self.create_session().session_id
        clean_idle = False
        try:
            body = self._build_message_body(
                content, role, user_params, webhook, message_fields
            )
            snapshot = sync_send_and_poll(
                self._http,
                self._base_url,
                session_id,
                body,
                turn_timeout=turn_timeout
                if turn_timeout is not None
                else self._turn_timeout,
                server_wait_slice=self._server_wait_slice,
                read_retry_budget=self._read_retry_budget,
                headers=self._headers(extra_headers),
            )
            if snapshot.status == SessionStatus.error:
                raise AgentError(snapshot.error or "agent error", session_id=session_id)
            interrupts = _interrupts_of(snapshot)
            clean_idle = snapshot.status == SessionStatus.idle and not interrupts
            return self._result(snapshot, session_id, interrupts)
        finally:
            if clean_idle:
                try:
                    self.delete_session(session_id)
                except MotusClientError as secondary:
                    logger.debug("ephemeral cleanup DELETE failed: %r", secondary)

    def session(
        self,
        *,
        session_id: str | None = None,
        keep: bool = False,
        initial_state: list["ChatMessage"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> "Session":
        """Create or attach a pinned Session.

        ``session_id=None`` creates a new server-side session (owned).
        Passing ``session_id`` attaches to an existing one (not owned — DELETE never issued).
        ``keep=True`` suppresses DELETE on owned sessions at close.
        """
        self._guard_event_loop()
        from .session import Session

        if session_id is not None and initial_state:
            raise ValueError(
                "initial_state cannot be passed with an existing session_id"
            )
        if session_id is None:
            sid = self.create_session(
                initial_state=initial_state, extra_headers=extra_headers
            ).session_id
            return Session(self, sid, owned=True, keep=keep)
        return Session(self, session_id, owned=False, keep=keep)

    def resume(
        self,
        session_id: str,
        interrupt_id: str,
        value: Any,
        *,
        turn_timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ChatResult:
        self._guard_event_loop()
        snapshot = sync_resume_and_poll(
            self._http,
            self._base_url,
            session_id,
            interrupt_id,
            value,
            turn_timeout=turn_timeout
            if turn_timeout is not None
            else self._turn_timeout,
            server_wait_slice=self._server_wait_slice,
            read_retry_budget=self._read_retry_budget,
            headers=self._headers(extra_headers),
        )
        if snapshot.status == SessionStatus.error:
            raise AgentError(snapshot.error or "agent error", session_id=session_id)
        return self._result(snapshot, session_id, _interrupts_of(snapshot))

    # ------------------------- helpers -------------------------

    @staticmethod
    def _check_uuid(session_id: str) -> None:
        try:
            uuid.UUID(session_id)
        except (ValueError, AttributeError, TypeError) as e:
            raise ValueError(
                f"session_id must be a valid UUID; got: {session_id!r}"
            ) from e

    @staticmethod
    def _build_message_body(
        content: str,
        role: str,
        user_params: Mapping[str, Any] | None,
        webhook: Mapping[str, Any] | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"role": role, "content": content}
        if user_params:
            body["user_params"] = dict(user_params)
        if webhook:
            body["webhook"] = dict(webhook)
        body.update(extra)
        return body

    def _result(
        self,
        snapshot: SessionResponse,
        session_id: str,
        interrupts: list[Interrupt],
    ) -> ChatResult:
        result = ChatResult(
            message=snapshot.response,
            interrupts=interrupts,
            session_id=session_id,
            status=snapshot.status,
            snapshot=snapshot,
        )
        result._client = self
        return result


def _interrupts_of(snapshot: SessionResponse) -> list[Interrupt]:
    return [Interrupt.from_info(i) for i in (snapshot.interrupts or [])]


__all__ = ["Client"]
