"""Async client + session for motus.cloud."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Mapping

import httpx

from motus.serve.schemas import SessionResponse, SessionStatus

from ._models import ChatResult, Interrupt, SessionEvent
from ._transport import (
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_READ_RETRY_BUDGET,
    DEFAULT_SERVER_WAIT_SLICE,
    async_poll_until_terminal,
    async_post_message,
    async_request,
    async_resume_and_poll,
    async_send_and_poll,
    build_headers,
    decode_json,
    parse_session_response,
    resolve_api_key,
    validate_base_url,
)
from .errors import AgentError, MotusClientError, SessionClosed, SessionNotFound

if TYPE_CHECKING:
    from motus.models import ChatMessage

logger = logging.getLogger("motus.cloud")


class AsyncClient:
    """Async counterpart of Client; shares transport and semantics."""

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
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
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
            self._http = httpx.AsyncClient(
                timeout=self._http_timeout, transport=transport
            )
            self._owns_http = True

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------- internals -------------------------

    def _headers(self, per_call: Mapping[str, str] | None = None) -> dict[str, str]:
        return build_headers(self._api_key, self._extra_headers, per_call)

    # ------------------------- low-level -------------------------

    async def health(self, *, extra_headers: Mapping[str, str] | None = None) -> dict:
        r = await async_request(
            self._http,
            "GET",
            f"{self._base_url}/health",
            headers=self._headers(extra_headers),
        )
        return decode_json(r)

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        initial_state: list["ChatMessage"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> SessionResponse:
        body: dict[str, Any] | None = None
        if initial_state:
            body = {
                "state": [
                    m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
                    for m in initial_state
                ]
            }
        if session_id is None:
            r = await async_request(
                self._http,
                "POST",
                f"{self._base_url}/sessions",
                headers=self._headers(extra_headers),
                json=body,
            )
        else:
            try:
                uuid.UUID(session_id)
            except (ValueError, AttributeError, TypeError) as e:
                raise ValueError(
                    f"session_id must be a valid UUID; got: {session_id!r}"
                ) from e
            r = await async_request(
                self._http,
                "PUT",
                f"{self._base_url}/sessions/{session_id}",
                headers=self._headers(extra_headers),
                json=body,
            )
        return parse_session_response(r)

    async def get_session(
        self,
        session_id: str,
        *,
        wait: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> SessionResponse:
        params: dict[str, Any] = {}
        if wait:
            params["wait"] = "true"
        if timeout is not None:
            params["timeout"] = str(timeout)
        r = await async_request(
            self._http,
            "GET",
            f"{self._base_url}/sessions/{session_id}",
            headers=self._headers(extra_headers),
            params=params or None,
        )
        return parse_session_response(r)

    async def list_sessions(
        self, *, extra_headers: Mapping[str, str] | None = None
    ) -> list[dict]:
        r = await async_request(
            self._http,
            "GET",
            f"{self._base_url}/sessions",
            headers=self._headers(extra_headers),
        )
        return decode_json(r)

    async def delete_session(
        self,
        session_id: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        try:
            await async_request(
                self._http,
                "DELETE",
                f"{self._base_url}/sessions/{session_id}",
                headers=self._headers(extra_headers),
            )
        except SessionNotFound:
            return

    async def get_messages(
        self,
        session_id: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> list[dict]:
        r = await async_request(
            self._http,
            "GET",
            f"{self._base_url}/sessions/{session_id}/messages",
            headers=self._headers(extra_headers),
        )
        return decode_json(r)

    async def send_message(
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
        body: dict[str, Any] = {"role": role}
        if content is not None:
            body["content"] = content
        if user_params:
            body["user_params"] = dict(user_params)
        if webhook:
            body["webhook"] = dict(webhook)
        body.update(message_fields)
        r = await async_request(
            self._http,
            "POST",
            f"{self._base_url}/sessions/{session_id}/messages",
            headers=self._headers(extra_headers),
            json=body,
        )
        return decode_json(r)

    # ------------------------- high-level -------------------------

    async def chat(
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
        session_id = (await self.create_session(extra_headers=extra_headers)).session_id
        clean_idle = False
        try:
            body: dict[str, Any] = {"role": role, "content": content}
            if user_params:
                body["user_params"] = dict(user_params)
            if webhook:
                body["webhook"] = dict(webhook)
            body.update(message_fields)
            snapshot = await async_send_and_poll(
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
            interrupts = [Interrupt.from_info(i) for i in (snapshot.interrupts or [])]
            clean_idle = snapshot.status == SessionStatus.idle and not interrupts
            return self._result(snapshot, session_id, interrupts)
        finally:
            if clean_idle:
                try:
                    await self.delete_session(session_id, extra_headers=extra_headers)
                except MotusClientError as secondary:
                    logger.debug("ephemeral cleanup DELETE failed: %r", secondary)

    async def chat_events(
        self,
        content: str,
        *,
        turn_timeout: float | None = None,
        role: str = "user",
        user_params: Mapping[str, Any] | None = None,
        webhook: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        **message_fields: Any,
    ):
        """Async coarse session-status iterator (counterpart of Client.chat_events).

        Yields ``SessionEvent(running)`` then the terminal event. Cleanup rules
        match ``AsyncClient.chat``. This is not token streaming.
        """
        session_id = (await self.create_session(extra_headers=extra_headers)).session_id
        clean_idle = False
        try:
            body: dict[str, Any] = {"role": role, "content": content}
            if user_params:
                body["user_params"] = dict(user_params)
            if webhook:
                body["webhook"] = dict(webhook)
            body.update(message_fields)
            headers = self._headers(extra_headers)
            await async_post_message(
                self._http, self._base_url, session_id, body, headers=headers
            )
            yield SessionEvent(type="running", session_id=session_id, snapshot=None)
            snapshot = await async_poll_until_terminal(
                self._http,
                self._base_url,
                session_id,
                turn_timeout=turn_timeout
                if turn_timeout is not None
                else self._turn_timeout,
                server_wait_slice=self._server_wait_slice,
                read_retry_budget=self._read_retry_budget,
                headers=headers,
            )
            clean_idle = (
                snapshot.status == SessionStatus.idle and not snapshot.interrupts
            )
            yield SessionEvent(
                type=snapshot.status.value,
                session_id=session_id,
                snapshot=snapshot,
            )
        finally:
            if clean_idle:
                try:
                    await self.delete_session(session_id, extra_headers=extra_headers)
                except MotusClientError as secondary:
                    logger.debug("chat_events cleanup DELETE failed: %r", secondary)

    def session(
        self,
        *,
        session_id: str | None = None,
        keep: bool = False,
        initial_state: list["ChatMessage"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> "_AsyncSessionCtx":
        """Async counterpart of Client.session().

        See Client.session() for the custom-ID ownership rules. Creation happens
        on ``__aenter__`` so the caller can ``async with client.session() as s:``.
        """
        if session_id is not None and initial_state:
            raise ValueError(
                "initial_state cannot be passed with an existing session_id"
            )
        return _AsyncSessionCtx(self, session_id, keep, initial_state, extra_headers)

    async def resume(
        self,
        session_id: str,
        interrupt_id: str,
        value: Any,
        *,
        turn_timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ChatResult:
        snapshot = await async_resume_and_poll(
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
        return self._result(
            snapshot,
            session_id,
            [Interrupt.from_info(i) for i in (snapshot.interrupts or [])],
        )

    # ------------------------- helpers -------------------------

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
        # Use the sync-facing convenience only when resume() is sync.
        # For async, callers should prefer ``await client.resume(...)``.
        result._client = _AsyncResumeAdapter(self)
        return result


class _AsyncResumeAdapter:
    """Adapter so ChatResult.resume(value) returns an awaitable for AsyncClient."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    def resume(self, session_id: str, interrupt_id: str, value: Any):
        return self._client.resume(session_id, interrupt_id, value)


class AsyncSession:
    """Pinned session for AsyncClient. Mirrors Session ownership/keep semantics.

    ``extra_headers`` is stored and applied to every lifecycle request
    (chat, resume, delete) issued through this session.
    """

    def __init__(
        self,
        client: AsyncClient,
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

    def _merged_headers(
        self, per_call: Mapping[str, str] | None
    ) -> dict[str, str] | None:
        if not self._session_headers and not per_call:
            return None
        merged: dict[str, str] = dict(self._session_headers)
        if per_call:
            merged.update(per_call)
        return merged

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned and not self._keep:
            try:
                await self._client.delete_session(
                    self._session_id,
                    extra_headers=self._session_headers or None,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("AsyncSession.aclose DELETE failed: %r", e)
        elif self._owned and self._keep:
            logger.info("Session kept alive: session_id=%s", self._session_id)

    async def chat(
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
        body: dict[str, Any] = {"role": role, "content": content}
        if user_params:
            body["user_params"] = dict(user_params)
        if webhook:
            body["webhook"] = dict(webhook)
        body.update(message_fields)
        snapshot = await async_send_and_poll(
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
        result = ChatResult(
            message=snapshot.response,
            interrupts=interrupts,
            session_id=self._session_id,
            status=snapshot.status,
            snapshot=snapshot,
        )
        result._client = _AsyncResumeAdapter(self._client)
        return result

    async def resume(
        self,
        interrupt_id: str,
        value: Any,
        *,
        turn_timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ChatResult:
        if self._closed:
            raise SessionClosed(f"session {self._session_id} is closed")
        return await self._client.resume(
            self._session_id,
            interrupt_id,
            value,
            turn_timeout=turn_timeout,
            extra_headers=self._merged_headers(extra_headers),
        )


class _AsyncSessionCtx:
    """Async context manager returned by AsyncClient.session()."""

    def __init__(
        self,
        client: AsyncClient,
        session_id: str | None,
        keep: bool,
        initial_state: list["ChatMessage"] | None,
        extra_headers: Mapping[str, str] | None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._keep = keep
        self._initial_state = initial_state
        self._extra_headers = extra_headers
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        from .errors import SessionConflict, SessionNotFound

        if self._session_id is None:
            created = await self._client.create_session(
                initial_state=self._initial_state,
                extra_headers=self._extra_headers,
            )
            self._session = AsyncSession(
                self._client,
                created.session_id,
                owned=True,
                keep=self._keep,
                extra_headers=self._extra_headers,
            )
            return self._session

        # GET-first attach (matches sync Client.session); only fall through to PUT
        # when the session doesn't exist yet.
        try:
            await self._client.get_session(
                self._session_id, extra_headers=self._extra_headers
            )
        except SessionNotFound:
            pass
        else:
            self._session = AsyncSession(
                self._client,
                self._session_id,
                owned=False,
                keep=self._keep,
                extra_headers=self._extra_headers,
            )
            return self._session

        try:
            created = await self._client.create_session(
                session_id=self._session_id, extra_headers=self._extra_headers
            )
        except SessionConflict:
            self._session = AsyncSession(
                self._client,
                self._session_id,
                owned=False,
                keep=self._keep,
                extra_headers=self._extra_headers,
            )
            return self._session
        self._session = AsyncSession(
            self._client,
            created.session_id,
            owned=True,
            keep=self._keep,
            extra_headers=self._extra_headers,
        )
        return self._session

    async def __aexit__(self, *exc: Any) -> None:
        if self._session is not None:
            await self._session.aclose()


__all__ = ["AsyncClient", "AsyncSession"]
