"""Internal transport primitives shared between sync and async clients."""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Mapping

import httpx

from motus.auth.credentials import get_api_key
from motus.serve.schemas import SessionResponse, SessionStatus

from .errors import (
    AgentError,
    AuthError,
    BackendUnavailable,
    BadRequest,
    ErrorContext,
    InterruptNotFound,
    MotusClientError,
    ProtocolError,
    ServerBusy,
    SessionConflict,
    SessionNotFound,
    SessionTimeout,
    SessionUnsupported,
)

USER_AGENT_PREFIX = "motus-client"
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
DEFAULT_SERVER_WAIT_SLICE = 90.0
DEFAULT_READ_RETRY_BUDGET = 3

logger = logging.getLogger("motus.cloud")


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("lithosai-motus")
    except PackageNotFoundError:
        return "0.0.0"


USER_AGENT = f"{USER_AGENT_PREFIX}/{_package_version()}"


def resolve_api_key(explicit: str | None) -> str | None:
    """Explicit arg > LITHOSAI_API_KEY > ~/.motus/credentials.json > None.

    Empty string passed explicitly is treated as "no key" (no Authorization header).
    """
    if explicit is not None:
        return explicit or None
    return get_api_key()


def validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("base_url is required and must be a non-empty string")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        raise ValueError(f"base_url must be http(s); got: {base_url!r}")
    return base_url.rstrip("/")


def build_headers(
    api_key: str | None,
    constructor_headers: Mapping[str, str] | None,
    per_call_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if constructor_headers:
        headers.update(constructor_headers)
    if per_call_headers:
        headers.update(per_call_headers)
    # Resolved credentials (explicit api_key / env / credentials file) take
    # precedence over any Authorization accidentally forwarded via
    # extra_headers, so downstream never authenticates as the wrong principal.
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_detail(r: httpx.Response) -> str:
    try:
        data = r.json()
    except Exception:
        return r.text
    if isinstance(data, dict) and "detail" in data:
        return str(data["detail"])
    return r.text


def _response_context(
    r: httpx.Response,
    *,
    session_id: str | None,
    interrupt_id: str | None,
) -> ErrorContext:
    return ErrorContext(
        session_id=session_id,
        interrupt_id=interrupt_id,
        method=r.request.method,
        url=str(r.request.url),
        status_code=r.status_code,
    )


def _transport_context(
    exc: Exception,
    *,
    session_id: str | None,
    interrupt_id: str | None,
) -> ErrorContext:
    req = getattr(exc, "request", None)
    return ErrorContext(
        session_id=session_id,
        interrupt_id=interrupt_id,
        method=getattr(req, "method", None),
        url=str(req.url) if req is not None else None,
    )


def map_status_error(
    r: httpx.Response,
    *,
    is_resume: bool = False,
    session_id: str | None = None,
    interrupt_id: str | None = None,
) -> MotusClientError:
    detail = _extract_detail(r)
    code = r.status_code
    ctx = _response_context(r, session_id=session_id, interrupt_id=interrupt_id)
    if code in (401, 403):
        return AuthError(detail or f"HTTP {code}", response=r, context=ctx)
    if code == 404:
        if is_resume and "session not found" not in detail.lower():
            return InterruptNotFound(
                detail or "interrupt not found", response=r, context=ctx
            )
        return SessionNotFound(detail or "session not found", response=r, context=ctx)
    if code == 405:
        return SessionUnsupported(
            detail or "operation not allowed", response=r, context=ctx
        )
    if code == 409:
        return SessionConflict(detail or "session conflict", response=r, context=ctx)
    if code == 503:
        return ServerBusy(detail or "server busy", response=r, context=ctx)
    if 400 <= code < 500:
        # Unmapped 4xx — validation errors (422), rate limits (429), etc.
        # Retrying without changing the request will not help.
        return BadRequest(detail or f"HTTP {code}", response=r, context=ctx)
    return BackendUnavailable(detail or f"HTTP {code}", response=r, context=ctx)


def map_transport_error(
    exc: Exception,
    *,
    session_id: str | None = None,
    interrupt_id: str | None = None,
) -> MotusClientError:
    """Wrap an httpx transport-level exception (Timeout or Transport) as BackendUnavailable."""
    ctx = _transport_context(exc, session_id=session_id, interrupt_id=interrupt_id)
    if isinstance(exc, httpx.TimeoutException):
        where = f"{ctx.method} {ctx.url}" if ctx.method and ctx.url else "request"
        message = f"{where} timeout: {exc}"
    else:
        message = str(exc) or type(exc).__name__
    return BackendUnavailable(message, context=ctx)


def decode_json(r: httpx.Response) -> Any:
    """Decode ``r`` as JSON, mapping parse failures to ProtocolError."""
    try:
        return r.json()
    except Exception as e:
        raise ProtocolError(
            f"non-JSON response from {r.request.method} {r.request.url}: {e!r}"
        ) from e


def parse_session_response(r: httpx.Response) -> SessionResponse:
    """Decode a response body and validate it as a SessionResponse."""
    return parse_session(decode_json(r))


def parse_session(data: Any) -> SessionResponse:
    try:
        return SessionResponse.model_validate(data)
    except Exception as e:
        raise ProtocolError(f"invalid session response: {e!r}") from e


def decode_json_as(r: httpx.Response, schema: Any) -> Any:
    """Decode ``r`` and validate the body against a pydantic model / adapter.

    Accepts any object exposing ``model_validate`` (pydantic BaseModel or TypeAdapter).
    """
    data = decode_json(r)
    try:
        return (
            schema.model_validate(data)
            if hasattr(schema, "model_validate")
            else schema.validate_python(data)
        )
    except Exception as e:
        raise ProtocolError(f"invalid response body: {e!r}") from e


def decode_json_validated(r: httpx.Response, schema: Any) -> Any:
    """Validate the response body against ``schema`` and return a plain dict/list.

    Raises ``ProtocolError`` when the JSON is missing required fields or has
    the wrong shape. Accepts any object with ``model_validate``/``model_dump``
    (pydantic BaseModel) or ``validate_python``/``dump_python`` (TypeAdapter).
    """
    data = decode_json(r)
    try:
        if hasattr(schema, "model_validate"):
            obj = schema.model_validate(data)
            return obj.model_dump(mode="json", exclude_none=True)
        obj = schema.validate_python(data)
        return schema.dump_python(obj, mode="json", exclude_none=True)
    except ProtocolError:
        raise
    except Exception as e:
        raise ProtocolError(f"invalid response body: {e!r}") from e


def _deadline(turn_timeout: float | None) -> float | None:
    return None if turn_timeout is None else time.monotonic() + turn_timeout


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else deadline - time.monotonic()


def _wait_for(deadline: float | None, server_wait_slice: float) -> float:
    remaining = _remaining(deadline)
    return server_wait_slice if remaining is None else min(remaining, server_wait_slice)


def _session_url(base_url: str, session_id: str) -> str:
    return f"{base_url}/sessions/{session_id}"


def _messages_url(base_url: str, session_id: str) -> str:
    return f"{base_url}/sessions/{session_id}/messages"


def _resume_url(base_url: str, session_id: str) -> str:
    return f"{base_url}/sessions/{session_id}/resume"


# ----------------------------- Sync poll loop -----------------------------


def sync_poll_until_terminal(
    http: httpx.Client,
    base_url: str,
    session_id: str,
    *,
    turn_timeout: float | None,
    server_wait_slice: float,
    read_retry_budget: int,
    headers: Mapping[str, str],
) -> SessionResponse:
    """Long-poll GET /sessions/{id} until a terminal status or client deadline."""
    deadline = _deadline(turn_timeout)
    consecutive_read_timeouts = 0
    last_snapshot: SessionResponse | None = None
    poll_url = _session_url(base_url, session_id)

    while True:
        remaining = _remaining(deadline)
        if remaining is not None and remaining <= 0:
            raise SessionTimeout(
                f"turn deadline exceeded (session={session_id})",
                session_id=session_id,
                elapsed=turn_timeout,
                last_snapshot=last_snapshot,
            )
        wait = _wait_for(deadline, server_wait_slice)
        per_call_timeout = wait_http_timeout(http.timeout, wait)
        try:
            r = http.get(
                poll_url,
                params={"wait": "true", "timeout": wait},
                headers=headers,
                timeout=per_call_timeout,
            )
        except httpx.TimeoutException as e:
            # A stalled poll may have consumed most or all of the turn's
            # deadline. Re-check before retrying so the caller sees a
            # SessionTimeout once they hit their budget, not a BackendUnavailable
            # from retry exhaustion.
            if deadline is not None and time.monotonic() >= deadline:
                raise SessionTimeout(
                    f"turn deadline exceeded (session={session_id})",
                    session_id=session_id,
                    elapsed=turn_timeout,
                    last_snapshot=last_snapshot,
                ) from e
            consecutive_read_timeouts += 1
            if consecutive_read_timeouts >= read_retry_budget:
                raise BackendUnavailable(
                    f"poll read timeout retries exhausted ({read_retry_budget})",
                    context=ErrorContext(
                        session_id=session_id, method="GET", url=poll_url
                    ),
                ) from e
            continue
        except httpx.TransportError as e:
            raise map_transport_error(e, session_id=session_id) from e

        if r.is_error:
            raise map_status_error(r, session_id=session_id)

        consecutive_read_timeouts = 0
        snapshot = parse_session_response(r)
        last_snapshot = snapshot
        if snapshot.status in (
            SessionStatus.idle,
            SessionStatus.interrupted,
            SessionStatus.error,
        ):
            return snapshot


def sync_post_message(
    http: httpx.Client,
    base_url: str,
    session_id: str,
    body: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
) -> None:
    """POST /messages. Raises MotusClientError on failure; returns None on 202."""
    try:
        r = http.post(
            _messages_url(base_url, session_id), json=dict(body), headers=headers
        )
    except httpx.TransportError as e:
        raise map_transport_error(e, session_id=session_id) from e
    if r.is_error:
        raise map_status_error(r, session_id=session_id)


def sync_send_and_poll(
    http: httpx.Client,
    base_url: str,
    session_id: str,
    body: Mapping[str, Any],
    *,
    turn_timeout: float | None,
    server_wait_slice: float,
    read_retry_budget: int,
    headers: Mapping[str, str],
) -> SessionResponse:
    """POST /messages then poll until terminal. Returns final SessionResponse."""
    try:
        r = http.post(
            _messages_url(base_url, session_id), json=dict(body), headers=headers
        )
    except httpx.TransportError as e:
        raise map_transport_error(e, session_id=session_id) from e
    if r.is_error:
        raise map_status_error(r, session_id=session_id)
    return sync_poll_until_terminal(
        http,
        base_url,
        session_id,
        turn_timeout=turn_timeout,
        server_wait_slice=server_wait_slice,
        read_retry_budget=read_retry_budget,
        headers=headers,
    )


def sync_resume_and_poll(
    http: httpx.Client,
    base_url: str,
    session_id: str,
    interrupt_id: str,
    value: Any,
    *,
    turn_timeout: float | None,
    server_wait_slice: float,
    read_retry_budget: int,
    headers: Mapping[str, str],
) -> SessionResponse:
    """POST /resume then poll until terminal."""
    try:
        r = http.post(
            _resume_url(base_url, session_id),
            json={"interrupt_id": interrupt_id, "value": value},
            headers=headers,
        )
    except httpx.TransportError as e:
        raise map_transport_error(
            e, session_id=session_id, interrupt_id=interrupt_id
        ) from e
    if r.is_error:
        raise map_status_error(
            r, is_resume=True, session_id=session_id, interrupt_id=interrupt_id
        )
    return sync_poll_until_terminal(
        http,
        base_url,
        session_id,
        turn_timeout=turn_timeout,
        server_wait_slice=server_wait_slice,
        read_retry_budget=read_retry_budget,
        headers=headers,
    )


def finalize_snapshot(snapshot: SessionResponse, session_id: str) -> SessionResponse:
    """Raise AgentError for status=error; otherwise return snapshot unchanged."""
    if snapshot.status == SessionStatus.error:
        raise AgentError(
            snapshot.error or "agent error",
            session_id=session_id,
            context=ErrorContext(session_id=session_id),
        )
    return snapshot


# ----------------------------- Async poll loop -----------------------------


async def async_poll_until_terminal(
    http: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    *,
    turn_timeout: float | None,
    server_wait_slice: float,
    read_retry_budget: int,
    headers: Mapping[str, str],
) -> SessionResponse:
    deadline = _deadline(turn_timeout)
    consecutive_read_timeouts = 0
    last_snapshot: SessionResponse | None = None
    poll_url = _session_url(base_url, session_id)

    while True:
        remaining = _remaining(deadline)
        if remaining is not None and remaining <= 0:
            raise SessionTimeout(
                f"turn deadline exceeded (session={session_id})",
                session_id=session_id,
                elapsed=turn_timeout,
                last_snapshot=last_snapshot,
            )
        wait = _wait_for(deadline, server_wait_slice)
        per_call_timeout = wait_http_timeout(http.timeout, wait)
        try:
            r = await http.get(
                poll_url,
                params={"wait": "true", "timeout": wait},
                headers=headers,
                timeout=per_call_timeout,
            )
        except httpx.TimeoutException as e:
            if deadline is not None and time.monotonic() >= deadline:
                raise SessionTimeout(
                    f"turn deadline exceeded (session={session_id})",
                    session_id=session_id,
                    elapsed=turn_timeout,
                    last_snapshot=last_snapshot,
                ) from e
            consecutive_read_timeouts += 1
            if consecutive_read_timeouts >= read_retry_budget:
                raise BackendUnavailable(
                    f"poll read timeout retries exhausted ({read_retry_budget})",
                    context=ErrorContext(
                        session_id=session_id, method="GET", url=poll_url
                    ),
                ) from e
            continue
        except httpx.TransportError as e:
            raise map_transport_error(e, session_id=session_id) from e

        if r.is_error:
            raise map_status_error(r, session_id=session_id)

        consecutive_read_timeouts = 0
        snapshot = parse_session_response(r)
        last_snapshot = snapshot
        if snapshot.status in (
            SessionStatus.idle,
            SessionStatus.interrupted,
            SessionStatus.error,
        ):
            return snapshot


async def async_post_message(
    http: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    body: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
) -> None:
    """Async POST /messages."""
    try:
        r = await http.post(
            _messages_url(base_url, session_id), json=dict(body), headers=headers
        )
    except httpx.TransportError as e:
        raise map_transport_error(e, session_id=session_id) from e
    if r.is_error:
        raise map_status_error(r, session_id=session_id)


async def async_send_and_poll(
    http: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    body: Mapping[str, Any],
    *,
    turn_timeout: float | None,
    server_wait_slice: float,
    read_retry_budget: int,
    headers: Mapping[str, str],
) -> SessionResponse:
    try:
        r = await http.post(
            _messages_url(base_url, session_id), json=dict(body), headers=headers
        )
    except httpx.TransportError as e:
        raise map_transport_error(e, session_id=session_id) from e
    if r.is_error:
        raise map_status_error(r, session_id=session_id)
    return await async_poll_until_terminal(
        http,
        base_url,
        session_id,
        turn_timeout=turn_timeout,
        server_wait_slice=server_wait_slice,
        read_retry_budget=read_retry_budget,
        headers=headers,
    )


async def async_resume_and_poll(
    http: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    interrupt_id: str,
    value: Any,
    *,
    turn_timeout: float | None,
    server_wait_slice: float,
    read_retry_budget: int,
    headers: Mapping[str, str],
) -> SessionResponse:
    try:
        r = await http.post(
            _resume_url(base_url, session_id),
            json={"interrupt_id": interrupt_id, "value": value},
            headers=headers,
        )
    except httpx.TransportError as e:
        raise map_transport_error(
            e, session_id=session_id, interrupt_id=interrupt_id
        ) from e
    if r.is_error:
        raise map_status_error(
            r, is_resume=True, session_id=session_id, interrupt_id=interrupt_id
        )
    return await async_poll_until_terminal(
        http,
        base_url,
        session_id,
        turn_timeout=turn_timeout,
        server_wait_slice=server_wait_slice,
        read_retry_budget=read_retry_budget,
        headers=headers,
    )


# --------------- Callable helpers (used by Client/AsyncClient) ---------------


SyncRequestFn = Callable[..., httpx.Response]
AsyncRequestFn = Callable[..., Awaitable[httpx.Response]]


def sync_request(
    http: httpx.Client,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    json: Any = None,
    params: Mapping[str, Any] | None = None,
    is_resume: bool = False,
    timeout: Any = httpx.USE_CLIENT_DEFAULT,
    session_id: str | None = None,
    interrupt_id: str | None = None,
) -> httpx.Response:
    try:
        r = http.request(
            method, url, headers=headers, json=json, params=params, timeout=timeout
        )
    except httpx.TransportError as e:
        raise map_transport_error(
            e, session_id=session_id, interrupt_id=interrupt_id
        ) from e
    if r.is_error:
        raise map_status_error(
            r,
            is_resume=is_resume,
            session_id=session_id,
            interrupt_id=interrupt_id,
        )
    return r


async def async_request(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    json: Any = None,
    params: Mapping[str, Any] | None = None,
    is_resume: bool = False,
    timeout: Any = httpx.USE_CLIENT_DEFAULT,
    session_id: str | None = None,
    interrupt_id: str | None = None,
) -> httpx.Response:
    try:
        r = await http.request(
            method, url, headers=headers, json=json, params=params, timeout=timeout
        )
    except httpx.TransportError as e:
        raise map_transport_error(
            e, session_id=session_id, interrupt_id=interrupt_id
        ) from e
    if r.is_error:
        raise map_status_error(
            r,
            is_resume=is_resume,
            session_id=session_id,
            interrupt_id=interrupt_id,
        )
    return r


def wait_http_timeout(
    http_timeout: httpx.Timeout, wait_seconds: float
) -> httpx.Timeout:
    """Derive a one-off httpx.Timeout whose read deadline can absorb ``wait_seconds``.

    Used for ``GET /sessions/{id}?wait=true&timeout=N`` when the caller asks the
    server to block longer than the client's default read timeout — otherwise
    the HTTP layer would surface BackendUnavailable before the server-side
    ``?timeout`` expires. If the caller intentionally configured ``read=None``
    (unbounded), the timeout is returned unchanged so an infinite read budget
    is never silently shrunk to a finite one.
    """
    # Unbounded reads are always sufficient — never replace None with a finite value.
    if http_timeout.read is None:
        return http_timeout
    margin = 10.0
    needed_read = wait_seconds + margin
    if needed_read <= http_timeout.read:
        return http_timeout
    return httpx.Timeout(
        connect=http_timeout.connect,
        read=needed_read,
        write=http_timeout.write,
        pool=http_timeout.pool,
    )
