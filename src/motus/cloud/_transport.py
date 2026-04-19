"""Internal transport primitives shared between sync and async clients."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping

import httpx

from motus.auth.credentials import get_api_key
from motus.serve.schemas import SessionResponse, SessionStatus

from .errors import (
    AgentError,
    AuthError,
    BackendUnavailable,
    InterruptNotFound,
    MotusClientError,
    ProtocolError,
    ServerBusy,
    SessionConflict,
    SessionNotFound,
    SessionTimeout,
    SessionUnsupported,
)

if TYPE_CHECKING:
    pass

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


def map_status_error(r: httpx.Response, *, is_resume: bool = False) -> MotusClientError:
    detail = _extract_detail(r)
    code = r.status_code
    if code in (401, 403):
        return AuthError(detail or f"HTTP {code}", response=r)
    if code == 404:
        if is_resume and "session not found" not in detail.lower():
            return InterruptNotFound(detail or "interrupt not found", response=r)
        return SessionNotFound(detail or "session not found", response=r)
    if code == 405:
        return SessionUnsupported(detail or "operation not allowed", response=r)
    if code == 409:
        return SessionConflict(detail or "session conflict", response=r)
    if code == 503:
        return ServerBusy(detail or "server busy", response=r)
    return BackendUnavailable(detail or f"HTTP {code}", response=r)


def map_transport_error(exc: Exception) -> MotusClientError:
    """Wrap an httpx transport-level exception for non-polling requests."""
    return BackendUnavailable(str(exc) or type(exc).__name__)


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
                _session_url(base_url, session_id),
                params={"wait": "true", "timeout": wait},
                headers=headers,
                timeout=per_call_timeout,
            )
        except httpx.TimeoutException as e:
            consecutive_read_timeouts += 1
            if consecutive_read_timeouts >= read_retry_budget:
                raise BackendUnavailable(
                    f"poll read timeout retries exhausted ({read_retry_budget})"
                ) from e
            continue
        except httpx.TransportError as e:
            raise BackendUnavailable(str(e)) from e

        if r.is_error:
            raise map_status_error(r)

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
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"message POST timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r)


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
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"message POST timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r)
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
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"resume POST timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r, is_resume=True)
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
        raise AgentError(snapshot.error or "agent error", session_id=session_id)
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
                _session_url(base_url, session_id),
                params={"wait": "true", "timeout": wait},
                headers=headers,
                timeout=per_call_timeout,
            )
        except httpx.TimeoutException as e:
            consecutive_read_timeouts += 1
            if consecutive_read_timeouts >= read_retry_budget:
                raise BackendUnavailable(
                    f"poll read timeout retries exhausted ({read_retry_budget})"
                ) from e
            continue
        except httpx.TransportError as e:
            raise BackendUnavailable(str(e)) from e

        if r.is_error:
            raise map_status_error(r)

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
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"message POST timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r)


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
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"message POST timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r)
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
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"resume POST timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r, is_resume=True)
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
) -> httpx.Response:
    try:
        r = http.request(
            method, url, headers=headers, json=json, params=params, timeout=timeout
        )
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"{method} {url} timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r, is_resume=is_resume)
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
) -> httpx.Response:
    try:
        r = await http.request(
            method, url, headers=headers, json=json, params=params, timeout=timeout
        )
    except httpx.TimeoutException as e:
        raise BackendUnavailable(f"{method} {url} timeout: {e}") from e
    except httpx.TransportError as e:
        raise map_transport_error(e) from e
    if r.is_error:
        raise map_status_error(r, is_resume=is_resume)
    return r


def wait_http_timeout(
    http_timeout: httpx.Timeout, wait_seconds: float
) -> httpx.Timeout:
    """Derive a one-off httpx.Timeout whose read deadline can absorb ``wait_seconds``.

    Used for ``GET /sessions/{id}?wait=true&timeout=N`` when the caller asks the
    server to block longer than the client's default read timeout — otherwise
    the HTTP layer would surface BackendUnavailable before the server-side
    ``?timeout`` expires.
    """
    margin = 10.0
    needed_read = wait_seconds + margin
    current_read = http_timeout.read if http_timeout.read is not None else 0.0
    if needed_read <= current_read:
        return http_timeout
    return httpx.Timeout(
        connect=http_timeout.connect,
        read=needed_read,
        write=http_timeout.write,
        pool=http_timeout.pool,
    )
