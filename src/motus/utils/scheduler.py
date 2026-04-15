"""Scheduling interface with cloud and in-process backends.

Two implementations:

- **CloudScheduler** — calls the LithosAI notification API (PR #42).
  Used when agents are deployed on the platform.
- **LocalScheduler** — in-process scheduling via stdlib ``sched``.
  Used for local development and testing.

Usage::

    from motus.utils import Scheduler, CloudScheduler, LocalScheduler

    # Cloud — delivers messages into the session via the platform
    scheduler = CloudScheduler()
    await scheduler.schedule("digest", schedule_type="rate",
                             expression="rate(5 minutes)",
                             message="Time to send the digest.")

    # Local — calls a Python function in-process
    scheduler = LocalScheduler()
    scheduler.on("digest", my_handler)
    await scheduler.schedule("digest", schedule_type="rate",
                             expression="rate(5 minutes)")
    asyncio.create_task(scheduler.run())  # drives the event loop
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sched
import time
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("motus.utils.scheduler")

_API_TIMEOUT = 10  # seconds


class SchedulerError(Exception):
    """Raised when a scheduling operation fails."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        self.status = status
        self.body = body
        super().__init__(message)


class Scheduler(ABC):
    """Abstract base for schedulers."""

    @abstractmethod
    async def schedule(
        self,
        name: str,
        *,
        schedule_type: str,
        expression: str,
        message: str = "",
        timezone: str = "UTC",
    ) -> str:
        """Schedule a notification.  Returns a notification/job ID."""
        ...

    @abstractmethod
    async def remove(self, notification_id: str) -> None:
        """Cancel and delete a scheduled notification."""
        ...


# ======================================================================
# Cloud backend — calls the LithosAI notification API
# ======================================================================


class CloudScheduler(Scheduler):
    """Calls the platform notification API to schedule persistent notifications.

    All constructor parameters default to environment variables set by the
    platform at runtime.
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        session_id: str | None = None,
        session_token: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self._api_url = (api_url or os.environ.get("LITHOSAI_API_URL", "")).rstrip("/")
        self._api_key = api_key or os.environ.get("LITHOSAI_API_KEY", "")
        self._session_id = session_id or os.environ.get("MOTUS_SESSION_ID", "")
        self._session_token = session_token or os.environ.get("SESSION_TOKEN", "")
        self._project_id = project_id or os.environ.get("MOTUS_PROJECT", "")

    async def schedule(
        self,
        name: str,
        *,
        schedule_type: str,
        expression: str,
        message: str = "",
        timezone: str = "UTC",
    ) -> str:
        body = {
            "project_id": self._project_id,
            "session_id": self._session_id,
            "session_token": self._session_token,
            "name": name,
            "schedule_type": schedule_type,
            "schedule_expression": expression,
            "message": message,
            "timezone": timezone,
        }
        resp = await self._request("POST", "/notifications", body=body)
        return resp["notification_id"]

    async def remove(self, notification_id: str) -> None:
        await self._request("DELETE", f"/notifications/{notification_id}")

    async def list(self) -> list[dict[str, Any]]:
        """List all notifications for the current user."""
        return await self._request("GET", "/notifications")

    async def get(self, notification_id: str) -> dict[str, Any]:
        """Get a single notification by ID."""
        return await self._request("GET", f"/notifications/{notification_id}")

    # -- HTTP helpers --------------------------------------------------

    def _build_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> urllib.request.Request:
        url = f"{self._api_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        return req

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        req = self._build_request(method, path, body)

        def _do() -> Any:
            try:
                with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
                    raw = resp.read()
                    if not raw:
                        return None
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                error_body = ""
                try:
                    error_body = e.read().decode()
                except Exception:
                    pass
                raise SchedulerError(
                    f"{method} {path} failed: HTTP {e.code}",
                    status=e.code,
                    body=error_body,
                ) from e
            except urllib.error.URLError as e:
                raise SchedulerError(
                    f"{method} {path} failed: {e.reason}",
                ) from e

        return await asyncio.to_thread(_do)


# ======================================================================
# Local backend — in-process scheduling
# ======================================================================


class LocalScheduler(Scheduler):
    """In-process scheduler for local development and testing.

    Register handlers with :meth:`on`, then call :meth:`run` as an
    ``asyncio`` background task to drive the scheduler loop::

        scheduler = LocalScheduler()
        scheduler.on("check-slack", my_handler)
        await scheduler.schedule("check-slack", schedule_type="rate",
                                 expression="rate(60 seconds)")
        asyncio.create_task(scheduler.run())
    """

    def __init__(self) -> None:
        self._scheduler = sched.scheduler(time.time, lambda _: None)
        self._handlers: dict[str, Callable[[], None] | Callable[[], Awaitable[None]]] = {}
        self._jobs: dict[str, _LocalJob] = {}

    def on(self, name: str, handler: Callable[[], None] | Callable[[], Awaitable[None]]) -> None:
        """Register a handler that fires when notification *name* triggers."""
        self._handlers[name] = handler

    async def schedule(
        self,
        name: str,
        *,
        schedule_type: str,
        expression: str,
        message: str = "",
        timezone: str = "UTC",
    ) -> str:
        job_id = f"local-{uuid.uuid4().hex[:12]}"
        interval = _parse_interval(schedule_type, expression)
        job = _LocalJob(name=name, job_id=job_id, interval=interval,
                        one_shot=(schedule_type == "at"))
        self._jobs[job_id] = job
        self._schedule_next(job)
        return job_id

    async def remove(self, notification_id: str) -> None:
        self._jobs.pop(notification_id, None)

    async def run(self) -> None:
        """Background loop — call as ``asyncio.create_task(scheduler.run())``."""
        while True:
            self._scheduler.run(blocking=False)
            await asyncio.sleep(1)

    def _schedule_next(self, job: _LocalJob) -> None:
        if job.job_id not in self._jobs:
            return
        next_time = time.time() + job.interval
        self._scheduler.enterabs(next_time, 0, self._fire, argument=(job,))

    def _fire(self, job: _LocalJob) -> None:
        if job.job_id not in self._jobs:
            return
        handler = self._handlers.get(job.name)
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                # Schedule the coroutine on the running event loop
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    pass
        if job.one_shot:
            self._jobs.pop(job.job_id, None)
        else:
            self._schedule_next(job)


class _LocalJob:
    __slots__ = ("name", "job_id", "interval", "one_shot")

    def __init__(self, name: str, job_id: str, interval: float, one_shot: bool) -> None:
        self.name = name
        self.job_id = job_id
        self.interval = interval
        self.one_shot = one_shot


def _parse_interval(schedule_type: str, expression: str) -> float:
    """Extract a repeat interval in seconds from a schedule expression.

    Handles the subset needed for local development:
    - ``rate(N minutes)``, ``rate(N hours)``, ``rate(N seconds)``
    - ``at(...)`` → 0 (fire once immediately-ish, after 1s)
    - ``cron(...)`` → 60 (fire every minute as a simple approximation)
    """
    if schedule_type == "rate":
        # "rate(5 minutes)" → 300
        inner = expression.removeprefix("rate(").removesuffix(")")
        parts = inner.strip().split()
        if len(parts) == 2:
            n, unit = int(parts[0]), parts[1].rstrip("s")  # "minutes" → "minute"
            multipliers = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
            return n * multipliers.get(unit, 60)
        return 60
    if schedule_type == "at":
        return 1  # fire once after 1 second
    # cron — approximate with 60s for local dev
    return 60
