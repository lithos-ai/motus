"""Single-agent, session-based REST server."""

import asyncio
import functools
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Callable

import httpx
from fastapi import FastAPI, HTTPException, Response

from motus.models import ChatMessage

from .schemas import (
    CreateSessionRequest,
    HealthResponse,
    InterruptInfo,
    MessageRequest,
    MessageResponse,
    ResumeRequest,
    SessionResponse,
    SessionStatus,
    SessionSummary,
    TraceMetrics,
    WebhookPayload,
    WebhookSpec,
)
from .session import Session, SessionAlreadyExists, SessionLimitReached, SessionStore
from .worker import WorkerExecutor, WorkerResult

logger = logging.getLogger("motus.serve")


class AgentServer:
    """Single-agent HTTP server with session-based conversations.

    Example::

        # myapp.py
        from motus.models import ChatMessage

        def my_agent(message, state):
            response = ChatMessage.assistant_message(content="hello")
            return response, state + [message, response]

    Then start with the CLI::

        python -m motus.serve start myapp:my_agent --port 8000
    """

    def __init__(
        self,
        agent_fn: Callable | str,
        *,
        max_workers: int | None = None,
        ttl: float = 0,
        timeout: float = 0,
        max_sessions: int = 0,
        shutdown_timeout: float = 0,
        allow_custom_ids: bool = False,
    ):
        if isinstance(agent_fn, str):
            self._import_path = agent_fn
        else:
            if agent_fn.__qualname__ != agent_fn.__name__:
                raise ValueError(
                    f"Agent function must be a module-level function, "
                    f"got {agent_fn.__module__}:{agent_fn.__qualname__}"
                )
            self._import_path = f"{agent_fn.__module__}:{agent_fn.__qualname__}"
        self._executor = WorkerExecutor(
            max_workers=max_workers, import_path=self._import_path
        )
        self._sessions = SessionStore(ttl=ttl, max_sessions=max_sessions)
        self._timeout = timeout
        self._shutdown_timeout = shutdown_timeout
        self._allow_custom_ids = allow_custom_ids
        self._background_tasks: set[asyncio.Task] = set()
        logger.info(f"Agent: {self._import_path}")

    @functools.cached_property
    def app(self) -> FastAPI:
        return self._create_app()

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info",
    ) -> None:
        import uvicorn
        from uvicorn.config import LOGGING_CONFIG

        log_config = {**LOGGING_CONFIG}
        log_config["loggers"]["motus.serve"] = {
            "handlers": ["default"],
            "level": log_level.upper(),
            "propagate": False,
        }
        uvicorn.run(
            self.app, host=host, port=port, log_level=log_level, log_config=log_config
        )

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            sweep_task = None
            if self._sessions.ttl > 0:
                sweep_task = asyncio.create_task(self._sweep_loop())
            try:
                yield
            finally:
                if sweep_task is not None:
                    sweep_task.cancel()
                await self._shutdown()

        app = FastAPI(
            title="Motus Agent Server",
            version="0.2.0",
            lifespan=lifespan,
        )

        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status="ok",
                max_workers=self._executor.max_workers,
                running_workers=self._executor.running_workers,
                total_sessions=len(self._sessions),
            )

        @app.post("/sessions", response_model=SessionResponse, status_code=201)
        async def create_session(
            response: Response, request: CreateSessionRequest | None = None
        ):
            try:
                state = request.state if request else []
                session = self._sessions.create(state=state)
            except SessionLimitReached:
                raise HTTPException(
                    status_code=503, detail="Maximum number of sessions reached"
                )
            logger.info(f"Session created: {session.session_id}")
            response.headers["Location"] = f"/sessions/{session.session_id}"
            return SessionResponse(
                session_id=session.session_id,
                status=SessionStatus.idle,
            )

        @app.put(
            "/sessions/{session_id}", response_model=SessionResponse, status_code=201
        )
        async def create_session_with_id(
            session_id: str,
            response: Response,
            request: CreateSessionRequest | None = None,
        ):
            if not self._allow_custom_ids:
                raise HTTPException(
                    status_code=405, detail="Custom session IDs are not enabled"
                )
            try:
                uuid.UUID(session_id)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Session ID must be a valid UUID"
                )
            try:
                state = request.state if request else []
                session = self._sessions.create(state=state, session_id=session_id)
            except SessionAlreadyExists:
                raise HTTPException(status_code=409, detail="Session already exists")
            except SessionLimitReached:
                raise HTTPException(
                    status_code=503, detail="Maximum number of sessions reached"
                )
            logger.info(f"Session created: {session.session_id}")
            response.headers["Location"] = f"/sessions/{session.session_id}"
            return SessionResponse(
                session_id=session.session_id,
                status=SessionStatus.idle,
            )

        @app.get("/sessions", response_model=list[SessionSummary])
        async def list_sessions():
            sessions = self._sessions.list()
            return [
                SessionSummary(
                    session_id=s.session_id,
                    total_messages=len(s.state),
                    status=s.status,
                )
                for s in sessions
            ]

        @app.get("/sessions/{session_id}", response_model=SessionResponse)
        async def get_session(
            session_id: str,
            wait: bool = False,
            timeout: float | None = None,
        ):
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            if wait and session.status == SessionStatus.running:
                await session.wait(timeout=timeout)
                # Re-fetch since session may have changed or been deleted
                session = self._sessions.get(session_id)
                if session is None:
                    raise HTTPException(status_code=404, detail="Session deleted")

            interrupts = None
            if session.status == SessionStatus.interrupted:
                interrupts = [
                    InterruptInfo(
                        interrupt_id=iid,
                        type=msg.payload.get("type", "unknown"),
                        payload=msg.payload,
                    )
                    for iid, msg in session.pending_interrupts.items()
                ]

            return SessionResponse(
                session_id=session.session_id,
                status=session.status,
                response=session.response,
                error=session.error,
                interrupts=interrupts,
            )

        @app.delete("/sessions/{session_id}", status_code=204)
        async def delete_session(session_id: str):
            if not self._sessions.delete(session_id):
                raise HTTPException(status_code=404, detail="Session not found")
            logger.info(f"Session deleted: {session_id}")

        @app.get(
            "/sessions/{session_id}/messages",
            response_model=list[ChatMessage],
            response_model_exclude_none=True,
        )
        async def get_session_messages(session_id: str):
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return session.state

        @app.post(
            "/sessions/{session_id}/messages",
            response_model=MessageResponse,
            status_code=202,
        )
        async def send_message(
            session_id: str, request: MessageRequest, response: Response
        ):
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            # create_task schedules but doesn't yield, so everything from
            # the status check through start_turn is atomic under asyncio
            # cooperative scheduling (no other coroutine can interleave).
            if session.status in (SessionStatus.running, SessionStatus.interrupted):
                raise HTTPException(
                    status_code=409,
                    detail=f"Session is {session.status.value}",
                )

            webhook = request.webhook
            message = ChatMessage(
                **request.model_dump(exclude_none=True, exclude={"webhook"})
            )
            task = asyncio.create_task(
                self._run_turn(session_id, message, webhook=webhook)
            )
            session.start_turn(task)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            response.headers["Location"] = f"/sessions/{session_id}"
            return MessageResponse(
                session_id=session.session_id,
                status=SessionStatus.running,
            )

        @app.post("/sessions/{session_id}/resume", status_code=200)
        async def resume_session(session_id: str, body: ResumeRequest) -> dict:
            """Submit a resume for a pending interrupt."""
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            if session.status != SessionStatus.interrupted:
                raise HTTPException(
                    status_code=409,
                    detail=f"Session is {session.status.value}, not interrupted",
                )
            try:
                session.submit_resume(body.interrupt_id, body.value)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return {"session_id": session_id, "status": session.status.value}

        return app

    async def _run_turn(
        self,
        session_id: str,
        message: ChatMessage,
        *,
        webhook: WebhookSpec | None = None,
    ) -> None:
        """Run an agent turn in the background and update the session."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        logger.info(f"Turn started: session={session_id}")

        resume_queue: asyncio.Queue = asyncio.Queue()
        session._resume_queue = resume_queue

        def on_interrupt(msg) -> None:
            session.interrupt_turn(msg)

        def on_worker_done() -> None:
            session._resume_queue = None

        result: WorkerResult | None = None
        try:
            result = await self._executor.submit_turn(
                import_path=self._import_path,
                message=message,
                state=list(session.state),
                timeout=self._timeout,
                session_id=session_id,
                on_interrupt=on_interrupt,
                resume_queue=resume_queue,
                on_worker_done=on_worker_done,
            )

            session = self._sessions.get(session_id)
            if session is None:
                return

            if result.success:
                response, new_state = result.value
                session.complete_turn(response, new_state)
                logger.info(f"Turn completed: session={session_id}")
            else:
                session.fail_turn(result.error or "Unknown error")
                logger.error(f"Turn failed: session={session_id}\n{result.error}")

        except asyncio.CancelledError:
            logger.info(f"Turn cancelled: session={session_id}")
            session = self._sessions.get(session_id)
            if session is not None:
                session.fail_turn("Turn cancelled")
        finally:
            session = self._sessions.get(session_id)
            if session is not None:
                session._resume_queue = None

        if webhook is not None:
            session = self._sessions.get(session_id)
            if session is not None:
                trace_metrics = result.trace_metrics if result is not None else None
                await self._deliver_webhook(
                    session, webhook, trace_metrics=trace_metrics
                )

    async def _deliver_webhook(
        self, session: Session, webhook: WebhookSpec, trace_metrics=None
    ) -> None:
        """POST turn result to the webhook URL.

        TODO: Add retry logic to make webhook delivery more robust.
        """
        payload = WebhookPayload(
            session_id=session.session_id,
            status=session.status,
            response=session.response,
            error=session.error,
            messages=list(session.state) if webhook.include_messages else None,
            trace_metrics=TraceMetrics(**trace_metrics) if trace_metrics else None,
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if webhook.token is not None:
            headers["Authorization"] = f"Bearer {webhook.token}"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    webhook.url,
                    content=payload.model_dump_json(exclude_none=True),
                    headers=headers,
                    timeout=10.0,
                )
            logger.info(
                f"Webhook delivered: session={session.session_id} "
                f"url={webhook.url} status={resp.status_code}"
            )
        except Exception:
            logger.exception(
                f"Webhook delivery failed: session={session.session_id} url={webhook.url}"
            )

    async def _sweep_loop(self) -> None:
        """Periodically sweep expired sessions."""
        interval = max(self._sessions.ttl / 2, 30)
        while True:
            await asyncio.sleep(interval)
            self._sessions.sweep()

    async def _shutdown(self) -> None:
        """Wait for in-flight tasks on server shutdown, cancelling stragglers."""
        if not self._background_tasks:
            return
        logger.info(f"Waiting for {len(self._background_tasks)} in-flight task(s)")
        timeout = self._shutdown_timeout or None
        done, pending = await asyncio.wait(self._background_tasks, timeout=timeout)
        if pending:
            logger.warning(f"Cancelling {len(pending)} timed-out task(s)")
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
