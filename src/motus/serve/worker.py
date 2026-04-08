"""Process-per-request worker execution for serve.

Each turn spawns a fresh subprocess via multiprocessing.Process.
A semaphore limits concurrency to max_workers.
"""

import asyncio
import importlib
import inspect
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable

from motus.models import ChatMessage
from motus.serve.interrupt import InterruptMessage

logger = logging.getLogger("motus.serve.worker")

DEFAULT_MAX_WORKERS = 4


@dataclass
class WorkerResult:
    """Result wrapper to distinguish success from failure without raising."""

    success: bool
    value: Any = None
    error: str | None = None
    trace_metrics: dict | None = None


def _resolve_import_path(import_path: str):
    """Resolve 'pkg.module:variable' to an Agent instance or callable."""
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path '{import_path}', expected 'module:variable'"
        )
    module_path, attr_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name)
    return obj


def _is_openai_agent(obj) -> bool:
    """Check if obj is an OpenAI Agents SDK Agent without top-level import."""
    try:
        from agents import Agent as _OAIAgent  # type: ignore[import-not-found]

        return isinstance(obj, _OAIAgent)
    except ImportError:
        return False


def _adapt_openai_agent(obj):
    """Wrap an OpenAI Agents SDK Agent into a serve-compatible async function."""

    async def _run_turn(message: ChatMessage, state: list[ChatMessage]):
        from motus.openai_agents import Runner

        oai_input: str | list = message.content or ""
        if state:
            oai_input = [
                {"role": m.role, "content": m.content or ""}
                for m in state
                if m.role in ("user", "assistant", "system")
            ] + [{"role": "user", "content": message.content or ""}]

        try:
            result = await Runner.run(obj, oai_input)
        except Exception as exc:
            # Guardrail tripwires are not errors — the agent refused the request.
            # Return a clean refusal message so the session stays in idle state.
            if "GuardrailTripwireTriggered" in type(exc).__name__:
                output = f"Request blocked by guardrail: {exc}"
                response = ChatMessage.assistant_message(content=output)
                return response, state + [message, response]
            raise

        output = result.final_output
        if not isinstance(output, str):
            # Structured output (Pydantic model, dataclass, etc.) — serialize
            if hasattr(output, "model_dump_json"):
                output = output.model_dump_json()
            else:
                output = str(output)
        response = ChatMessage.assistant_message(content=output)
        return response, state + [message, response]

    return _run_turn


def _validate_result(result) -> tuple[ChatMessage, list[ChatMessage]]:
    """Validate and unpack an agent function's return value."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError(
            f"Agent must return a (response, state) tuple, got {type(result).__name__}"
        )
    response, new_state = result
    if not isinstance(response, ChatMessage):
        raise TypeError(
            f"Agent response must be a ChatMessage, got {type(response).__name__}"
        )
    if not isinstance(new_state, list) or not all(
        isinstance(m, ChatMessage) for m in new_state
    ):
        raise TypeError(
            f"Agent state must be a list[ChatMessage], got {type(new_state).__name__}"
        )
    return response, new_state


def _get_trace_metrics() -> dict | None:
    """Collect trace metrics from the motus runtime if available."""
    try:
        from motus.runtime.agent_runtime import get_runtime

        rt = get_runtime()
        if hasattr(rt, "scheduler") and hasattr(rt.scheduler, "tracer"):
            return rt.scheduler.tracer.get_turn_metrics()
    except Exception:
        pass
    return None


def _finalize_trace() -> None:
    """Flush remaining spans and mark the trace as complete.

    Called after the agent turn finishes but BEFORE sending the result back,
    so the trace is fully written to the cloud API regardless of how quickly
    the main process kills this subprocess.
    """
    try:
        from motus.runtime.agent_runtime import get_runtime

        rt = get_runtime()
        if hasattr(rt, "scheduler") and hasattr(rt.scheduler, "tracer"):
            exporter = rt.scheduler.tracer._cloud_exporter
            if exporter is not None:
                exporter.close()  # flush remaining spans + POST /complete
    except Exception:
        pass


def _worker_entry(conn, import_path, message, state, session_id=None):
    """Subprocess entry point that runs an agent and sends the result over pipe."""
    # Set session_id before any motus import triggers runtime init
    if session_id:
        os.environ["MOTUS_SESSION_ID"] = session_id

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    async def _main():
        from motus.serve.interrupt import _init_interrupt_channel

        _init_interrupt_channel(conn)

        from motus.serve.protocol import ServableAgent

        agent_or_fn = _resolve_import_path(import_path)

        if session_id:
            try:
                from motus.runtime.agent_runtime import get_runtime

                rt = get_runtime()
                if hasattr(rt, "scheduler") and hasattr(rt.scheduler, "tracer"):
                    rt.scheduler.tracer.set_session_id(session_id)
            except Exception:
                pass  # tracer unavailable is not fatal

        if isinstance(agent_or_fn, ServableAgent):
            return await agent_or_fn.run_turn(message, state)
        elif _is_openai_agent(agent_or_fn):
            adapted = _adapt_openai_agent(agent_or_fn)
            return await adapted(message, state)
        elif callable(agent_or_fn):
            if inspect.iscoroutinefunction(agent_or_fn):
                return await agent_or_fn(message, state)
            else:
                # Sync callables must run off-loop to avoid deadlocking
                # any @agent_task they call internally.
                return await asyncio.to_thread(agent_or_fn, message, state)
        else:
            raise TypeError(
                f"'{import_path}' resolved to {type(agent_or_fn).__name__}, "
                f"expected a ServableAgent, OpenAI Agent, or callable"
            )

    try:
        result = asyncio.run(_main())
        response, new_state = _validate_result(result)
        metrics = _get_trace_metrics()
        _finalize_trace()
        conn.send(
            WorkerResult(
                success=True,
                value=(response, new_state),
                trace_metrics=metrics,
            )
        )
    except Exception:
        metrics = _get_trace_metrics()
        _finalize_trace()
        conn.send(
            WorkerResult(
                success=False,
                error=traceback.format_exc(),
                trace_metrics=metrics,
            )
        )
    finally:
        conn.close()
        try:
            from motus.runtime.agent_runtime import shutdown as _rt_shutdown

            _rt_shutdown()
        except Exception:
            pass


def _run_worker(
    conn,
    loop: asyncio.AbstractEventLoop,
    on_interrupt: Callable | None = None,
) -> "WorkerResult":
    """Thread-pool recv loop. Dispatches InterruptMessages to main loop,
    returns on WorkerResult. Threading: this thread recv()s only, main loop send()s only.
    """
    import pickle

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError, ConnectionResetError):
            return WorkerResult(
                success=False, error="Worker process exited unexpectedly"
            )
        except pickle.UnpicklingError as e:
            return WorkerResult(success=False, error=f"Worker pipe corrupted: {e}")

        if isinstance(msg, InterruptMessage):
            if on_interrupt is not None:
                loop.call_soon_threadsafe(on_interrupt, msg)
        elif isinstance(msg, WorkerResult):
            return msg
        else:
            logger.warning("Unknown message from worker: %s", type(msg).__name__)


async def _cleanup_worker(
    proc: "mp.Process",
    timeout: float = 3.0,
) -> None:
    """Reap worker subprocess via is_alive() polling; SIGKILL on timeout.

    Avoids proc.join() and asyncio.to_thread to prevent thread-pool contention.
    """
    deadline = time.monotonic() + timeout
    while proc.is_alive():
        if time.monotonic() >= deadline:
            proc.kill()
            for _ in range(20):
                await asyncio.sleep(0.05)
                if not proc.is_alive():
                    return
            logger.warning("Worker pid=%s still alive after SIGKILL", proc.pid)
            return
        await asyncio.sleep(0.05)  # 50ms poll interval


async def _teardown_worker(
    proc: "mp.Process",
    parent_conn: "Connection",
    resume_task: "asyncio.Task | None",
    on_worker_done: Callable | None,
) -> None:
    """Clean up after a worker turn finishes (success, timeout, or cancel).

    Steps are ordered deliberately:
    1. Signal "worker done" — so the session immediately rejects any late
       POST /resume instead of silently dropping it into a dead queue.
    2. Cancel forward_resumes — it's no longer needed and holds a pipe ref.
    3. Reap subprocess — wait for exit, SIGKILL if stuck.
    4. Close pipe — release the fd.
    """
    # 1. Notify caller the worker is done (closes resume channel)
    if on_worker_done is not None:
        try:
            on_worker_done()
        except Exception:
            logger.exception("on_worker_done callback failed")

    # 2. Stop forwarding resumes to the (now-dead) worker
    if resume_task is not None:
        resume_task.cancel()

    # 3. Wait for subprocess to exit; kill after 3s
    await _cleanup_worker(proc, timeout=3.0)

    # 4. Close parent side of the pipe
    parent_conn.close()


async def _forward_resumes(
    queue: "asyncio.Queue",
    conn: "Connection",
) -> None:
    """Main-loop coroutine: reads ResumeMessages from queue, sends to worker via pipe.

    Threading: runs on the main loop — the ONLY thread allowed to conn.send().
    """
    while True:
        msg = await queue.get()
        try:
            conn.send(msg)
        except (BrokenPipeError, OSError) as e:
            logger.debug("forward_resumes: pipe closed: %s", e)
            return


class WorkerExecutor:
    """Executes agent turns in isolated worker processes."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        import_path: str | None = None,
    ):
        self.max_workers = max_workers or os.cpu_count() or DEFAULT_MAX_WORKERS
        self._semaphore = asyncio.Semaphore(self.max_workers)
        # Prefer forkserver: faster than spawn (reuses a warm fork with preloaded
        # imports) and safer than fork (no risk of copying locked mutexes from a
        # multithreaded parent). Fire-and-forget a background thread to warm the
        # daemon; if it hasn't finished by the first request, the request itself
        # triggers ensure_running() internally (and it's idempotent).
        if "forkserver" in mp.get_all_start_methods():
            self._mp_context = mp.get_context("forkserver")
            preload = [import_path.rsplit(":", 1)[0]] if import_path else []
            self._mp_context.set_forkserver_preload(preload)

            from multiprocessing.forkserver import ensure_running

            threading.Thread(target=ensure_running, daemon=True).start()
        else:
            self._mp_context = mp.get_context("spawn")

    @property
    def running_workers(self) -> int:
        return self.max_workers - self._semaphore._value

    async def submit_turn(
        self,
        import_path: str,
        message: ChatMessage,
        state: list,
        *,
        timeout: float = 0,
        session_id: str | None = None,
        on_interrupt: Callable | None = None,
        resume_queue: "asyncio.Queue | None" = None,
        on_worker_done: Callable | None = None,
    ) -> WorkerResult:
        """Run an agent turn in a fresh worker process.

        Always returns a WorkerResult; only raises CancelledError.

        Args:
            on_interrupt: Called on the main loop (via call_soon_threadsafe)
                each time the worker sends an InterruptMessage.
            resume_queue: If provided, a coroutine forwards ResumeMessages
                from this queue to the worker over the pipe.
            on_worker_done: Called once in finally, BEFORE cleanup starts.
                Use this to atomically close the resume channel so late
                POST /resume requests get a clean 404 instead of being
                silently dropped.
        """
        try:
            async with self._semaphore:
                parent_conn, child_conn = self._mp_context.Pipe(duplex=True)
                proc = self._mp_context.Process(  # type: ignore[attr-defined]
                    target=_worker_entry,
                    args=(child_conn, import_path, message, state, session_id),
                )
                proc.start()
                child_conn.close()

                loop = asyncio.get_running_loop()
                recv_future = loop.run_in_executor(
                    None, _run_worker, parent_conn, loop, on_interrupt
                )
                resume_task = (
                    loop.create_task(_forward_resumes(resume_queue, parent_conn))
                    if resume_queue is not None
                    else None
                )

                try:
                    if timeout > 0:
                        return await asyncio.wait_for(recv_future, timeout=timeout)
                    return await recv_future
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    proc.kill()
                    raise
                finally:
                    await _teardown_worker(
                        proc, parent_conn, resume_task, on_worker_done
                    )
        except asyncio.TimeoutError:
            return WorkerResult(success=False, error="Agent timed out")
        except asyncio.CancelledError:
            raise
        except Exception:
            return WorkerResult(success=False, error=traceback.format_exc())
