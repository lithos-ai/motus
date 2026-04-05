"""Process-per-request worker execution for serve.

Each turn spawns a fresh subprocess via multiprocessing.Process.
A semaphore limits concurrency to max_workers.
"""

import asyncio
import importlib
import inspect
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any

from motus.models import ChatMessage

DEFAULT_MAX_WORKERS = 4


@dataclass
class WorkerResult:
    """Result wrapper to distinguish success from failure without raising."""

    success: bool
    value: Any = None
    error: str | None = None


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
        from agents import Agent as _OAIAgent

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


def _worker_entry(conn, import_path, message, state):
    """Subprocess entry point that runs an agent and sends the result over pipe."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        from motus.serve.protocol import ServableAgent

        agent_or_fn = _resolve_import_path(import_path)

        if isinstance(agent_or_fn, ServableAgent):
            # ServableAgent protocol (AgentBase, Google ADK, Anthropic ToolRunner, etc.)
            result = asyncio.run(agent_or_fn.run_turn(message, state))
            response, new_state = _validate_result(result)
        elif _is_openai_agent(agent_or_fn):
            # OpenAI Agents SDK mode: dataclass, needs adapter
            adapted = _adapt_openai_agent(agent_or_fn)
            result = asyncio.run(adapted(message, state))
            response, new_state = _validate_result(result)
        elif callable(agent_or_fn):
            # Custom function mode: fn(message, state) -> (response, state)
            if inspect.iscoroutinefunction(agent_or_fn):
                result = asyncio.run(agent_or_fn(message, state))
            else:
                result = agent_or_fn(message, state)
            response, new_state = _validate_result(result)
        else:
            raise TypeError(
                f"'{import_path}' resolved to {type(agent_or_fn).__name__}, "
                f"expected a ServableAgent, OpenAI Agent, or callable"
            )

        conn.send(WorkerResult(success=True, value=(response, new_state)))

    except Exception:
        conn.send(WorkerResult(success=False, error=traceback.format_exc()))

    finally:
        conn.close()
        try:
            from motus.runtime.agent_runtime import shutdown as _rt_shutdown

            _rt_shutdown()  # Shut down motus runtime to release executor threads
        except Exception:
            pass


def _run_worker(conn, proc) -> WorkerResult:
    """Receive result, join process, close pipe. Runs in thread pool."""
    try:
        try:
            result = conn.recv()
        except EOFError:
            result = WorkerResult(
                success=False, error="Worker process exited unexpectedly"
            )
        return result
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        conn.close()


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
        # multithreaded parent).
        if "forkserver" in mp.get_all_start_methods():
            self._mp_context = mp.get_context("forkserver")
            preload = [import_path.rsplit(":", 1)[0]] if import_path else []
            self._mp_context.set_forkserver_preload(preload)
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
    ) -> WorkerResult:
        """Run an agent turn in a fresh worker process.

        Waits if max_workers processes are already running.  Always returns a
        ``WorkerResult``; only raises ``CancelledError`` (task cancellation).
        """
        try:
            async with self._semaphore:
                parent_conn, child_conn = self._mp_context.Pipe(duplex=False)
                proc = self._mp_context.Process(  # type: ignore[attr-defined]
                    target=_worker_entry,
                    args=(child_conn, import_path, message, state),
                )
                proc.start()
                child_conn.close()
                try:
                    loop = asyncio.get_running_loop()
                    coro = loop.run_in_executor(None, _run_worker, parent_conn, proc)
                    if timeout > 0:
                        return await asyncio.wait_for(coro, timeout=timeout)
                    return await coro
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    proc.kill()  # _run_worker thread handles join and close
                    raise
        except asyncio.TimeoutError:
            return WorkerResult(success=False, error="Agent timed out")
        except Exception:
            return WorkerResult(success=False, error=traceback.format_exc())
