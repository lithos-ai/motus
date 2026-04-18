import asyncio
import contextvars
import inspect
import logging
import os
import threading
import traceback
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
)

from opentelemetry import trace

from ..tracing.agent_tracer import (
    ATTR_ERROR,
    ATTR_FUNC,
    ATTR_TASK_TYPE,
    export_trace,
    get_config,
    shutdown_tracing,
)
from .agent_future import AgentFuture
from .hooks import emit_on_task, hooks  # noqa: F401 — re-exported for tests
from .task_instance import (
    DEFAULT_POLICY,
    TaskCancelledError,
    TaskInstance,
    TaskKind,
    TaskPolicy,
    TaskStatus,
    _deep_unwrap,
    _scan_deps,
    stitch_creation_chain,
)
from .tracing import (
    ATTR_PARENT_TASK_ID,
    ATTR_TASK_ID,
    _current_task_id,
    get_stack,
)
from .types import TASK, AgentFutureId, AgentTaskId, TaskType

tracer = trace.get_tracer(__name__)

logging.basicConfig(
    level=getattr(
        logging, os.environ.get("MOTUS_LOG_LEVEL", "DEBUG").upper(), logging.DEBUG
    ),
    format="%(asctime)s [%(threadName)s][%(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("AgentRuntime")


class GraphScheduler:
    """
    Run in background Loop, the core scheduler.
    Maintain the state of the graph, and the dependencies between the nodes.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
    ):
        self.loop = loop
        self.task_counter = 1000
        self.af_counter = 1000000

        # Executor for sync @agent_task functions (run_in_executor).
        # Owned here so shutdown() can clean it up explicitly.
        # Note: we pass self._executor explicitly to run_in_executor() rather
        # than calling loop.set_default_executor() — in single-loop mode the
        # loop belongs to the caller and we must not clobber its default.
        self._executor = ThreadPoolExecutor(thread_name_prefix="motus-task")

        # All task state lives here.
        self.tasks: Dict[AgentTaskId, TaskInstance] = {}
        # Graph edges: which tasks depend on each future.
        self.future_to_dependents: Dict[AgentFutureId, List[AgentTaskId]] = {}
        # Future registry (needed for create_agent_future and lookups by id).
        self.agent_futures: Dict[AgentFutureId, AgentFuture] = {}

        # Tracing is configured from MOTUS_TRACING / MOTUS_COLLECTION_LEVEL env vars
        # by agent_tracer.setup_tracing(). Parent propagation rides the OTel
        # context via the per-task wrapper below, so no hook registrations
        # are needed here.

    def shutdown(self) -> None:
        """Cancel in-flight tasks, poison pending futures, shut down executor."""
        # 1. Cancel in-flight asyncio.Tasks.  In single-loop mode the
        #    caller's loop keeps running after shutdown, so without this
        #    wrapper() coroutines would continue executing on a dead
        #    scheduler.
        for task in list(self.tasks.values()):
            if task._asyncio_task is not None and not task._asyncio_task.done():
                task._asyncio_task.cancel()

        # 2. Poison futures → unblocks executor threads waiting on resolve()
        shutdown_err = RuntimeError("Motus runtime is shutting down")
        for af in list(self.agent_futures.values()):
            if not af.af_done():
                try:
                    af._set_exception(shutdown_err)
                except Exception:
                    pass  # race: another thread resolved it first

        # 3. Shut down the executor (threads are already unblocked)
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def _next_task_id(self) -> AgentTaskId:
        self.task_counter += 1
        return AgentTaskId(self.task_counter)

    def _next_agent_future_id(self) -> int:
        self.af_counter += 1
        return self.af_counter

    def create_agent_future(
        self, triggered_task=None, future: ConcurrentFuture = None
    ) -> AgentFuture:
        af = AgentFuture(self._next_agent_future_id(), triggered_task, future)
        self.agent_futures[af._agent_future_id] = af
        self.future_to_dependents[af._agent_future_id] = []
        return af

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def register_task(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        parent_stack: tuple[int, ...] | None = None,
        policy: TaskPolicy = DEFAULT_POLICY,
        creation_stack: traceback.StackSummary | None = None,
        num_returns: int = 1,
        task_type: TaskType = TASK,
    ) -> AgentFuture | tuple[AgentFuture, ...]:
        """
        Register a new task. Return immediately with placeholder AgentFuture(s).
        """
        if parent_stack is None:
            parent_stack = get_stack()

        task_id = self._next_task_id()
        logger.debug(f"[Scheduler] Registering task {task_id.id} for: {func.__name__}")
        try:
            func.task_id = task_id.id
        except AttributeError:
            pass  # builtins (operator.add, etc.) don't support attribute setting

        result_futures = [
            self.create_agent_future(triggered_task=func.__name__)
            for _ in range(num_returns)
        ]

        task = TaskInstance(
            task_id,
            TaskKind.COMPUTE,
            result_futures=result_futures,
            func=func,
            args=args,
            kwargs=kwargs,
            parent_stack=parent_stack or (),
            policy=policy,
            creation_stack=creation_stack,
            task_type=task_type,
        )
        self.tasks[task_id] = task

        logger.debug(
            f"[Scheduler] Task {task_id.id} registered. Pending: "
            f"{[af._agent_future_id.id for af in task._prerequisites.values()]}"
        )

        # Wire up dependency edges
        for prerequisite in task._prerequisites.values():
            self.future_to_dependents[prerequisite._agent_future_id].append(task_id)
            logger.debug(
                f"[Scheduler] Task {task_id.id} depends "
                f"on AgentFuture {prerequisite._agent_future_id.id}"
            )

        if task.is_ready():
            self._execute_task(task)

        logger.debug(
            f"[Scheduler] Task {task_id.id} registered with result_agent_future_id: "
            f"{[rf._agent_future_id.id for rf in result_futures]}"
        )
        if num_returns == 1:
            return result_futures[0]
        return tuple(result_futures)

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def _execute_task(self, task: TaskInstance):
        """Execute a ready task.

        RESOLVE tasks skip function execution — they just unwrap nested
        futures and resolve via the single entry point.
        """
        task.status = TaskStatus.RUNNING

        if task.kind == TaskKind.RESOLVE:
            logger.debug(
                f"[Scheduler] RESOLVE task {task.id.id} prerequisites met. Settling."
            )
            unwrapped = task.unwrap_value()
            self._resolve_task(task, results=unwrapped)
            return

        real_args, real_kwargs = task.unwrap_args()
        func = task.func

        logger.debug(f"[Scheduler] Executing task {task.id.id}: {task.name}")

        async def _invoke():
            """Run the user function (with optional timeout)."""
            if inspect.iscoroutinefunction(func):
                logger.debug(f"[Scheduler] Task {task.name} is a coroutine function")
                coro = func(*real_args, **real_kwargs)
            else:
                logger.debug(
                    "[Scheduler] Task is a normal function, offloading to the "
                    "background loop"
                )
                executor = self._executor
                if executor is None:
                    raise RuntimeError("Motus runtime is shutting down")
                p_func = partial(func, *real_args, **real_kwargs)
                ctx = contextvars.copy_context()
                coro = self.loop.run_in_executor(executor, ctx.run, p_func)

            if task.policy.timeout is not None:
                return await asyncio.wait_for(coro, timeout=task.policy.timeout)
            else:
                return await coro

        async def wrapper():
            result: Any = None
            had_error = False
            retrying = False
            caught_exception: Exception | None = None

            # Set hook_data eagerly so _cancel_task can emit hooks at any time.
            task.hook_data = {
                "func": func,
                "args": tuple(real_args),
                "kwargs": dict(real_kwargs),
                "task_id": task.id,
                "task_type": task.task_type,
            }

            logger.debug(
                f"enter {task.name}, with task_id: {getattr(func, 'task_id', None)}"
            )
            # Wrap the task body in an OTel span so children spawned from
            # inside ``_invoke()`` inherit the parent context naturally.
            # motus.task_id_int / motus.parent_task_id attributes let the
            # viewer rebuild the task tree independent of OTel span IDs.
            parent_task_id = task.parent_stack[-1] if task.parent_stack else -1
            span_ctx = tracer.start_as_current_span(
                task.name,
                attributes={
                    ATTR_FUNC: task.name,
                    ATTR_TASK_TYPE: task.task_type or "",
                    ATTR_TASK_ID: task.id.id,
                    ATTR_PARENT_TASK_ID: parent_task_id,
                },
            )
            try:
                with span_ctx as span:
                    tid_token = _current_task_id.set(task.id.id)
                    try:
                        await emit_on_task(
                            "task_start",
                            func,
                            tuple(real_args),
                            dict(real_kwargs),
                            task_id=task.id,
                            metadata={"parent_stack": task.parent_stack},
                            task_type=task.task_type,
                        )
                        result = await _invoke()
                    except BaseException as e:
                        # Record error on the active span before it closes so
                        # the trace carries the failure even if downstream
                        # async hooks never fire (e.g., during shutdown).
                        span.set_attribute(ATTR_ERROR, str(e))
                        span.set_status(trace.StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        _current_task_id.reset(tid_token)

            except asyncio.CancelledError:
                # asyncio.Task was cancelled (e.g. via _cancel_task).
                # If _cancel_task already handled everything, just bail out.
                if task.status == TaskStatus.CANCELLED:
                    return
                # Otherwise convert to TaskCancelledError so it flows
                # through ConcurrentFuture.
                had_error = True
                caught_exception = TaskCancelledError(task.name, task.id.id)

            except TaskCancelledError as e:
                # A prerequisite was cancelled — do NOT retry.
                had_error = True
                caught_exception = e

            except Exception as e:
                # Retry if policy allows
                if task.retry_count < task.policy.retries:
                    task.retry_count += 1
                    delay = task.policy.retry_delay
                    logger.warning(
                        f"[Scheduler] Task {task.name} failed (attempt "
                        f"{task.retry_count}/{task.policy.retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    retrying = True
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # Re-enter wrapper for the retry
                    task._asyncio_task = self.loop.create_task(wrapper())
                    return

                # Retries exhausted — stitch creation-site into unified chain
                stitch_creation_chain(e, task.creation_stack)

                logger.error(f"Task {task.name} failed: {e}")
                had_error = True
                caught_exception = e

            finally:
                if retrying:
                    logger.debug(f"exit {task.name}, with task_id: {task.id.id}")
                    return

                # If _cancel_task already resolved everything, nothing to do.
                if task.status == TaskStatus.CANCELLED:
                    task._asyncio_task = None
                    logger.debug(f"exit {task.name}, with task_id: {task.id.id}")
                    return

                if had_error:
                    # Propagate error msg
                    self._resolve_task(task, error=caught_exception)
                elif task.num_returns == 1:
                    self._settle_or_defer(task, [result])
                elif (
                    not isinstance(result, (tuple, list))
                    or len(result) != task.num_returns
                ):
                    # Multiple return number mismatch
                    length = len(result) if isinstance(result, (tuple, list)) else "N/A"
                    err = ValueError(
                        f"Task {task.name} declared num_returns="
                        f"{task.num_returns} but returned "
                        f"{type(result).__name__} of length {length}"
                    )
                    self._resolve_task(task, error=err)
                else:
                    self._settle_or_defer(task, list(result))

                task._asyncio_task = None
                logger.debug(f"exit {task.name}, with task_id: {task.id.id}")

        logger.debug(
            f"[Scheduler] Creating wrapper task for {task.name} ({task.id.id})"
        )
        task._asyncio_task = self.loop.create_task(wrapper())
        logger.debug(f"[Scheduler] Wrapper task for {task.name} ({task.id.id}) created")

    # ------------------------------------------------------------------
    # Result settlement & dependency propagation
    # ------------------------------------------------------------------

    def _resolve_task(
        self,
        task: TaskInstance,
        results: list[Any] | None = None,
        error: Exception | None = None,
    ):
        """Single entry point for resolving a task's result futures.

        Every task resolution — immediate success, deferred (RESOLVE)
        success, function error, propagated error, or cancellation —
        goes through here.  This is THE only place that sets
        results/exceptions on result futures and emits lifecycle hooks.
        """
        if error is not None:
            for rf in task.result_futures:
                if not rf.af_done():
                    rf._set_exception(error)
        else:
            for rf, val in zip(task.result_futures, results):
                if not rf.af_done():
                    rf._set_result(val)

        # Emit hook — hook_data is set at the start of wrapper(), so it is
        # available for tasks that reached RUNNING state, including those
        # cancelled mid-execution.  Tasks cancelled while still PENDING
        # (before wrapper() runs) will have hook_data=None and emit no hook.
        # Errors are recorded synchronously on the task's OTel span inside
        # wrapper() itself, so we don't need a separate state hatch here.
        if task.hook_data is not None:
            info = task.hook_data
            if error is not None:
                # Distinguish cancellation from regular errors.
                hook_type = (
                    "task_cancelled"
                    if isinstance(error, TaskCancelledError)
                    else "task_error"
                )
                self.loop.create_task(
                    emit_on_task(
                        hook_type,
                        info["func"],
                        info["args"],
                        info["kwargs"],
                        error=error,
                        task_id=info["task_id"],
                        task_type=info.get("task_type"),
                    )
                )
            else:
                resolved = results[0] if len(results) == 1 else tuple(results)
                self.loop.create_task(
                    emit_on_task(
                        "task_end",
                        info["func"],
                        info["args"],
                        info["kwargs"],
                        result=resolved,
                        task_id=info["task_id"],
                        task_type=info.get("task_type"),
                    )
                )

        self._terminate_task(task, error)

    def _settle_or_defer(self, task: TaskInstance, results: list[Any]):
        """Check if results can be settled immediately, or defer via RESOLVE.

        If results contain no unresolved AgentFutures, resolves the task
        immediately via _resolve_task. Otherwise, creates a RESOLVE task
        that waits for the nested futures and will call _resolve_task later.
        """
        nested_deps = _scan_deps(results)

        if not nested_deps:
            # No nested deps — unwrap and resolve immediately
            unwrapped = _deep_unwrap(results)
            self._resolve_task(task, results=unwrapped)
            return

        # Has unresolved nested futures — create a RESOLVE task
        logger.debug(
            f"[Scheduler] Result has {len(nested_deps)} unresolved nested futures. "
            f"Creating RESOLVE task."
        )
        resolve_id = self._next_task_id()
        resolve_task = TaskInstance(
            resolve_id,
            TaskKind.RESOLVE,
            task.result_futures,
            args=(results,),
            hook_data=task.hook_data,
        )
        self.tasks[resolve_id] = resolve_task

        for dep in resolve_task._prerequisites.values():
            self.future_to_dependents[dep._agent_future_id].append(resolve_id)

        # Drop the original task — RESOLVE takes over its result futures
        task.status = TaskStatus.COMPLETED
        self.tasks.pop(task.id, None)

    def _propagate_to_dependents(
        self,
        futures: list[AgentFuture],
        exception: Exception | None = None,
    ):
        """Notify all tasks that depend on the given futures."""
        for future in futures:
            af_id = future._agent_future_id
            dependents = self.future_to_dependents.pop(af_id, [])

            logger.debug(
                f"[Scheduler] AgentFuture {future!r} done. Triggering "
                f"{len(dependents)} dependents: "
                f"{[str(tid.id) for tid in dependents]}."
            )

            # Detect exception from the future itself (if not passed explicitly)
            fut_exception = exception
            if fut_exception is None and future.af_done():
                try:
                    future._future.result()
                except Exception as e:
                    fut_exception = e

            for task_id in dependents:
                task = self.tasks.get(task_id)
                if task is None:
                    logger.debug(
                        f"[Scheduler] Dependent task {task_id} already removed "
                        f"(likely because a dependency task failed)"
                    )
                    continue

                if fut_exception:
                    logger.debug(
                        f"[Scheduler] Propagating exception to task {task_id.id}"
                    )
                    self._resolve_task(task, error=fut_exception)
                else:
                    # Normal success: resolve one dependency
                    is_ready = task.dependency_resolved(future)
                    if is_ready:
                        self._execute_task(task)
                    else:
                        logger.debug(
                            f"[Scheduler] Task {task_id.id} still has "
                            f"{task.prerequisite_count} prerequisites to fulfill."
                        )

        # Future entries no longer needed after all dependents processed
        for future in futures:
            self.agent_futures.pop(future._agent_future_id, None)

    def _terminate_task(
        self,
        task: TaskInstance,
        exception: Exception | None = None,
    ):
        """Set terminal status, propagate to dependents, and remove the task."""
        if exception is not None:
            if isinstance(exception, TaskCancelledError):
                task.status = TaskStatus.CANCELLED
            else:
                task.status = TaskStatus.FAILED
        else:
            task.status = TaskStatus.COMPLETED
        self._propagate_to_dependents(task.result_futures, exception=exception)
        self.tasks.pop(task.id, None)

    # ------------------------------------------------------------------
    # Task cancellation
    # ------------------------------------------------------------------

    def _cancel_future(self, future: AgentFuture) -> None:
        """Cancel a future and the task that produces it.

        Must be called on the event-loop thread (use
        ``loop.call_soon_threadsafe`` from other threads).
        """
        if future.af_done():
            return

        task = self._find_task_for_future(future)
        if task is None:
            # Orphan future (task already completed/removed) — just poison it.
            try:
                future._set_exception(TaskCancelledError("unknown"))
            except Exception:
                pass
            return

        self._cancel_task(task)

    def _cancel_task(self, task: TaskInstance) -> None:
        """Cancel a task: stop execution, poison futures, cascade to dependents."""
        if task.status in (
            TaskStatus.CANCELLED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        ):
            return

        logger.debug(f"[Scheduler] Cancelling task {task.id.id}: {task.name}")

        cancel_err = TaskCancelledError(task.name, task.id.id)

        # If RUNNING and has an asyncio.Task, request its cancellation.
        if task.status == TaskStatus.RUNNING and task._asyncio_task is not None:
            task._asyncio_task.cancel()

        # Resolve immediately so that any resolve() callers unblock right away
        # (especially important for sync executor tasks whose threads
        # cannot be interrupted).  _resolve_task detects TaskCancelledError
        # and emits the correct hook / sets CANCELLED status.
        self._resolve_task(task, error=cancel_err)

    def _find_task_for_future(self, future: AgentFuture) -> TaskInstance | None:
        """Find the task that produces a given future."""
        af_id = future._agent_future_id
        for task in self.tasks.values():
            for rf in task.result_futures:
                if rf._agent_future_id == af_id:
                    return task
        return None


class AgentRuntime:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        """Initialize AgentRuntime.

        Args:
            loop: If provided, reuse this event loop (single-loop mode).
                  If *None*, create a dedicated background thread with its
                  own event loop (dual-loop mode, the default for sync callers).

        Tracing is configured via environment variables:
        - MOTUS_COLLECTION_LEVEL: disabled/basic/standard/detailed (default: standard)
        - MOTUS_TRACING=1: Enable detailed tracing with export (legacy)
        - MOTUS_TRACING_EXPORT=1: Enable trace export to files
        - MOTUS_TRACING_ONLINE=1: Enable live tracing with auto-refresh viewer
        """
        if loop is not None:
            # Single-loop mode: reuse caller's loop, no background thread.
            if loop.is_closed():
                raise ValueError("Cannot use a closed event loop")
            self._loop = loop
            self._thread = threading.current_thread()
            self._owns_loop = False
        else:
            # Dual-loop mode: create dedicated loop + thread.
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True

            def run_loop():
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()

            self._thread = threading.Thread(
                target=run_loop, daemon=True, name="AgentEngine"
            )
            self._thread.start()

            # LIFO: our shutdown runs before _python_exit (ThreadPoolExecutor
            # import at module level ensures _python_exit is registered first).
            threading._register_atexit(self.shutdown)

        self._scheduler = GraphScheduler(self._loop)
        self._shutdown = False

    @property
    def scheduler(self):
        return self._scheduler

    def submit_task_registration(
        self,
        func,
        args,
        kwargs,
        parent_stack: tuple[int, ...] | None = None,
        policy: TaskPolicy = DEFAULT_POLICY,
        num_returns: int = 1,
        creation_stack: traceback.StackSummary | None = None,
        task_type: TaskType = TASK,
    ) -> AgentFuture | tuple[AgentFuture, ...]:
        """
        Submit a task registration request to the background loop (different thread).
        Return when the registration is completed and the AgentFuture is returned.
        """
        func_name = func.__name__
        logger.debug(f"Submitting task registration: {func_name}")

        if threading.current_thread() is self._thread:
            logger.debug(
                f"Submitting task registration: {func_name} from the same thread"
            )
            return self._scheduler.register_task(
                func,
                args,
                kwargs,
                parent_stack,
                policy,
                creation_stack,
                num_returns=num_returns,
                task_type=task_type,
            )
        else:
            logger.debug(
                f"Submitting task registration: {func_name} from a different thread"
            )
            coroutine_future = asyncio.run_coroutine_threadsafe(
                self._register_wrapper(
                    func,
                    args,
                    kwargs,
                    parent_stack,
                    policy,
                    creation_stack,
                    num_returns=num_returns,
                    task_type=task_type,
                ),
                self._loop,
            )
            logger.debug(f"Task registration submitted: {func_name}")
            register_result = coroutine_future.result()
            logger.debug(
                f"Task registration of {func_name} completed with register_result:"
                f" {register_result!r}"
            )
            return register_result

    def cancel_future(self, future: AgentFuture) -> None:
        """Cancel a future and its owning task (thread-safe).

        When called from the runtime's own event-loop thread the
        cancellation takes effect immediately; from other threads it is
        scheduled via ``call_soon_threadsafe``.
        """
        if threading.current_thread() is self._thread:
            self._scheduler._cancel_future(future)
        else:
            self._loop.call_soon_threadsafe(self._scheduler._cancel_future, future)

    def export_trace(self):
        export_trace()

    def shutdown(self):
        """Shut down the runtime: export traces, poison futures, stop loop."""
        if self._shutdown:
            return
        self._shutdown = True
        if get_config().export_enabled:
            self.export_trace()
        shutdown_tracing()
        self._scheduler.shutdown()
        if self._owns_loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def __del__(self):
        try:
            # In single-loop mode the caller owns the loop; touching
            # scheduler state from a GC thread would be a data race.
            if self._owns_loop:
                self.shutdown()
        except Exception:
            pass

    async def _register_wrapper(
        self,
        func,
        args,
        kwargs,
        parent_stack: tuple[int, ...] | None = None,
        policy: TaskPolicy = DEFAULT_POLICY,
        creation_stack: traceback.StackSummary | None = None,
        num_returns: int = 1,
        task_type: TaskType = TASK,
    ) -> AgentFuture | tuple[AgentFuture, ...]:
        # Run in background loop.
        return self._scheduler.register_task(
            func,
            args,
            kwargs,
            parent_stack,
            policy,
            creation_stack,
            num_returns=num_returns,
            task_type=task_type,
        )


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

_runtime: AgentRuntime | None = None
_lock = threading.Lock()


def init(loop: asyncio.AbstractEventLoop | None = None) -> AgentRuntime:
    """Initialize the motus runtime.

    Call this before using ``@agent_task`` or other runtime features.
    If not called explicitly, the runtime will auto-initialize with
    defaults on first use.

    Args:
        loop: If provided, reuse this event loop (single-loop mode).
              If *None*, create a dedicated background thread with its
              own event loop (dual-loop mode).

    Tracing is configured via environment variables:
        - MOTUS_COLLECTION_LEVEL: disabled/basic/standard/detailed
        - MOTUS_TRACING=1: Enable detailed tracing with export
        - MOTUS_TRACING_EXPORT=1: Enable trace file export
        - MOTUS_TRACING_ONLINE=1: Enable live tracing with auto-refresh viewer

    Returns:
        The initialized :class:`AgentRuntime` instance.

    Raises:
        RuntimeError: If the runtime is already initialized.
    """
    global _runtime
    with _lock:
        if _runtime is not None and not _runtime._shutdown:
            raise RuntimeError(
                "motus runtime is already initialized. "
                "Call motus.shutdown() before re-initializing."
            )
        _runtime = AgentRuntime(loop=loop)
        return _runtime


def shutdown() -> None:
    """Shut down the motus runtime.

    Exports traces (if enabled) and stops the background event loop.
    After shutdown, :func:`init` can be called again to create a fresh
    runtime.
    """
    global _runtime
    with _lock:
        if _runtime is None:
            return
        rt = _runtime
        _runtime = None
    rt.shutdown()


def is_initialized() -> bool:
    """Return *True* if the motus runtime is currently active."""
    return _runtime is not None and not _runtime._shutdown


def get_runtime() -> AgentRuntime:
    """Return the current runtime, auto-initializing if needed.

    Internal helper used by :func:`agent_task` and other runtime
    components.  Provides backward compatibility: code that never calls
    :func:`init` will get a runtime with default settings on first use.

    Auto-detection: if called from within a running event loop, the
    runtime is created in single-loop mode (reusing the caller's loop).
    Otherwise, a dedicated background thread is created (dual-loop mode).
    """
    global _runtime
    rt = _runtime
    if rt is not None and not rt._shutdown:
        return rt
    with _lock:
        # Double-check inside lock
        rt = _runtime
        if rt is not None and not rt._shutdown:
            return rt
        # Auto-detect: if already in async context, reuse that loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        logger.info("motus runtime auto-initializing with default settings")
        _runtime = AgentRuntime(loop=loop)
        return _runtime
