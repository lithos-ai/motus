from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from motus.models import ChatCompletion, ChatMessage

from .tracing import AgentTrace, TraceTurn, load_trace

SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]


@dataclass
class ReplayTurnResult:
    trace_id: str
    agent_instance_id: str
    agent_class_id: str
    turn_index: int
    output_tokens_requested: int
    output_tokens_observed: int | None
    planned_tools: list[dict[str, Any]]
    duration_ms_requested: float | None
    started_at: float
    ended_at: float
    finish_reason: str
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceReplayResult:
    trace_id: str
    agent_instance_id: str
    agent_class_id: str
    turns_completed: int
    turn_results: list[ReplayTurnResult] = field(default_factory=list)
    arrival_offset_seconds: float | None = None
    scheduled_arrival_at: float | None = None
    actual_launch_at: float | None = None
    completed_at: float | None = None
    job_completion_latency_seconds: float | None = None
    launch_delay_seconds: float | None = None


@dataclass
class ReplaySummary:
    total_traces: int
    completed_traces: int
    failed_traces: int
    elapsed_seconds: float
    results: list[TraceReplayResult] = field(default_factory=list)
    errors: list[ReplayError] = field(default_factory=list)
    arrival_mode: str = "concurrency"
    arrival_rate_jps: float | None = None
    arrival_seed: int | None = None
    scheduled_span_seconds: float | None = None


@dataclass
class ReplayError:
    error: str
    trace_id: str | None = None
    trace_path: str | None = None


def build_agent_replay(
    trace: AgentTrace,
    turn: TraceTurn,
    *,
    include_tool_duration: bool = False,
) -> dict[str, Any]:
    planned_tools = []
    for tool in turn.tools:
        planned_tool = {
            "name": tool.name,
            "args": tool.args,
        }
        if include_tool_duration:
            planned_tool["duration_ms"] = tool.duration_ms
        planned_tools.append(planned_tool)

    return {
        "schema_version": 1,
        "trace_id": trace.trace_id,
        "turn_index": turn.turn_index,
        "planned_tools": planned_tools,
    }


def build_sampling_kwargs(output_tokens: int) -> dict[str, Any]:
    return {
        "max_tokens": output_tokens,
        "temperature": 0,
        "extra_body": {
            "min_tokens": output_tokens,
            "ignore_eos": True,
        },
    }


class TraceReplayRunner:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        concurrency: int,
        sleep: SleepFn = asyncio.sleep,
        clock: ClockFn = time.time,
        arrival_mode: str = "concurrency",
        arrival_rate_jps: float | None = None,
        arrival_seed: int | None = None,
        include_tool_duration_metadata: bool = False,
    ):
        self.client = client
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.sleep = sleep
        self.clock = clock
        if arrival_mode not in {"concurrency", "poisson"}:
            raise ValueError("arrival_mode must be 'concurrency' or 'poisson'")
        self.arrival_mode = arrival_mode
        self.arrival_rate_jps = (
            float(arrival_rate_jps) if arrival_rate_jps is not None else None
        )
        if self.arrival_mode == "poisson" and (
            self.arrival_rate_jps is None or self.arrival_rate_jps <= 0
        ):
            raise ValueError("arrival_rate_jps must be positive for poisson arrival mode")
        self.arrival_seed = arrival_seed
        self.include_tool_duration_metadata = include_tool_duration_metadata

    async def run_trace(
        self,
        trace: AgentTrace,
        trace_path: str | Path | None = None,
    ) -> TraceReplayResult:
        messages = self._initial_messages(trace)
        turn_results: list[ReplayTurnResult] = []

        for turn in trace.turns:
            messages.append(self._input_delta_message(turn))
            request_kwargs = build_sampling_kwargs(turn.output_tokens)
            extra_body = dict(request_kwargs.pop("extra_body"))
            extra_body.update(self._extra_body(trace, turn))
            started_at = time.time()
            completion = await self._create_completion(
                model=self.model,
                messages=list(messages),
                extra_body=extra_body,
                **request_kwargs,
            )
            ended_at = time.time()
            turn_results.append(
                self._turn_result(trace, turn, completion, started_at, ended_at)
            )
            messages.append(ChatMessage.assistant_message(completion.content or ""))

            for tool in turn.tools:
                if tool.subagent_trace_file:
                    child_path = self._resolve_child_trace(trace_path, tool.subagent_trace_file)
                    await self.run_trace(load_trace(child_path), trace_path=child_path)
                elif tool.duration_ms:
                    await self.sleep(float(tool.duration_ms) / 1000.0)

        return TraceReplayResult(
            trace_id=trace.trace_id,
            agent_instance_id=trace.agent.agent_instance_id,
            agent_class_id=trace.agent.agent_class_id,
            turns_completed=len(turn_results),
            turn_results=turn_results,
        )

    async def run_many(
        self,
        traces: Iterable[AgentTrace | str | Path],
    ) -> ReplaySummary:
        items = list(traces)
        started_at = self.clock()
        arrival_offsets = self._arrival_offsets(len(items))
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(
            item: AgentTrace | str | Path,
            *,
            scheduled_arrival_at: float,
            arrival_offset_seconds: float,
        ):
            delay = scheduled_arrival_at - self.clock()
            if delay > 0:
                await self.sleep(delay)
            async with semaphore:
                actual_launch_at = self.clock()
                if isinstance(item, AgentTrace):
                    result = await self.run_trace(item)
                else:
                    path = Path(item)
                    result = await self.run_trace(load_trace(path), trace_path=path)
                completed_at = self.clock()
                return self._with_job_timing(
                    result,
                    arrival_offset_seconds=arrival_offset_seconds,
                    scheduled_arrival_at=scheduled_arrival_at,
                    actual_launch_at=actual_launch_at,
                    completed_at=completed_at,
                )

        async def run_one_with_error(
            item: AgentTrace | str | Path,
            *,
            scheduled_arrival_at: float,
            arrival_offset_seconds: float,
        ):
            trace_id: str | None = item.trace_id if isinstance(item, AgentTrace) else None
            trace_path: str | None = None if isinstance(item, AgentTrace) else str(Path(item))
            try:
                return await run_one(
                    item,
                    scheduled_arrival_at=scheduled_arrival_at,
                    arrival_offset_seconds=arrival_offset_seconds,
                )
            except Exception as exc:
                if trace_id is None and trace_path is not None:
                    try:
                        trace_id = load_trace(trace_path).trace_id
                    except Exception:
                        pass
                return ReplayError(error=str(exc), trace_id=trace_id, trace_path=trace_path)

        tasks = [
            asyncio.create_task(
                run_one_with_error(
                    item,
                    scheduled_arrival_at=started_at + arrival_offset,
                    arrival_offset_seconds=arrival_offset,
                )
            )
            for item, arrival_offset in zip(items, arrival_offsets)
        ]
        results: list[TraceReplayResult] = []
        errors: list[ReplayError] = []
        for task in tasks:
            result = await task
            if isinstance(result, ReplayError):
                errors.append(result)
            else:
                results.append(result)

        return ReplaySummary(
            total_traces=len(tasks),
            completed_traces=len(results),
            failed_traces=len(errors),
            elapsed_seconds=self.clock() - started_at,
            results=results,
            errors=errors,
            arrival_mode=self.arrival_mode,
            arrival_rate_jps=self.arrival_rate_jps,
            arrival_seed=self.arrival_seed,
            scheduled_span_seconds=arrival_offsets[-1] if arrival_offsets else 0.0,
        )

    def _arrival_offsets(self, count: int) -> list[float]:
        if count <= 0:
            return []
        if self.arrival_mode == "concurrency":
            return [0.0] * count

        assert self.arrival_rate_jps is not None
        rng = random.Random(self.arrival_seed)
        offsets = [0.0]
        for _ in range(1, count):
            offsets.append(offsets[-1] + rng.expovariate(self.arrival_rate_jps))
        return offsets

    def _with_job_timing(
        self,
        result: TraceReplayResult,
        *,
        arrival_offset_seconds: float,
        scheduled_arrival_at: float,
        actual_launch_at: float,
        completed_at: float,
    ) -> TraceReplayResult:
        result.arrival_offset_seconds = arrival_offset_seconds
        result.scheduled_arrival_at = scheduled_arrival_at
        result.actual_launch_at = actual_launch_at
        result.completed_at = completed_at
        result.job_completion_latency_seconds = completed_at - scheduled_arrival_at
        result.launch_delay_seconds = actual_launch_at - scheduled_arrival_at
        return result

    def _initial_messages(self, trace: AgentTrace) -> list[ChatMessage]:
        if trace.system_prompt.text:
            return [ChatMessage.system_message(trace.system_prompt.text)]
        return []

    def _input_delta_message(self, turn: TraceTurn) -> ChatMessage:
        return ChatMessage.user_message(turn.input_delta.text or "")

    def _extra_body(self, trace: AgentTrace, turn: TraceTurn) -> dict[str, Any]:
        return {
            "agent_instance_id": trace.agent.agent_instance_id,
            "agent_class_id": trace.agent.agent_class_id,
            "is_last_step": turn.is_terminal,
            "agent_replay": build_agent_replay(
                trace,
                turn,
                include_tool_duration=self.include_tool_duration_metadata,
            ),
        }

    async def _create_completion(self, **kwargs: Any) -> ChatCompletion:
        create = getattr(self.client, "create_non_streaming", self.client.create)
        return await create(**kwargs)

    def _turn_result(
        self,
        trace: AgentTrace,
        turn: TraceTurn,
        completion: ChatCompletion,
        started_at: float,
        ended_at: float,
    ) -> ReplayTurnResult:
        usage = completion.usage or {}
        observed = usage.get("completion_tokens")
        durations = [tool.duration_ms for tool in turn.tools]
        known_durations = [duration for duration in durations if duration is not None]
        duration_ms_requested = (
            float(sum(known_durations))
            if durations and len(known_durations) == len(durations)
            else None
        )
        return ReplayTurnResult(
            trace_id=trace.trace_id,
            agent_instance_id=trace.agent.agent_instance_id,
            agent_class_id=trace.agent.agent_class_id,
            turn_index=turn.turn_index,
            output_tokens_requested=turn.output_tokens,
            output_tokens_observed=observed if isinstance(observed, int) else None,
            planned_tools=build_agent_replay(
                trace,
                turn,
                include_tool_duration=True,
            )["planned_tools"],
            duration_ms_requested=duration_ms_requested,
            started_at=started_at,
            ended_at=ended_at,
            finish_reason=completion.finish_reason,
            usage=usage,
        )

    def _resolve_child_trace(
        self,
        parent_trace_path: str | Path | None,
        subagent_trace_file: str,
    ) -> Path:
        child = Path(subagent_trace_file)
        if child.is_absolute():
            return child
        if parent_trace_path is None:
            return child
        return Path(parent_trace_path).parent / child
