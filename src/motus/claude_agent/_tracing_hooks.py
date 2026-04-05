"""Tracing hooks for Claude Agent SDK -> motus TraceManager bridge.

Hooks (PreToolUse/PostToolUse/SubagentStart/SubagentStop) provide real-time
tool and agent lifecycle events. Consumer messages (AssistantMessage/UserMessage/
ResultMessage) provide model call content and session cost/usage.

With the AsyncIterable streaming fix in __init__.py, both hooks and consumer
messages arrive interleaved in real-time, enabling accurate chronological spans.

Subagent nesting: tool/model spans from a subagent are children of its
agent_call span, which itself is a child of the parent agent's root or
enclosing agent_call span.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple

from motus.runtime.types import MODEL_CALL, TOOL_CALL

if TYPE_CHECKING:
    from motus.runtime.tracing import TraceManager

logger = logging.getLogger("AgentTracer")

_TRUNCATE_LIMIT = 4000


# ── Helpers ──


def _now_us() -> int:
    """Current wall-clock time in microseconds (consistent across all spans)."""
    return time.time_ns() // 1000


class _AgentContext(NamedTuple):
    """Active subagent on the agent stack."""

    agent_id: str
    task_id: int  # motus task_id for the agent_call span
    tuid: str  # tool_use_id of the Agent tool that spawned this subagent
    start_us: int  # start time in microseconds


def _truncate(value: Any, max_len: int = _TRUNCATE_LIMIT) -> Any:
    """Truncate large values to prevent huge SSE payloads.

    Preserves original type for small values. For large dict/list, serializes
    to JSON before truncating (avoids lossy Python repr).
    """
    if isinstance(value, str):
        if len(value) > max_len:
            return value[:max_len] + "...[truncated]"
        return value
    if isinstance(value, (dict, list)):
        try:
            s = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "...[truncated]"
        return value
    return value


def _parse_content_blocks(
    blocks: Any,
) -> tuple[list[str], list[str], list[dict]]:
    """Parse SDK content blocks into (thinking_parts, text_parts, tool_calls).

    Shared by _msg_to_dict and _observe_assistant to avoid duplicating
    the block-type dispatch logic.
    """
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in blocks:
        bt = type(block).__name__
        if bt == "ThinkingBlock":
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif bt == "TextBlock":
            text_parts.append(getattr(block, "text", ""))
        elif bt == "ToolUseBlock":
            args = getattr(block, "input", None)
            tool_calls.append(
                {
                    "function": {
                        "name": getattr(block, "name", None),
                        "arguments": json.dumps(args)
                        if isinstance(args, (dict, list))
                        else str(args or ""),
                    },
                    "id": getattr(block, "id", None),
                }
            )
    return thinking_parts, text_parts, tool_calls


# ── TracingHookState ──


class TracingHookState:
    """Per-session mutable state shared between hooks and consumer observer."""

    def __init__(self) -> None:
        self.root_task_id: int | None = None

        # tool_use_id -> (task_id, start_us, tool_name, tool_input)
        self.pending_tools: dict[str, tuple[int, int, str, dict]] = {}

        # Subagent nesting
        self._agent_stack: list[_AgentContext] = []
        self._tuid_to_agent: dict[
            str, int
        ] = {}  # Agent tool_use_id -> agent_call task_id
        self._pending_agent_tuids: deque[str] = (
            deque()
        )  # PreToolUse(Agent) → SubagentStart
        self._fallback_counter: int = 0  # for unique synthetic tuids

        # Consumer-observed tool ownership: tool_use_id → parent_tool_use_id
        # Populated from AssistantMessages BEFORE hooks fire (SDK delivers
        # messages before dispatching tools). Used by hooks for correct parent
        # resolution — especially for parallel subagents that break stack ordering.
        self._tool_owner: dict[str, str | None] = {}

        # Track which agent contexts have had at least one AssistantMessage observed.
        # Agents without any (e.g. simple one-turn, no-tool subagents) get a
        # synthetic model_call created from PostToolUse's tool_response.
        self._agent_has_assistant: set[str] = set()  # set of parent_tool_use_ids

        # Per-agent conversation history (key = parent_tool_use_id, None = root)
        self._conversations: dict[str | None, list[dict]] = {None: []}
        now_us = _now_us()
        self._last_msg_us: dict[str | None, int] = {None: now_us}

    # ── Parent resolution ──

    def _resolve_parent_for_ptuid(self, ptuid: str | None) -> int | None:
        """Resolve parent task_id from a parent_tool_use_id.

        ptuid is None → root agent context.
        ptuid is a string → subagent context (look up in _tuid_to_agent).
        """
        if ptuid is None:
            return self.root_task_id
        return self._tuid_to_agent.get(ptuid, self.root_task_id)

    def _resolve_parent_for_tool(self, tool_use_id: str) -> int | None:
        """Resolve parent task_id for a tool via consumer-observed ownership.

        Uses _tool_owner (populated from AssistantMessages) to determine which
        agent context produced this tool_call. Falls back to stack-based
        resolution if the consumer hasn't observed the message yet.
        """
        if tool_use_id in self._tool_owner:
            return self._resolve_parent_for_ptuid(self._tool_owner[tool_use_id])
        # Fallback: consumer message hasn't been processed yet
        if self._agent_stack:
            return self._agent_stack[-1].task_id
        return self.root_task_id

    # ── Message conversion ──

    @staticmethod
    def _msg_to_dict(message: Any) -> dict | None:
        """Convert SDK message to a conversation-history dict."""
        btype = type(message).__name__
        if btype == "AssistantMessage":
            thinking_parts, text_parts, tool_calls = _parse_content_blocks(
                getattr(message, "content", [])
            )
            result: dict[str, Any] = {"role": "assistant"}
            text = "\n".join(thinking_parts + text_parts)
            if text:
                result["content"] = text
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        if btype == "UserMessage":
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return {"role": "user", "content": content}
            parts: list[str] = []
            for block in content:
                bt = type(block).__name__
                if bt == "ToolResultBlock":
                    raw = getattr(block, "content", None)
                    tid = getattr(block, "tool_use_id", None)
                    text = str(raw)[:2000] if raw else ""
                    prefix = f"[tool_result {tid}] " if tid else "[tool_result] "
                    parts.append(prefix + text)
                elif bt == "TextBlock":
                    parts.append(getattr(block, "text", ""))
            return {"role": "user", "content": "\n".join(parts)}

        return None

    # ── Consumer message observation ──

    def observe_message(self, message: Any, tm: TraceManager) -> None:
        """Observe a yielded message from the SDK stream."""
        try:
            from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage
        except ImportError:
            return
        try:
            ptuid = getattr(message, "parent_tool_use_id", None)
            if isinstance(message, AssistantMessage):
                self._observe_assistant(message, ptuid, tm)
            elif isinstance(message, UserMessage):
                d = self._msg_to_dict(message)
                if d:
                    self._conversations.setdefault(ptuid, []).append(d)
                self._last_msg_us[ptuid] = _now_us()
            elif isinstance(message, ResultMessage):
                self._observe_result(message, tm)
        except Exception:
            logger.debug("observe_message error", exc_info=True)

    def _observe_assistant(self, msg: Any, ptuid: str | None, tm: TraceManager) -> None:
        """Create a model_call span from an AssistantMessage."""
        now_us = _now_us()
        blocks = getattr(msg, "content", [])
        thinking_parts, content_parts, tool_calls = _parse_content_blocks(blocks)

        # Record tool ownership for every ToolUseBlock (hooks use this for parent resolution)
        for tc in tool_calls:
            block_id = tc.get("id")
            if block_id:
                self._tool_owner[block_id] = ptuid

        # Skip truly empty AssistantMessages (no blocks at all)
        if not thinking_parts and not content_parts and not tool_calls:
            self._last_msg_us[ptuid] = now_us
            return

        model_name = getattr(msg, "model", None) or "claude"
        parent = self._resolve_parent_for_ptuid(ptuid)

        # Track that this agent context had at least one AssistantMessage
        if ptuid is not None:
            self._agent_has_assistant.add(ptuid)

        output_meta: dict[str, Any] = {"model": model_name, "role": "assistant"}
        if thinking_parts:
            output_meta["reasoning"] = "\n".join(thinking_parts)
        if content_parts:
            output_meta["content"] = "\n".join(content_parts)
        if tool_calls:
            output_meta["tool_calls"] = tool_calls

        # Conversation snapshot as input
        conv = self._conversations.setdefault(ptuid, [])
        start_us = self._last_msg_us.get(ptuid, now_us)

        meta: dict[str, Any] = {
            "func": model_name,
            "task_type": MODEL_CALL,
            "parent": parent,
            "start_us": start_us,
            "end_us": now_us,
            "model_name": model_name,
            "model_output_meta": output_meta,
        }
        if conv:
            meta["model_input_meta"] = list(conv)

        tm.ingest_external_span(meta)

        # Update conversation and timestamp
        d = self._msg_to_dict(msg)
        if d:
            conv.append(d)
        self._last_msg_us[ptuid] = now_us

    def _observe_result(self, msg: Any, tm: TraceManager) -> None:
        """Update root span with session summary from ResultMessage."""
        end_us = _now_us()
        if self.root_task_id is not None:
            updates = {
                "end_us": end_us,
                "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_cost_usd": getattr(msg, "total_cost_usd", None),
                "usage": getattr(msg, "usage", None) or {},
                "session_duration_ms": getattr(msg, "duration_ms", None),
                "num_turns": getattr(msg, "num_turns", None),
                "stop_reason": getattr(msg, "stop_reason", None),
            }
            tm.update_external_span(self.root_task_id, updates)
            return

        # Fallback: create standalone session_summary span
        duration_ms = getattr(msg, "duration_ms", 0) or 0
        start_us = end_us - duration_ms * 1000 if duration_ms else end_us
        tm.ingest_external_span(
            {
                "func": "session",
                "task_type": "session_summary",
                "parent": None,
                "start_us": start_us,
                "end_us": end_us,
                "total_cost_usd": getattr(msg, "total_cost_usd", None),
                "usage": getattr(msg, "usage", None) or {},
                "session_duration_ms": duration_ms,
                "num_turns": getattr(msg, "num_turns", None),
                "stop_reason": getattr(msg, "stop_reason", None),
            }
        )


# ── Hook injection ──


def inject_tracing_hooks(options: Any, tm: TraceManager | None) -> TracingHookState:
    """Merge tracing hooks into options.hooks without replacing user hooks."""
    state = TracingHookState()
    if tm is None or not tm.config.is_collecting:
        return state

    # Create root session span
    root_id = tm.allocate_external_task_id()
    now_us = _now_us()
    tm.ingest_external_span(
        {
            "func": "session",
            "task_type": "session_summary",
            "parent": None,
            "start_us": now_us,
            "end_us": now_us + 1,  # placeholder, updated by _observe_result
        },
        task_id=root_id,
    )
    state.root_task_id = root_id

    from claude_agent_sdk import HookMatcher

    hooks = {
        "PreToolUse": [
            HookMatcher(
                matcher=None, hooks=[_make_pre_tool_hook(state, tm)], timeout=None
            )
        ],
        "PostToolUse": [
            HookMatcher(
                matcher=None, hooks=[_make_post_tool_hook(state, tm)], timeout=None
            )
        ],
        "PostToolUseFailure": [
            HookMatcher(
                matcher=None,
                hooks=[_make_post_tool_failure_hook(state, tm)],
                timeout=None,
            )
        ],
        "SubagentStart": [
            HookMatcher(
                matcher=None, hooks=[_make_subagent_start_hook(state, tm)], timeout=None
            )
        ],
        "SubagentStop": [
            HookMatcher(
                matcher=None, hooks=[_make_subagent_stop_hook(state, tm)], timeout=None
            )
        ],
    }
    if options.hooks is None:
        options.hooks = {}
    for event_name, matchers in hooks.items():
        options.hooks[event_name] = options.hooks.get(event_name, []) + matchers

    return state


# ── Hook factories ──


def _make_pre_tool_hook(state: TracingHookState, tm: TraceManager):
    async def hook(input_data, match_result, context):
        try:
            tool_use_id = input_data.get("tool_use_id", "")
            tool_name = input_data.get("tool_name", "tool")
            tool_input = input_data.get("tool_input", {})

            task_id = tm.allocate_external_task_id()
            start_us = _now_us()
            state.pending_tools[tool_use_id] = (
                task_id,
                start_us,
                tool_name,
                tool_input,
            )

            # If this is the Agent tool, queue the tool_use_id for SubagentStart
            if tool_name == "Agent":
                state._pending_agent_tuids.append(tool_use_id)
        except Exception:
            logger.debug("pre_tool_hook error", exc_info=True)
        return {}

    return hook


def _make_post_tool_hook(state: TracingHookState, tm: TraceManager):
    async def hook(input_data, match_result, context):
        try:
            tool_use_id = input_data.get("tool_use_id", "")
            pending = state.pending_tools.pop(tool_use_id, None)
            if pending is None:
                return {}
            task_id, start_us, tool_name, tool_input = pending

            # Agent tool: span is managed by SubagentStart/Stop.
            # For simple subagents (no tool calls, one model turn), the SDK
            # doesn't yield their AssistantMessage — create a synthetic
            # model_call from the tool_response so the agent_call isn't empty.
            if tool_use_id in state._tuid_to_agent:
                if tool_use_id not in state._agent_has_assistant:
                    agent_task_id = state._tuid_to_agent[tool_use_id]
                    end_us = _now_us()
                    conv = state._conversations.get(tool_use_id, [])
                    agent_start_us = state._last_msg_us.get(tool_use_id, end_us)
                    tool_response = input_data.get("tool_response")
                    output_meta: dict[str, Any] = {
                        "role": "assistant",
                        "model": "claude",
                    }
                    if tool_response:
                        output_meta["content"] = _truncate(str(tool_response))
                    synth_meta: dict[str, Any] = {
                        "func": "claude",
                        "task_type": MODEL_CALL,
                        "parent": agent_task_id,
                        "start_us": agent_start_us,
                        "end_us": end_us,
                        "model_name": "claude",
                        "model_output_meta": output_meta,
                    }
                    if conv:
                        synth_meta["model_input_meta"] = list(conv)
                    tm.ingest_external_span(synth_meta)
                # Agent tool fully done — clean up lookup dicts
                state._tuid_to_agent.pop(tool_use_id, None)
                return {}

            end_us = _now_us()
            meta: dict[str, Any] = {
                "func": tool_name,
                "task_type": TOOL_CALL,
                "parent": state._resolve_parent_for_tool(tool_use_id),
                "start_us": start_us,
                "end_us": end_us,
                "tool_input_meta": {"name": tool_name, "arguments": tool_input},
            }
            tool_response = input_data.get("tool_response")
            if tool_response is not None:
                meta["tool_output_meta"] = _truncate(tool_response)
            tm.ingest_external_span(meta, task_id=task_id)
        except Exception:
            logger.debug("post_tool_hook error", exc_info=True)
        return {}

    return hook


def _make_post_tool_failure_hook(state: TracingHookState, tm: TraceManager):
    async def hook(input_data, match_result, context):
        try:
            tool_use_id = input_data.get("tool_use_id", "")
            pending = state.pending_tools.pop(tool_use_id, None)
            if pending is None:
                return {}
            task_id, start_us, tool_name, tool_input = pending

            # If this was an Agent tool that failed before SubagentStart,
            # remove its stale tuid from the queue to prevent mispairing.
            if tool_use_id in state._pending_agent_tuids:
                state._pending_agent_tuids.remove(tool_use_id)

            # Agent tool failure: still create a span but mark as error
            end_us = _now_us()
            task_type = (
                "agent_call" if tool_use_id in state._tuid_to_agent else TOOL_CALL
            )
            meta: dict[str, Any] = {
                "func": tool_name,
                "task_type": task_type,
                "parent": state._resolve_parent_for_tool(tool_use_id),
                "start_us": start_us,
                "end_us": end_us,
                "error": input_data.get("error", "unknown error"),
                "tool_input_meta": {"name": tool_name, "arguments": tool_input},
            }
            tm.ingest_external_span(meta, task_id=task_id)
        except Exception:
            logger.debug("post_tool_failure_hook error", exc_info=True)
        return {}

    return hook


def _make_subagent_start_hook(state: TracingHookState, tm: TraceManager):
    async def hook(input_data, match_result, context):
        try:
            agent_id = input_data.get("agent_id", "agent")

            # Consume the first queued tool_use_id from PreToolUse(Agent)
            tuid = (
                state._pending_agent_tuids.popleft()
                if state._pending_agent_tuids
                else None
            )

            if tuid and tuid in state.pending_tools:
                task_id, start_us, _, _ = state.pending_tools[tuid]
            else:
                task_id = tm.allocate_external_task_id()
                start_us = _now_us()
                if tuid is None:
                    state._fallback_counter += 1
                    tuid = f"_agent_{agent_id}_{state._fallback_counter}"

            # Resolve parent via consumer-observed tool ownership
            parent = state._resolve_parent_for_tool(tuid)

            # Register state AFTER parent resolution but BEFORE ingest
            state._tuid_to_agent[tuid] = task_id
            state._agent_stack.append(
                _AgentContext(
                    agent_id=agent_id,
                    task_id=task_id,
                    tuid=tuid,
                    start_us=start_us,
                )
            )
            state._conversations[tuid] = []
            state._last_msg_us[tuid] = _now_us()

            tm.ingest_external_span(
                {
                    "func": agent_id,
                    "task_type": "agent_call",
                    "parent": parent,
                    "start_us": start_us,
                    "end_us": start_us + 1,  # placeholder, updated on stop
                    "agent_id": agent_id,
                    "agent_type": input_data.get("agent_type", ""),
                },
                task_id=task_id,
            )
        except Exception:
            logger.debug("subagent_start_hook error", exc_info=True)
        return {}

    return hook


def _make_subagent_stop_hook(state: TracingHookState, tm: TraceManager):
    async def hook(input_data, match_result, context):
        try:
            if not state._agent_stack:
                return {}

            # Find the matching agent context by agent_id
            # (parallel subagents may stop in any order)
            agent_id = input_data.get("agent_id")
            idx = None
            if agent_id:
                for i in range(len(state._agent_stack) - 1, -1, -1):
                    if state._agent_stack[i].agent_id == agent_id:
                        idx = i
                        break
            if idx is None and agent_id:
                logger.debug(
                    f"SubagentStop: agent_id={agent_id} not found on stack, popping top"
                )
            agent_ctx = state._agent_stack.pop(idx if idx is not None else -1)
            end_us = _now_us()

            # Flush orphaned pending tools that belong to this agent.
            # Tools that started (PreToolUse) but never completed (e.g.
            # permission denied, timeout, session ended) are finalized here.
            orphan_tuids = [
                tuid
                for tuid, (_, _, _, _) in state.pending_tools.items()
                if state._tool_owner.get(tuid) == agent_ctx.tuid
            ]
            for tuid in orphan_tuids:
                task_id_orphan, start_us_orphan, tool_name_orphan, tool_input_orphan = (
                    state.pending_tools.pop(tuid)
                )
                tm.ingest_external_span(
                    {
                        "func": tool_name_orphan,
                        "task_type": TOOL_CALL,
                        "parent": agent_ctx.task_id,
                        "start_us": start_us_orphan,
                        "end_us": end_us,
                        "tool_input_meta": {
                            "name": tool_name_orphan,
                            "arguments": tool_input_orphan,
                        },
                        "error": "abandoned (agent session ended)",
                    },
                    task_id=task_id_orphan,
                )

            # Update agent_call span's end time
            tm.update_external_span(agent_ctx.task_id, {"end_us": end_us})

            # Clean up per-subagent state to prevent memory growth
            state._conversations.pop(agent_ctx.tuid, None)
            state._last_msg_us.pop(agent_ctx.tuid, None)
            state._agent_has_assistant.discard(agent_ctx.tuid)
        except Exception:
            logger.debug("subagent_stop_hook error", exc_info=True)
        return {}

    return hook
