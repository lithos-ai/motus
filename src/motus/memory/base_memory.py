"""
Abstract base class for memory implementations.

Defines the minimal interface required by AgentBase for:
- Working memory management (messages list)
- Context construction (system prompt + messages)
- Token estimation
- Compaction
- Lifecycle management

BaseMemory.__init__() sets common attributes (_model_name, _messages,
_trace_log, _system_prompt, etc). Subclasses should call super().__init__()
or pass keyword arguments to it.
"""

import copy
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import tiktoken

from motus.models import ChatMessage

logger = logging.getLogger(__name__)


class BaseMemory(ABC):
    """
    Abstract base class for agent memory implementations.

    Provides default implementations for common operations on the
    working memory (messages list, system prompt, context construction,
    token estimation, tool result truncation, trace logging).

    Subclasses only need to implement:
    - compact(): memory-specific compaction strategy
    - reset(): memory-specific cleanup logic
    - _auto_compact(): when/whether to trigger auto-compaction (default no-op)

    And optionally override:
    - construct_system_prompt(): if system prompt needs enrichment (e.g. Memory adds memory context)
    - _add_trace_event(): if richer trace logging is needed
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        system_prompt: str = "",
        max_tool_result_tokens: int = 50000,
        tool_result_truncation_suffix: str = "\n\n... [content truncated due to length]",
        enable_memory_tools: bool = True,
    ):
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._messages: List[ChatMessage] = []
        self._trace_log: List[Dict[str, Any]] = []
        self._max_tool_result_tokens = max_tool_result_tokens
        self._tool_result_truncation_suffix = tool_result_truncation_suffix
        self._enable_memory_tools = enable_memory_tools

    # -------------------------------------------------------------------------
    # Working Memory (default implementations)
    # -------------------------------------------------------------------------

    @property
    def messages(self) -> List[ChatMessage]:
        """Get current conversation messages (defensive copy)."""
        return self._messages.copy()

    async def add_message(self, message: ChatMessage) -> None:
        """Add a message to working memory.

        Truncates tool results if they exceed the token limit,
        then records a trace event, then checks auto-compaction.
        """
        self._append_message(message)
        await self._auto_compact()

    def _append_message(self, message: ChatMessage) -> None:
        """Append a message with truncation and trace, but no auto-compaction.

        Used by add_message() (which adds auto-compaction on top) and by
        sync deserialization paths like Memory.from_dict().
        """
        if (
            message.role == "tool"
            and self._max_tool_result_tokens > 0
            and message.content
        ):
            message = self._truncate_tool_result(message)

        self._messages.append(message)
        self._add_trace_event(message)

    def clear_messages(self) -> None:
        """Clear all messages from working memory."""
        self._messages.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for context construction."""
        self._system_prompt = prompt

    def construct_system_prompt(self) -> ChatMessage:
        """Construct the system prompt message.

        Override to enrich (e.g. append memory context).
        """
        return ChatMessage.system_message(self._system_prompt)

    def get_context(self) -> List[ChatMessage]:
        """Get messages for the current context window.

        Constructs the full context by prepending a system message (containing
        the system prompt and memory prompt) to the current working memory messages.

        Returns:
            List of messages for the context window, with system message first
        """

        # Use get_memory_prompt() to provide memory context
        # The model can retrieve the summary via memory tools if needed
        # Skip memory prompt if memory is disabled
        return [self.construct_system_prompt()] + self._messages

    # -------------------------------------------------------------------------
    # Auto-compaction hook (default no-op, override in subclasses)
    # -------------------------------------------------------------------------

    async def _auto_compact(self) -> None:
        """Check if auto-compaction should trigger. Default: no-op."""
        pass

    # -------------------------------------------------------------------------
    # Token Estimation (shared concrete implementations)
    # -------------------------------------------------------------------------

    def _get_tiktoken_encoding(self) -> tiktoken.Encoding:
        """Get the tiktoken encoding for the current model."""
        try:
            return tiktoken.encoding_for_model(self._model_name or "")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def estimate_message_tokens(self, message: ChatMessage) -> int:
        """Estimate token count for a single message."""
        encoding = self._get_tiktoken_encoding()
        # Base overhead for message structure
        tokens = 4
        if message.content:
            tokens += len(encoding.encode(message.content, disallowed_special=()))
        if message.tool_calls:
            for tc in message.tool_calls:
                tokens += len(encoding.encode(tc.function.name, disallowed_special=()))
                tokens += len(
                    encoding.encode(tc.function.arguments, disallowed_special=())
                )
                tokens += 5
        if message.name:
            tokens += len(encoding.encode(message.name, disallowed_special=()))
        return tokens

    def estimate_working_memory_tokens(self) -> int:
        """Estimate total token count for current working memory."""
        return sum(self.estimate_message_tokens(m) for m in self.get_context()) + 3

    # -------------------------------------------------------------------------
    # Tool result truncation (shared concrete implementation)
    # -------------------------------------------------------------------------

    def _truncate_tool_result(self, message: ChatMessage) -> ChatMessage:
        """Truncate tool result content if it exceeds token limit."""
        if not message.content:
            return message
        max_tokens = self._max_tool_result_tokens
        if max_tokens <= 0:
            return message

        encoding = self._get_tiktoken_encoding()
        tokens = encoding.encode(message.content, disallowed_special=())
        if len(tokens) <= max_tokens:
            return message

        suffix = self._tool_result_truncation_suffix
        suffix_tokens = (
            len(encoding.encode(suffix, disallowed_special=())) if suffix else 0
        )

        # Only append suffix if it fits within the budget
        if suffix and suffix_tokens < max_tokens:
            truncate_at = max_tokens - suffix_tokens
            truncated_content = encoding.decode(tokens[:truncate_at]) + suffix
        else:
            truncate_at = max_tokens
            truncated_content = encoding.decode(tokens[:truncate_at])

        logger.info(
            f"Truncated tool result from {len(tokens)} to {truncate_at} tokens "
            f"(tool: {message.name})"
        )

        return ChatMessage(
            role=message.role,
            content=truncated_content,
            name=message.name,
            tool_call_id=message.tool_call_id,
        )

    # -------------------------------------------------------------------------
    # Trace (shared concrete implementations, _add_trace_event overridable)
    # -------------------------------------------------------------------------

    def _add_trace_event(self, message: ChatMessage) -> None:
        """Log a message event to the trace. Override for richer logging."""
        event: Dict[str, Any] = {
            "type": "message",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": message.role,
        }
        if message.content:
            event["content"] = message.content
        if message.role == "assistant" and message.tool_calls:
            event["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]
        if message.role == "tool":
            event["tool_call_id"] = getattr(message, "tool_call_id", None)
            event["tool_name"] = message.name
        self._trace_log.append(event)

    def get_memory_trace(self) -> Dict[str, Any]:
        """Get trace of all memory operations for debugging."""
        compaction_count = sum(
            1 for e in self._trace_log if e.get("type") == "compaction"
        )
        return {
            "events": self._trace_log.copy(),
            "total_events": len(self._trace_log),
            "total_compactions": compaction_count,
        }

    # -------------------------------------------------------------------------
    # Compaction (REQUIRED)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def compact(self, **kwargs) -> Any:
        """Compact working memory to reduce context size."""
        ...

    # -------------------------------------------------------------------------
    # Lifecycle (REQUIRED)
    # -------------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> Dict[str, int]:
        """Reset memory to initial state. Returns counts of cleared items."""
        ...

    # -------------------------------------------------------------------------
    # Fork
    # -------------------------------------------------------------------------

    def fork(self) -> "BaseMemory":
        """Create an independent copy of this memory instance.

        Copies messages and trace log, but shares the client reference
        (clients hold connection pools and should not be deep copied).
        Subclasses with external state (e.g. Memory with file/vector
        stores) should override to handle scope isolation.
        """
        clone = copy.copy(self)
        clone._messages = copy.deepcopy(self._messages)
        clone._trace_log = copy.deepcopy(self._trace_log)
        return clone

    # -------------------------------------------------------------------------
    # Optional: Lifecycle hooks (default no-op)
    # -------------------------------------------------------------------------

    def build_tools(self) -> list:
        """Return memory-provided tools for the agent.

        Subclasses can override to provide tool callables (decorated with @tool
        or plain functions). The agent merges these into its tool set during
        _init_tools(). Returns empty list when ``enable_memory_tools=False``.
        """
        return []

    def stop_promotion_task(self) -> None:
        """Stop any background tasks. No-op if not applicable."""
        pass
