"""
Base chat client interface and common data structures.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field


class ReasoningConfig:
    """Configuration for LLM extended thinking / reasoning.

    Controls how much the model "thinks" before responding. Different models
    support different modes:

    - Opus 4.6 / Sonnet 4.6: adaptive mode with effort levels
    - Sonnet 4.5, Haiku 4.5, etc.: manual mode with explicit budget_tokens

    Usage:
        # Adaptive (Opus 4.6 / Sonnet 4.6) — model decides how much to think
        ReasoningConfig(effort="medium")
        ReasoningConfig(effort="low")   # minimal thinking, lowest cost

        # Manual (older models) — explicit token budget
        ReasoningConfig(budget_tokens=5000)

        # Disable thinking entirely
        ReasoningConfig.disabled()

        # Shorthand
        ReasoningConfig.auto()    # adaptive with default effort (high)
    """

    def __init__(
        self,
        enabled: bool = True,
        effort: str | None = None,
        budget_tokens: int | None = None,
    ):
        self.enabled = enabled
        self.effort = effort  # "max" (Opus only), "high", "medium", "low"
        self.budget_tokens = budget_tokens  # explicit budget for manual mode

    @classmethod
    def disabled(cls) -> "ReasoningConfig":
        """No thinking — cheapest, fastest."""
        return cls(enabled=False)

    @classmethod
    def auto(cls) -> "ReasoningConfig":
        """Adaptive thinking with default effort (high)."""
        return cls(enabled=True)

    @classmethod
    def light(cls) -> "ReasoningConfig":
        """Adaptive with low effort — minimal thinking, good for routine tasks."""
        return cls(enabled=True, effort="low")

    def to_anthropic_param(self, model: str, max_tokens: int) -> dict | None:
        """Convert to Anthropic API thinking parameter.

        Args:
            model: Model identifier (determines adaptive vs manual support).
            max_tokens: The max_tokens being used for the request.

        Returns:
            The ``thinking`` dict for the API, or None if disabled.
        """
        if not self.enabled:
            return {"type": "disabled"}

        is_adaptive_model = (
            "claude-opus-4-7" in model
            or "claude-opus-4-6" in model
            or "claude-sonnet-4-6" in model
        )

        if is_adaptive_model:
            param: dict = {"type": "adaptive"}
            # Effort is passed separately but we track it here for convenience
            return param
        else:
            # Manual mode — use explicit budget or default to max
            budget = self.budget_tokens or (max_tokens - 1)
            budget = max(budget, 1024)  # Anthropic minimum
            return {"type": "enabled", "budget_tokens": budget}

    def to_anthropic_effort(self, model: str) -> str | None:
        """Return the effort level for the Anthropic output_config, if applicable."""
        if not self.enabled:
            return None
        is_adaptive_model = (
            "claude-opus-4-7" in model
            or "claude-opus-4-6" in model
            or "claude-sonnet-4-6" in model
        )
        if is_adaptive_model and self.effort:
            return self.effort
        return None

    def __repr__(self) -> str:
        if not self.enabled:
            return "ReasoningConfig(disabled)"
        parts = []
        if self.effort:
            parts.append(f"effort={self.effort!r}")
        if self.budget_tokens:
            parts.append(f"budget_tokens={self.budget_tokens}")
        return f"ReasoningConfig({', '.join(parts) or 'auto'})"


class CachePolicy(str, Enum):
    """Prompt caching strategy for providers that support it (e.g. Anthropic).

    NONE     – No caching.
    STATIC   – Cache system prompt and tool definitions (5-min TTL).
    AUTO     – STATIC + cache conversation turn prefix (5-min TTL).
    AUTO_1H  – Same as AUTO but with 1-hour TTL. Best for long-running
               agents where the system prompt and tools don't change.
    """

    NONE = "none"
    STATIC = "static"
    AUTO = "auto"
    AUTO_1H = "auto_1h"


if TYPE_CHECKING:
    from motus.models import ChatCompletion


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str
    function: FunctionCall
    type: str = "function"


class ToolDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}  # JSON Schema
    strict: Optional[bool] = None


class ChatMessage(BaseModel):
    """Unified message format for chat conversations."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    reasoning: Optional[str] = None  # Readable reasoning text
    reasoning_details: Optional[list[dict]] = None  # Opaque data for round-tripping
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool messages
    name: Optional[str] = None  # Tool name for tool messages
    base64_image: Optional[str] = None  # Optional base64 encoded image
    user_params: Optional[dict] = None  # Per-request parameters passed to the agent

    def __add__(self, other) -> List["ChatMessage"]:
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, ChatMessage):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and "
                f"'{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["ChatMessage"]:
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and "
                f"'{type(self).__name__}'"
            )

    @classmethod
    def system_message(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "ChatMessage":
        return cls(role="user", content=content, base64_image=base64_image)

    @classmethod
    def assistant_message(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        base64_image: Optional[str] = None,
        reasoning: Optional[str] = None,
        reasoning_details: Optional[list[dict]] = None,
    ) -> "ChatMessage":
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            base64_image=base64_image,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
        )

    @classmethod
    def tool_message(
        cls,
        content: str,
        tool_call_id: str,
        name: str,
        base64_image: Optional[str] = None,
    ) -> "ChatMessage":
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> "ChatMessage":
        """Create ChatMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            ToolCall(id=call.id, function=call.function, type="function")
            for call in tool_calls
        ]
        return cls(
            role="assistant",
            content=content if isinstance(content, str) else "".join(content),
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )

    @classmethod
    def from_completion(cls, completion: "ChatCompletion") -> "ChatMessage":
        """Create ChatMessage from ChatCompletion.

        Args:
            completion: ChatCompletion from chat client
        """
        tool_calls = completion.tool_calls or []
        content = completion.content or ""

        assistant_msg = (
            cls.from_tool_calls(
                content=content,
                tool_calls=tool_calls,
                reasoning=completion.reasoning,
                reasoning_details=completion.reasoning_details,
            )
            if tool_calls
            else cls.assistant_message(
                content=content,
                reasoning=completion.reasoning,
                reasoning_details=completion.reasoning_details,
            )
        )

        return assistant_msg


class ChatCompletion(BaseModel):
    """Unified completion response format."""

    id: str
    model: str
    content: Optional[str] = None
    reasoning: Optional[str] = None  # Readable reasoning text
    reasoning_details: Optional[list[dict]] = None  # Opaque data for round-tripping
    tool_calls: Optional[list[ToolCall]] = None
    finish_reason: str = "stop"
    parsed: Optional[Any] = None  # For structured output
    usage: dict = Field(default_factory=dict)

    def to_message(self) -> ChatMessage:
        """Convert completion to a ChatMessage for appending to conversation."""
        return ChatMessage.assistant_message(
            content=self.content,
            tool_calls=self.tool_calls,
            reasoning=self.reasoning,
            reasoning_details=self.reasoning_details,
        )


class BaseChatClient(ABC):
    """
    Abstract base class for chat completion clients.

    Implementations must provide:
    - create(): Standard chat completion
    - parse(): Structured output with Pydantic model parsing
    """

    @abstractmethod
    async def create(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion."""
        pass

    @abstractmethod
    async def parse(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Type[BaseModel],
        tools: Optional[list[ToolDefinition]] = None,
        reasoning: ReasoningConfig = ReasoningConfig.auto(),
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion with structured output parsing."""
        pass
