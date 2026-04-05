import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, List

from motus.models import ChatMessage, ToolCall


# Helper Functions
def get_today_str() -> str:
    """Get today's date as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d")


def format_assistant_message_for_history(msg: ChatMessage) -> ChatMessage:
    """Pass through ChatMessage for message history (deprecated, kept for compatibility)."""
    return msg


def format_tool_results_for_history(
    tool_calls: List[ToolCall], results: List[str]
) -> List[ChatMessage]:
    """Format tool results as ChatMessage objects for message history."""

    def to_jsonable(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {k: to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return {k: to_jsonable(v) for k, v in value.__dict__.items()}
        return value

    tool_messages = []
    for tc, result in zip(tool_calls, results):
        tool_messages.append(
            ChatMessage(
                role="tool",
                tool_call_id=tc.id,
                content=(
                    result
                    if isinstance(result, str)
                    else json.dumps(to_jsonable(result), ensure_ascii=True, default=str)
                ),
            )
        )
    return tool_messages
