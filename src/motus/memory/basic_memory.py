"""
BasicMemory - Simple append-only memory with no compaction.

Messages are appended to a list. No compaction, no memory tools,
no conversation logging. When the context window overflows, the
agent will receive an API error from the model provider.

This is the default memory for motus agents — suitable for short
conversations or when the caller manages context externally.
"""

from typing import Dict, Optional

from .base_memory import BaseMemory


class BasicMemory(BaseMemory):
    """Append-only memory with no compaction or tools.

    The simplest memory implementation. Messages accumulate until the
    conversation ends or the context window is exceeded.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        system_prompt: str = "",
        max_tool_result_tokens: int = 50000,
        tool_result_truncation_suffix: str = "\n\n... [content truncated due to length]",
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            max_tool_result_tokens=max_tool_result_tokens,
            tool_result_truncation_suffix=tool_result_truncation_suffix,
        )

    async def compact(self, **kwargs) -> None:
        """No-op — BasicMemory does not support compaction."""
        return None

    def reset(self) -> Dict[str, int]:
        """Clear all messages."""
        count = len(self._messages)
        self.clear_messages()
        return {"messages": count, "short_term": 0, "long_term": 0}
