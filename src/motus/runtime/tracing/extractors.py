"""Task metadata extractors for tracing.

This module provides a pluggable system for extracting task-specific metadata
during tracing. Each task type can have its own extractor that knows how to
extract relevant information from the task's arguments and results.

To add a new task type:
1. Create a class that inherits from TaskMetadataExtractor
2. Implement extract_start_meta() and extract_end_meta()
3. Register it using register_extractor() or add to TASK_EXTRACTORS
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from motus.runtime.types import AGENT_CALL, MODEL_CALL, TOOL_CALL

logger = logging.getLogger("AgentTracer")


def safe_dump(obj: Any) -> Any:
    """Safely convert an object to a JSON-serializable dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


class TaskMetadataExtractor(ABC):
    """Abstract base class for extracting task-specific tracing metadata."""

    @abstractmethod
    def extract_start_meta(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Extract metadata when task starts."""
        ...

    @abstractmethod
    def extract_end_meta(self, result: Any) -> dict[str, Any]:
        """Extract metadata when task ends."""
        ...

    def extract_error_meta(self, error: Exception) -> dict[str, Any]:
        """Extract metadata on error. Override if needed."""
        return {}

    def on_task_start(
        self, task_id: Any, parent: Any, args: tuple, kwargs: dict
    ) -> None:
        """Called when a task starts. Override for custom logging or side effects.

        Args:
            task_id: The task identifier.
            parent: The parent task identifier, or None if no parent.
            args: Positional arguments passed to the task.
            kwargs: Keyword arguments passed to the task.
        """
        pass


class DefaultTaskExtractor(TaskMetadataExtractor):
    """Default extractor that returns empty metadata."""

    def extract_start_meta(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        return {}

    def extract_end_meta(self, result: Any) -> dict[str, Any]:
        return {}


class ModelServeTaskExtractor(TaskMetadataExtractor):
    """Extractor for model_serve_task - extracts model info, messages, and tools."""

    def _get_tool_json_schema(self, tool_name: str, tool_calls: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_calls[tool_name].description,
                "parameters": tool_calls[tool_name].json_schema,
            },
        }

    def extract_start_meta(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        # model_serve_task signature: (client, model, messages, tools, response_format)
        model_name = kwargs.get("model") or (
            args[1] if args and len(args) > 1 else None
        )
        messages = kwargs.get("messages") or (args[2] if args and len(args) > 2 else [])
        tool_calls = kwargs.get("tools") or (
            args[3] if args and len(args) > 3 else None
        )

        tool_meta = (
            [self._get_tool_json_schema(name, tool_calls) for name in tool_calls]
            if tool_calls
            else []
        )

        return {
            "model_name": model_name,
            "model_input_meta": [safe_dump(msg) for msg in messages],
            "tool_meta": tool_meta,
        }

    def extract_end_meta(self, result: Any) -> dict[str, Any]:
        # Extract which tools were actually chosen (if any)
        chosen_tools = []
        if result and hasattr(result, "to_message"):
            message = result.to_message()
            if hasattr(message, "tool_calls") and message.tool_calls:
                chosen_tools = [tc.function.name for tc in message.tool_calls]

        return {
            "model_output_meta": safe_dump(result),
            "chosen_tools": chosen_tools,  # Track which tools were actually chosen
        }


class AgentCallExtractor(TaskMetadataExtractor):
    """Extractor for AgentBase._execute - extracts agent name and object_id from self."""

    def extract_start_meta(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        # args[0] is self (the agent instance)
        agent = args[0] if args else None
        agent_id = getattr(agent, "name", None) or (
            agent.__class__.__name__ if agent else None
        )
        # id(agent) is unique per object instance - distinguishes same-named agents
        object_id = id(agent) if agent else None
        return {"agent_id": agent_id, "object_id": object_id}

    def extract_end_meta(self, result: Any) -> dict[str, Any]:
        return {}


class ToolCallExtractor(TaskMetadataExtractor):
    """Extractor for Tool._execute — extracts tool name and arguments."""

    def extract_start_meta(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        # Tool._execute signature: (self, **kwargs)
        # args[0] = Tool instance, kwargs = parsed tool arguments
        tool = args[0] if args else None
        tool_name = getattr(tool, "name", "unknown")

        return {
            "tool_input_meta": {
                "name": tool_name,
                "arguments": kwargs,
            }
        }

    def extract_end_meta(self, result: Any) -> dict[str, Any]:
        return {"tool_output_meta": safe_dump(result)}


# Registry mapping function names to their metadata extractors
_TASK_EXTRACTORS: dict[str, TaskMetadataExtractor] = {
    MODEL_CALL: ModelServeTaskExtractor(),
    TOOL_CALL: ToolCallExtractor(),
    AGENT_CALL: AgentCallExtractor(),
}
_DEFAULT_EXTRACTOR = DefaultTaskExtractor()


def get_extractor(task_type: str) -> TaskMetadataExtractor:
    """Get the metadata extractor for a task type."""
    return _TASK_EXTRACTORS.get(task_type, _DEFAULT_EXTRACTOR)


def register_extractor(func_name: str, extractor: TaskMetadataExtractor) -> None:
    """Register a custom extractor for a task type.

    This allows external code to add extractors for new task types without
    modifying this module.

    Args:
        func_name: The function/task name to register the extractor for.
        extractor: The extractor instance to use for this task type.
    """
    _TASK_EXTRACTORS[func_name] = extractor
