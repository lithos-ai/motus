"""Task type constants used by tracing decorators and framework integrations."""

from typing import Literal

TaskType = Literal["normal_task", "tool_call", "model_call", "agent_call"]

TASK: TaskType = "normal_task"
TOOL_CALL: TaskType = "tool_call"
MODEL_CALL: TaskType = "model_call"
AGENT_CALL: TaskType = "agent_call"
