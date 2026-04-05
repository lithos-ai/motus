from typing import Literal

# Task type categories.
TaskType = Literal["normal_task", "tool_call", "model_call", "agent_call", "magic_task"]

TASK: TaskType = "normal_task"
TOOL_CALL: TaskType = "tool_call"
MODEL_CALL: TaskType = "model_call"
AGENT_CALL: TaskType = "agent_call"
# Internal/deferred operations from AgentFuture magic methods (__getattr__, __getitem__, etc.).
# Hidden from the main trace view by default.
MAGIC_TASK: TaskType = "magic_task"


class AgentFutureId:
    id: int

    def __init__(self, id: int):
        self.id = id

    def next(self):
        self.id += 1


class AgentTaskId:
    id: int

    def __init__(self, id: int):
        self.id = id

    def next(self):
        self.id += 1
