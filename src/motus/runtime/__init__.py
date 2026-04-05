from .agent_future import AgentFuture, cancel, cancelled, resolve
from .agent_runtime import (
    AgentRuntime,
    get_runtime,
    init,
    is_initialized,
    shutdown,
)
from .agent_task import AgentTaskDefinition, agent_task, register_agent_task
from .hooks import (
    HookEvent,
    HookManager,
    HookType,
    agent_task_hook,
    global_hook,
    hooks,
    model_task_hook,
    register_agent_hook,
    register_hook,
    register_model_hook,
    register_task_hook,
    register_tool_hook,
    register_type_hook,
    task_hook,
    tool_task_hook,
    type_hook,
)
from .task_instance import TaskPolicy

# The runtime is lazily initialized. Call init() to start explicitly,
# or it will auto-initialize on first use (e.g., when an @agent_task
# function is invoked).

__all__ = [
    "AgentFuture",
    "AgentRuntime",
    "AgentTaskDefinition",
    "HookEvent",
    "HookManager",
    "HookType",
    "TaskPolicy",
    "agent_task",
    "agent_task_hook",
    "cancel",
    "cancelled",
    "get_runtime",
    "global_hook",
    "hooks",
    "init",
    "is_initialized",
    "model_task_hook",
    "register_agent_hook",
    "register_agent_task",
    "register_hook",
    "register_model_hook",
    "register_task_hook",
    "register_tool_hook",
    "register_type_hook",
    "resolve",
    "shutdown",
    "task_hook",
    "tool_task_hook",
    "type_hook",
]
