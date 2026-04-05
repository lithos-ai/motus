from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from ..core import InputSchema
from ..core.decorators import tool

TodoStatus = Literal["pending", "in_progress", "completed"]

_items: list[dict] = []


class TodoItem(InputSchema):
    content: str = Field(description="Brief, actionable title of the task.")
    status: TodoStatus = Field(description="Current status of this task item.")
    activeForm: str | None = Field(
        default=None,
        description='Present-continuous label shown while in progress (e.g. "Running tests").',
    )

    model_config = ConfigDict(extra="forbid")


class TodoInput(InputSchema):
    todos: list[TodoItem] = Field(
        description="Full list of task items. Each call replaces the entire list.",
    )

    model_config = ConfigDict(extra="forbid")


@tool(schema=TodoInput)
async def to_do(todos: list) -> list[dict]:
    """Manage a structured task list for the current coding session. Helps track progress,
    organize multi-step work, and keep the user informed of overall status.

    #### When to Use This Tool
    Use this tool proactively in the following cases:

    1. Multi-step work - When the task needs 3 or more distinct steps
    2. Non-trivial work - Tasks that require planning or multiple operations
    3. User asks for a todo list - When the user explicitly requests it
    4. Multiple tasks provided - Numbered or comma-separated task lists
    5. New instructions arrive - Capture requirements immediately as todos
    6. Starting a task - Set one item to in_progress before beginning
    7. Finishing a task - Mark it completed and add follow-ups you discover

    #### When NOT to Use This Tool

    Do not use this tool when:
    1. There is a single, straightforward task
    2. The task is trivial and tracking adds no value
    3. The work can be done in fewer than 3 trivial steps
    4. The request is purely conversational or informational

    Note: if there is only one trivial task, skip this tool and do the task directly.

    #### Task States and Management

    1. **Task States**: Use these to track progress:
    - pending: Task not yet started
    - in_progress: Currently working on (limit to ONE task at a time)
    - completed: Task finished successfully

    **IMPORTANT**: Each task must include two forms:
    - content: imperative form (e.g., "Run tests", "Build the project")
    - activeForm: present continuous form shown during execution (e.g., "Running tests", "Building the project")

    2. **Task Management**:
    - Update task status as work progresses
    - Mark tasks complete immediately after finishing (no batching)
    - Exactly ONE task must be in_progress at a time
    - Finish the current task before starting a new one
    - Remove tasks that are no longer relevant

    3. **Task Completion Requirements**:
    - Only mark a task completed when it is fully done
    - If you hit errors or blockers, keep it in_progress
    - When blocked, add a new task describing what is needed to unblock
    - Never mark a task completed if:
        - Tests are failing
        - Implementation is partial
        - You encountered unresolved errors
        - You couldn't find necessary files or dependencies

    When unsure, use this tool. Proactive task tracking shows attentiveness and helps ensure all requirements are completed.
    """
    _items.clear()
    for t in todos:
        _items.append(
            {
                "content": t["content"],
                "status": t["status"],
                "activeForm": t.get("activeForm"),
            }
        )
    return list(_items)
