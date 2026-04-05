"""Hooks Demo — global and per-task lifecycle callbacks.

motus fires hook events at task_start, task_end, and task_error.
You can register:
  - Global hooks   — fire for every task
  - Per-task hooks  — fire only for a specific task name
  - Decorator hooks — convenient @global_hook / @task_hook syntax

Run:  MOTUS_LOG_LEVEL=WARNING python examples/runtime/hooks_demo.py
"""

import time

from motus.runtime import shutdown
from motus.runtime.agent_task import agent_task
from motus.runtime.hooks import (
    global_hook,
    register_hook,
    register_task_hook,
    task_hook,
)

SETTLE = 0.1  # seconds — let async hooks fire after .af_result() returns


# ═══════════════════════════════════════════════════════════════════════
# 1. Global hooks — fire for every task
# ═══════════════════════════════════════════════════════════════════════

global_log = []


@agent_task
def double(x):
    return x * 2


@agent_task
def triple(x):
    return x * 3


def demo_global_hooks():
    print("── 1. Global Hooks ──")

    def on_start(event):
        global_log.append(f"start:{event.name}")

    def on_end(event):
        global_log.append(f"end:{event.name}={event.result}")

    register_hook("task_start", on_start)
    register_hook("task_end", on_end)

    double(5).af_result(timeout=5)
    triple(3).af_result(timeout=5)
    time.sleep(SETTLE)

    print(f"  log: {global_log}")
    assert any("double" in e and e.startswith("start:") for e in global_log)
    assert any("double" in e and "=10" in e for e in global_log)
    assert any("triple" in e and e.startswith("start:") for e in global_log)
    assert any("triple" in e and "=9" in e for e in global_log)


# ═══════════════════════════════════════════════════════════════════════
# 2. Per-task hooks — fire only for a specific task name
# ═══════════════════════════════════════════════════════════════════════

per_task_log = []


@agent_task
def important_task(x):
    return x + 1


@agent_task
def boring_task(x):
    return x - 1


def demo_per_task_hooks():
    print("\n── 2. Per-Task Hooks ──")

    def on_important_end(event):
        per_task_log.append(f"end:{event.name}")

    # Pass the function itself so the hook matches the qualified name
    register_task_hook(important_task, "task_end", on_important_end)

    important_task(10).af_result(timeout=5)
    boring_task(10).af_result(timeout=5)
    time.sleep(SETTLE)

    print(f"  log: {per_task_log}")
    assert any("important_task" in e for e in per_task_log)
    assert not any("boring_task" in e for e in per_task_log)


# ═══════════════════════════════════════════════════════════════════════
# 3. Error hooks — observe task failures
# ═══════════════════════════════════════════════════════════════════════

error_log = []


@agent_task
def failing_task():
    raise ValueError("oops")


def demo_error_hooks():
    print("\n── 3. Error Hooks ──")

    def on_error(event):
        error_log.append(f"error:{event.name}:{type(event.error).__name__}")

    register_hook("task_error", on_error)

    try:
        failing_task().af_result(timeout=5)
    except ValueError:
        pass

    time.sleep(SETTLE)
    print(f"  log: {error_log}")
    assert any("failing_task" in e and "ValueError" in e for e in error_log)


# ═══════════════════════════════════════════════════════════════════════
# 4. Decorator syntax — @global_hook / @task_hook
# ═══════════════════════════════════════════════════════════════════════

decorator_log = []


@agent_task
def decorated_task(x):
    return x**2


@agent_task
def other_task(x):
    return x


def demo_decorator_hooks():
    print("\n── 4. Decorator Hooks ──")

    @global_hook("task_start")
    def log_all_starts(event):
        decorator_log.append(f"deco_start:{event.name}")

    @task_hook(decorated_task, "task_end")
    def log_decorated_end(event):
        decorator_log.append(f"deco_end:{event.name}={event.result}")

    decorated_task(4).af_result(timeout=5)
    other_task(7).af_result(timeout=5)
    time.sleep(SETTLE)

    print(f"  log: {decorator_log}")
    assert any(
        "decorated_task" in e and e.startswith("deco_start:") for e in decorator_log
    )
    assert any("other_task" in e and e.startswith("deco_start:") for e in decorator_log)
    assert any("decorated_task" in e and "=16" in e for e in decorator_log)
    assert not any(
        "other_task" in e and e.startswith("deco_end:") for e in decorator_log
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. Hook metadata — access task_id and parent_stack
# ═══════════════════════════════════════════════════════════════════════

metadata_log = []


@agent_task
def simple():
    return 42


def demo_metadata():
    print("\n── 5. Hook Metadata ──")

    def on_start(event):
        metadata_log.append(
            {
                "name": event.name,
                "task_id": event.task_id,
                "has_parent_stack": "parent_stack" in event.metadata,
            }
        )

    register_hook("task_start", on_start)

    simple().af_result(timeout=5)

    entry = [e for e in metadata_log if "simple" in e["name"]][0]
    print(f"  task_id present: {entry['task_id'] is not None}")
    print(f"  parent_stack present: {entry['has_parent_stack']}")
    assert entry["task_id"] is not None
    assert entry["has_parent_stack"]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    demo_global_hooks()
    demo_per_task_hooks()
    demo_error_hooks()
    demo_decorator_hooks()
    demo_metadata()

    shutdown()
    print("\nAll hook demos passed!")


if __name__ == "__main__":
    main()
