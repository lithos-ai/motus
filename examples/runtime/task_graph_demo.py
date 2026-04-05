"""Task Graph Demo — building and executing dataflow graphs with motus.

Covers: sync/async tasks, dependency tracking, non-blocking operators,
multi-return (num_returns), class method tasks, and await support.

See also:
  - resilient_tasks.py   — retries, timeouts, policy overrides
  - hooks_demo.py        — global / per-task lifecycle hooks
  - stacktrace_demo.py   — creation-chain error reporting

Run:  MOTUS_LOG_LEVEL=WARNING python examples/runtime/task_graph_demo.py
"""

import asyncio

from motus.runtime import resolve, shutdown
from motus.runtime.agent_task import agent_task

# ═══════════════════════════════════════════════════════════════════════
# 1. Sync & async tasks — both work transparently
# ═══════════════════════════════════════════════════════════════════════


@agent_task
def add(a, b):
    return a + b


@agent_task
async def async_add(a, b):
    await asyncio.sleep(0.01)
    return a + b


def demo_sync_async():
    print("── 1. Sync & Async Tasks ──")
    r1 = add(2, 3)
    r2 = async_add(10, 20)
    print(f"  sync  add(2, 3)       = {resolve(r1, timeout=5)}")
    print(f"  async async_add(10,20)= {resolve(r2, timeout=5)}")


# ═══════════════════════════════════════════════════════════════════════
# 2. Automatic dependency tracking — pass futures as arguments
# ═══════════════════════════════════════════════════════════════════════


@agent_task
def multiply(x, y):
    return x * y


@agent_task
def format_result(label, value):
    return f"{label}: {value}"


def demo_dependency_tracking():
    print("\n── 2. Dependency Tracking ──")
    a = add(3, 4)  # → 7
    b = multiply(a, 10)  # waits for a → 70
    msg = format_result("answer", b)  # waits for b
    print(f"  add(3,4) → multiply(·,10) → format = {resolve(msg, timeout=5)}")


# ═══════════════════════════════════════════════════════════════════════
# 3. Non-blocking operators — build computation graphs with Python syntax
# ═══════════════════════════════════════════════════════════════════════


@agent_task
def make_data():
    return {"scores": [10, 20, 30], "name": "Alice"}


def demo_nonblocking_ops():
    print("\n── 3. Non-blocking Operators ──")

    x = add(10, 5)  # → 15
    y = add(3, 2)  # → 5

    # Arithmetic — returns new AgentFutures, no blocking
    total = x + y  # 15 + 5 = 20
    diff = x - y  # 15 - 5 = 10
    scaled = x * 3  # 15 * 3 = 45
    reverse = 100 - x  # 100 - 15 = 85  (radd/rsub)
    print(f"  x + y    = {resolve(total, timeout=5)}")
    print(f"  x - y    = {resolve(diff, timeout=5)}")
    print(f"  x * 3    = {resolve(scaled, timeout=5)}")
    print(f"  100 - x  = {resolve(reverse, timeout=5)}")

    # Getitem / getattr — also non-blocking
    data = make_data()
    scores = data["scores"]  # deferred __getitem__
    first = scores[0]  # chained __getitem__
    name = data["name"]
    print(f"  data['scores'][0] = {resolve(first, timeout=5)}")
    print(f"  data['name']      = {resolve(name, timeout=5)}")


# ═══════════════════════════════════════════════════════════════════════
# 4. Multi-return tasks — independent futures per return element
# ═══════════════════════════════════════════════════════════════════════


@agent_task(num_returns=2)
def split(data):
    mid = len(data) // 2
    return data[:mid], data[mid:]


@agent_task(num_returns=3)
async def analyze(text):
    words = text.split()
    return len(words), words[0], words[-1]


@agent_task
def process_half(half):
    return sum(half)


def demo_multi_return():
    print("\n── 4. Multi-Return Tasks ──")

    # Basic: split a list into two independent futures
    left, right = split([1, 2, 3, 4, 5, 6])
    # Each downstream task depends on only ONE future — fine-grained scheduling
    sum_left = process_half(left)
    sum_right = process_half(right)
    print(
        f"  split([1..6])  left={resolve(left, timeout=5)}, right={resolve(right, timeout=5)}"
    )
    print(
        f"  sum(left)={resolve(sum_left, timeout=5)}, sum(right)={resolve(sum_right, timeout=5)}"
    )

    # Async multi-return
    count, first, last = analyze("the quick brown fox")
    print(
        f"  analyze('the quick brown fox')  count={resolve(count, timeout=5)}, "
        f"first={resolve(first, timeout=5)!r}, last={resolve(last, timeout=5)!r}"
    )

    # Policy override: add num_returns at call site
    @agent_task
    def pair(a, b):
        return a, b

    x, y = pair.policy(num_returns=2)(10, 20)
    print(
        f"  pair.policy(num_returns=2)(10,20)  x={resolve(x, timeout=5)}, y={resolve(y, timeout=5)}"
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. Multi-return with nested futures (RESOLVE path)
# ═══════════════════════════════════════════════════════════════════════


@agent_task(num_returns=2)
def parallel_fetch():
    """Each returned element is itself a future — triggers the RESOLVE path."""
    a = add(100, 1)  # AgentFuture → 101
    b = add(200, 2)  # AgentFuture → 202
    return a, b


def demo_nested_multi_return():
    print("\n── 5. Multi-Return + Nested Futures ──")
    r1, r2 = parallel_fetch()
    print(
        f"  parallel_fetch()  r1={resolve(r1, timeout=5)}, r2={resolve(r2, timeout=5)}"
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. Descriptor protocol — @agent_task on class methods
# ═══════════════════════════════════════════════════════════════════════


class Calculator:
    def __init__(self, base):
        self.base = base

    @agent_task
    def compute(self, x):
        return self.base + x

    @agent_task(retries=1)
    def risky_compute(self, x):
        return self.base * x


def demo_descriptor():
    print("\n── 6. Class Method Tasks ──")
    calc = Calculator(100)
    r1 = calc.compute(42)
    r2 = calc.risky_compute(3)
    # Policy override on bound method
    r3 = calc.risky_compute.policy(retries=0)(5)
    print(f"  Calculator(100).compute(42)       = {resolve(r1, timeout=5)}")
    print(f"  Calculator(100).risky_compute(3)  = {resolve(r2, timeout=5)}")
    print(f"  ...policy(retries=0)(5)           = {resolve(r3, timeout=5)}")


# ═══════════════════════════════════════════════════════════════════════
# 7. Await support inside async tasks
# ═══════════════════════════════════════════════════════════════════════


@agent_task
async def step_a():
    await asyncio.sleep(0.01)
    return 10


@agent_task
async def step_b(x):
    return x + 5


@agent_task
async def orchestrator():
    """Use 'await' to get AgentFuture results without blocking the event loop."""
    a = step_a()
    val_a = await a  # non-blocking await inside async task
    b = step_b(val_a)
    val_b = await b
    return val_b


def demo_await():
    print("\n── 7. Await in Async Tasks ──")
    result = resolve(orchestrator(), timeout=5)
    print(f"  orchestrator() = {result}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    demo_sync_async()
    demo_dependency_tracking()
    demo_nonblocking_ops()
    demo_multi_return()
    demo_nested_multi_return()
    demo_descriptor()
    demo_await()

    shutdown()
    print("\n✓ All demos passed!")


if __name__ == "__main__":
    main()
