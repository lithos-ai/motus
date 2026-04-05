"""Resilient Tasks — showcasing retries, timeouts, and runtime policy overrides.

Run:  MOTUS_LOG_LEVEL=WARNING python examples/resilient_tasks.py
No API keys or network access needed.
"""

import time

from motus.runtime import shutdown
from motus.runtime.agent_task import agent_task

# ---------------------------------------------------------------------------
# 1. Retries: a "flaky" function that fails the first 2 times
# ---------------------------------------------------------------------------
call_count = 0


@agent_task(retries=3)
def fetch_data(url: str) -> dict:
    global call_count
    call_count += 1
    if call_count <= 2:
        raise ConnectionError(f"Network glitch #{call_count}")
    return {"url": url, "payload": [1, 2, 3]}


# ---------------------------------------------------------------------------
# 2. Timeout: a slow computation guarded by a deadline
# ---------------------------------------------------------------------------
@agent_task(timeout=0.3)
def slow_task():
    time.sleep(10)
    return "unreachable"


@agent_task(timeout=2)
def fast_task():
    return "done"


# ---------------------------------------------------------------------------
# 3. Policy override: same function, different behavior at call site
# ---------------------------------------------------------------------------
flaky_count = 0


@agent_task(retries=0)
def flaky_service():
    """Fails twice, then succeeds. With retries=0 it fails; with retries=2 it succeeds."""
    global flaky_count
    flaky_count += 1
    if flaky_count <= 2:
        raise ConnectionError(f"flake #{flaky_count}")
    return "recovered"


# ---------------------------------------------------------------------------
# 4. Task composition: futures as arguments, automatic dependency tracking
# ---------------------------------------------------------------------------
@agent_task
def pipeline(url: str) -> int:
    raw = fetch_data(url)  # returns AgentFuture (may retry internally)
    result = transform(raw)  # depends on raw — scheduled automatically
    return result.af_result()


@agent_task
def transform(data: dict) -> int:
    return sum(data["payload"]) * 10


def main():
    global call_count, flaky_count

    # --- Retries demo ---
    call_count = 0
    future = fetch_data("https://api.example.com/data")
    result = future.af_result(timeout=10)
    print(f"[retries]  fetch_data succeeded after {call_count} attempts: {result}")

    # --- Timeout demo ---
    # fast_task completes within its 2s timeout
    print(f"[timeout]  fast_task result: {fast_task().af_result(timeout=5)}")
    # slow_task exceeds its 0.3s timeout → TimeoutError
    try:
        slow_task().af_result(timeout=5)
    except TimeoutError:
        print("[timeout]  slow_task timed out as expected (0.3s deadline)")

    # --- Policy override demo ---
    # Default retries=0 → fails on first flake
    flaky_count = 0
    try:
        flaky_service().af_result(timeout=5)
    except ConnectionError:
        print(f"[override] retries=0: failed after {flaky_count} attempt(s)")

    # Override to retries=3 → survives flakes and succeeds
    flaky_count = 0
    result = flaky_service.policy(retries=3)().af_result(timeout=10)
    print(f"[override] retries=3: succeeded after {flaky_count} attempt(s) → {result}")

    # --- Pipeline composition demo ---
    call_count = 0
    p = pipeline("https://api.example.com/v2")
    print(f"[pipeline] composed result: {p.af_result(timeout=10)}")

    shutdown()
    print("\nAll demos passed!")


if __name__ == "__main__":
    main()
