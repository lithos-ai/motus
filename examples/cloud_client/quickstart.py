"""End-to-end tour of motus.cloud.

Walks through every public capability of the Python client in one script, so
a reader can scroll top to bottom and see each feature demonstrated against
a live server. Prompts are intentionally tiny; the script cleans up after
itself on success.

Run against a local ``motus serve`` instance:

    motus serve start tests.unit.serve.mock_agent:echo_agent --port 8000
    uv run python examples/cloud_client/quickstart.py

Run against a deployed Motus Cloud agent:

    export LITHOSAI_API_KEY=...   # or: motus login
    uv run python examples/cloud_client/quickstart.py https://<agent>.agent.lithosai.cloud

API key resolution mirrors the SDK: constructor argument > ``LITHOSAI_API_KEY``
env > ``~/.motus/credentials.json`` > no auth. No key is fine against a local
``motus serve`` that has auth disabled.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from motus.cloud import (
    AsyncClient,
    Client,
    MotusClientError,
    SessionNotFound,
)


def header(title: str) -> None:
    print()
    print(f"--- {title} ---")


def demo_health(c: Client) -> None:
    header("health()")
    print(c.health())


def demo_one_shot_chat(c: Client) -> None:
    header("one-shot chat()")
    t0 = time.monotonic()
    result = c.chat("Say hello.")
    elapsed = time.monotonic() - t0
    print(f"status={result.status.value}  elapsed={elapsed:.2f}s")
    print(f"session_id={result.session_id}")
    print(f"reply={result.message.content!r}")


def demo_chat_with_turn_timeout(c: Client) -> None:
    header("chat() with a turn_timeout budget")
    try:
        result = c.chat("Write one short sentence about clouds.", turn_timeout=30.0)
        print(f"completed within budget: {result.message.content!r}")
    except MotusClientError as exc:
        print(f"{type(exc).__name__}: {exc}")
        print(f"  context={exc.context}")


def demo_multi_turn_session(c: Client) -> None:
    header("multi-turn session()")
    with c.session() as s:
        r1 = s.chat("Pick any noun and say it.")
        print(f"turn 1: {r1.message.content!r}")
        r2 = s.chat("Now use it in a sentence.")
        print(f"turn 2: {r2.message.content!r}")
        print(f"session_id={s.session_id}  owned={s.owned}")
    print("session deleted on context exit")


def demo_keep_and_reattach(c: Client) -> None:
    header("session(keep=True) + reattach + manual delete")
    with c.session(keep=True) as s:
        s.chat("First turn of a kept session.")
        sid = s.session_id
        print(f"left alive: session_id={sid}")

    with c.session(session_id=sid) as s2:
        r = s2.chat("Second turn, same session.")
        print(f"reattached: {r.message.content!r}")

    c.delete_session(sid)
    print(f"manually deleted {sid}")


def demo_chat_events(c: Client) -> None:
    header("chat_events() event stream")
    for event in c.chat_events("Say one short word."):
        snap = getattr(event, "snapshot", None)
        reply = snap.response.content if snap and snap.response else None
        print(f"  event type={event.type}  reply={reply!r}")


def demo_low_level(c: Client) -> None:
    header("low-level: create_session + send_message + get_session(wait=True)")
    sid = c.create_session().session_id
    print(f"created {sid}")
    msg = c.send_message(sid, "Reply with anything short.")
    print(f"send_message accepted: {msg}")
    snap = c.get_session(sid, wait=True, timeout=60.0)
    print(f"wait=true snapshot status={snap.status.value}")
    reply = snap.response.content if snap.response else None
    print(f"reply={reply!r}")
    c.delete_session(sid)
    print(f"deleted {sid}")


def demo_typed_errors(c: Client) -> None:
    header("typed errors with ErrorContext")
    bogus = "00000000-0000-0000-0000-000000000000"
    try:
        c.get_session(bogus)
    except SessionNotFound as exc:
        print(f"{type(exc).__name__}: {exc}")
        print(f"  context.session_id={exc.context.session_id}")
        print(f"  context.method={exc.context.method}")
        print(f"  context.url={exc.context.url}")
        print(f"  context.status_code={exc.context.status_code}")


async def demo_async_parity(base_url: str, api_key: str | None) -> None:
    header("AsyncClient parity")
    async with AsyncClient(base_url=base_url, api_key=api_key) as ac:
        r = await ac.chat("Reply with one word.")
        print(f"async reply={r.message.content!r}  status={r.status.value}")


SYNC_DEMOS = [
    demo_health,
    demo_one_shot_chat,
    demo_chat_with_turn_timeout,
    demo_multi_turn_session,
    demo_keep_and_reattach,
    demo_chat_events,
    demo_low_level,
    demo_typed_errors,
]


def main(base_url: str, api_key: str | None = None) -> int:
    api_key = api_key or os.environ.get("LITHOSAI_API_KEY")

    with Client(base_url=base_url, api_key=api_key) as c:
        for demo in SYNC_DEMOS:
            try:
                demo(c)
            except MotusClientError as exc:
                print(f"[{demo.__name__}] {type(exc).__name__}: {exc}")
                print(f"  context={exc.context}")
                return 1

    try:
        asyncio.run(demo_async_parity(base_url, api_key))
    except MotusClientError as exc:
        print(f"[demo_async_parity] {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("MOTUS_BASE_URL", "http://localhost:8000")
    )
    sys.exit(main(url))
