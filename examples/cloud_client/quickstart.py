"""Quickstart for motus.cloud.Client.

Run against a local ``motus serve`` instance:

    motus serve start tests.unit.serve.mock_agent:echo_agent --port 8000
    uv run python examples/cloud_client/quickstart.py

Or against a deployed Motus Cloud agent by exporting ``LITHOSAI_API_KEY`` and
passing that agent's URL.
"""

from __future__ import annotations

import os
import sys

from motus.cloud import AgentError, Client, MotusClientError, SessionTimeout


def main(base_url: str) -> int:
    with Client(base_url=base_url) as client:
        try:
            single = client.chat("hello")
        except SessionTimeout as e:
            print(f"turn deadline exceeded (session={e.session_id})")
            return 1
        except AgentError as e:
            print(f"agent error: {e} (session={e.session_id})")
            return 1
        except MotusClientError as e:
            print(f"client error: {type(e).__name__}: {e}")
            return 1

        if single.message is not None:
            print("one-shot:", single.message.content)

        with client.session() as s:
            a = s.chat("what is your name?")
            b = s.chat("what can you do?")
            print("multi-turn[1]:", a.message.content if a.message else "")
            print("multi-turn[2]:", b.message.content if b.message else "")

    return 0


if __name__ == "__main__":
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("MOTUS_BASE_URL", "http://localhost:8000")
    )
    sys.exit(main(url))
