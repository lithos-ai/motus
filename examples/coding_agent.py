"""CodingAgent example.

Uses the bundled coding-agent template — preconfigured with bash, file,
search, web, todo, plan-mode, and subagent tools, plus a system prompt
and ``AGENTS.md`` / ``CLAUDE.md`` auto-injection.

Usage::

    # Interactive REPL (default — chat back and forth):
    python examples/coding_agent.py

    # One-shot with a prompt:
    python examples/coding_agent.py "what's in this repo?"

    # With real-time trace viewer in the browser:
    MOTUS_TRACING_ONLINE=1 python examples/coding_agent.py

The agent runs on the local shell, so its actions affect your real
filesystem. For sandboxed runs, pass an explicit ``sandbox=`` (e.g. a
``DockerSandbox``).
"""

import asyncio
import os
import pathlib
import sys

from dotenv import load_dotenv

from motus.agent import CodingAgent
from motus.models import AnthropicChatClient

# Load .env at the motus repo root (sibling of this file's parent).
load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")


def _make_agent() -> CodingAgent:
    return CodingAgent(
        client=AnthropicChatClient(),
        model_name="claude-sonnet-4-6",
    )


async def _one_shot(prompt: str) -> None:
    agent = _make_agent()
    result = await agent(prompt)
    print(result)


async def _repl() -> None:
    agent = _make_agent()
    print("CodingAgent ready. Type a request, or 'quit' / Ctrl-D to exit.\n")
    while True:
        try:
            user_input = await asyncio.to_thread(input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        if not user_input.strip():
            continue
        try:
            result = await agent(user_input)
        except Exception as e:
            print(f"[agent error] {e}\n")
            continue
        print(f"\nagent> {result}\n")


def main() -> None:
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(_one_shot(prompt))
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "ANTHROPIC_API_KEY is not set. Either export it, or put it "
                "in a .env file at the repo root and ensure it's loaded."
            )
            return
        asyncio.run(_repl())


if __name__ == "__main__":
    main()
