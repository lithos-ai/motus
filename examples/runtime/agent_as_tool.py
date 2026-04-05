"""
Agent-as-Tool: Using one agent as a tool for another
=====================================================

Demonstrates three patterns for nesting agents:

1. ``agent.as_tool()`` — convenience method
2. ``AgentTool(agent, ...)`` — explicit wrapper
3. Direct agent in tool list — ``normalize_tools`` auto-wraps

An **orchestrator** agent delegates to two specialist sub-agents:
- **researcher**: given a topic, returns a short research summary
- **writer**: given a topic + notes, produces a polished paragraph

The orchestrator decides which sub-agent to call and composes
the final answer.

Usage:
    OPENROUTER_API_KEY=sk-... python examples/agent_as_tool.py
    OPENAI_API_KEY=sk-... OPENROUTER_BASE_URL=https://api.openai.com/v1 MODEL=gpt-4o-mini python examples/agent_as_tool.py
"""

import logging
import os

from dotenv import load_dotenv

from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import AgentTool, tool

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARN").upper())

MODEL = os.getenv("MODEL", "google/gemini-2.5-flash")


def make_client() -> OpenAIChatClient:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY")
    return OpenAIChatClient(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


# ---------------------------------------------------------------------------
# Simple function tool — gives the orchestrator a basic capability
# ---------------------------------------------------------------------------


@tool
async def get_current_year() -> str:
    """Return the current year."""
    import datetime

    return str(datetime.date.today().year)


# ---------------------------------------------------------------------------
# Sub-agent 1: Researcher
# ---------------------------------------------------------------------------

client = make_client()

researcher = ReActAgent(
    client=client,
    model_name=MODEL,
    name="researcher",
    system_prompt=(
        "You are a research assistant. When given a topic, provide a concise "
        "factual summary (3-5 bullet points). Be specific and include numbers "
        "or dates where relevant. Do NOT write prose — only bullet points."
    ),
    tools=[get_current_year],
    max_steps=5,
)

# ---------------------------------------------------------------------------
# Sub-agent 2: Writer
# ---------------------------------------------------------------------------

writer = ReActAgent(
    client=client,
    model_name=MODEL,
    name="writer",
    system_prompt=(
        "You are a professional writer. Given a topic and research notes, "
        "write a single polished paragraph (60-100 words). "
        "Be engaging and informative. Do NOT use bullet points."
    ),
    max_steps=3,
)

# ---------------------------------------------------------------------------
# Orchestrator: uses sub-agents as tools
# ---------------------------------------------------------------------------

orchestrator = ReActAgent(
    client=client,
    model_name=MODEL,
    name="orchestrator",
    system_prompt=(
        "You are an orchestrator that produces well-researched articles. "
        "For each user request:\n"
        "1. Call the 'researcher' tool to gather facts about the topic.\n"
        "2. Call the 'writer' tool, passing both the topic and the research "
        "notes, to produce a polished paragraph.\n"
        "3. Return the writer's output as your final answer.\n"
        "Always use both tools. Do not write content yourself."
    ),
    tools=[
        # Pattern 1: as_tool() convenience method
        researcher.as_tool(
            name="researcher",
            description="Research a topic. Input: a topic string. Output: bullet-point facts.",
        ),
        # Pattern 2: AgentTool explicit wrapper
        AgentTool(
            writer,
            name="writer",
            description=(
                "Write a polished paragraph. Input: a topic plus research notes. "
                "Output: a well-written paragraph."
            ),
        ),
    ],
    max_steps=10,
)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def main():
    print("=== Agent-as-Tool Demo ===")
    print(f"Model: {MODEL}")
    print("The orchestrator delegates to 'researcher' and 'writer' sub-agents.")
    print("Type /quit to exit.\n")

    while True:
        try:
            prompt = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() == "/quit":
            break

        result = await orchestrator(prompt)
        print(f"\n[Orchestrator]: {result}\n")

    print("Goodbye!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
