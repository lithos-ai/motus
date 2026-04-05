"""
Deep Research using Claude Agent SDK (claude_code_sdk).

Flow: Plan → Research (parallel) → Compress → Report

Usage:
    python -m deep_research.claude "What is quantum computing?"

Note: Claude Agent SDK uses Claude Code as the execution engine with built-in
tool access (WebSearch, WebFetch, etc.) — no Brave API key needed.
"""

import asyncio
import json

from deep_research.common.prompts import (
    COMPRESS_RESEARCH_SYSTEM_PROMPT,
    COMPRESS_RESEARCH_USER_MESSAGE,
    FINAL_REPORT_PROMPT,
    PLANNING_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from deep_research.common.utils import get_today_str
from motus.claude_agent import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

# ── Config ────────────────────────────────────────────────────────

MAX_CONCURRENT_RESEARCH = 3
_today = get_today_str()


# ── Sub-agents ────────────────────────────────────────────────────

AGENTS = {
    "planner": AgentDefinition(
        description="Research planner",
        prompt="You are a research planner. Follow the user's instructions to create a research plan.",
        tools=[],
        model="haiku",
    ),
    "researcher": AgentDefinition(
        description="Web researcher",
        prompt=RESEARCH_SYSTEM_PROMPT.format(date=_today),
        tools=["WebSearch", "WebFetch"],
        model="sonnet",
    ),
    "synthesizer": AgentDefinition(
        description="Research synthesizer",
        prompt=COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=_today),
        tools=[],
        model="sonnet",
    ),
    "report_writer": AgentDefinition(
        description="Report writer",
        prompt="You write research reports. Follow the user's instructions exactly.",
        tools=[],
        model="sonnet",
    ),
}


# ── Helpers ───────────────────────────────────────────────────────


async def _run(agents: dict[str, AgentDefinition], prompt: str) -> str:
    name = next(iter(agents))
    options = ClaudeAgentOptions(agents=agents, max_turns=10)
    parts = []
    async for message in query(
        prompt=f"Use the {name} agent to: {prompt}", options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    return "\n".join(parts)


# ── Pipeline ──────────────────────────────────────────────────────


async def deep_research(
    question: str, user_context: str = "No additional context provided by user."
) -> str:
    # Step 1: Plan
    plan_text = await _run(
        {"planner": AGENTS["planner"]},
        PLANNING_PROMPT.format(question=question, user_answer=user_context),
    )

    try:
        plan = json.loads(plan_text[plan_text.index("{") : plan_text.rindex("}") + 1])
        topic_names = plan.get("topics", []) or ["General research"]
        queries = plan.get("search_queries", topic_names) or [question]
    except (json.JSONDecodeError, ValueError):
        topic_names = ["General research"]
        queries = [question]

    # Step 2: Research (parallel, batched)
    findings = []
    items = []
    for i, q in enumerate(queries):
        topic = topic_names[i] if i < len(topic_names) else topic_names[-1]
        items.append(
            f"Research Topic: {topic}\n\nResearch Query: {q}\n\nPlease use the available tools to gather comprehensive information."
        )

    for i in range(0, len(items), MAX_CONCURRENT_RESEARCH):
        batch = items[i : i + MAX_CONCURRENT_RESEARCH]
        results = await asyncio.gather(
            *[_run({"researcher": AGENTS["researcher"]}, p) for p in batch]
        )
        findings.extend(results)

    # Step 3: Compress
    findings_text = "\n\n---\n\n".join(findings)
    compressed = await _run(
        {"synthesizer": AGENTS["synthesizer"]},
        f"{findings_text}\n\n{COMPRESS_RESEARCH_USER_MESSAGE}",
    )

    # Step 4: Report
    report = await _run(
        {"report_writer": AGENTS["report_writer"]},
        FINAL_REPORT_PROMPT.format(
            date=_today,
            question=question,
            user_context=user_context,
            findings=compressed,
        ),
    )
    return report
