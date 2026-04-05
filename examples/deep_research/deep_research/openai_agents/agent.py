"""
Deep Research using OpenAI Agents SDK.

Flow: Plan → Research (parallel) → Compress → Report

Usage:
    motus serve start deep_research.openai_agents.serve:agent
"""

import asyncio
import json
import os

from deep_research.common.prompts import (
    COMPRESS_RESEARCH_SYSTEM_PROMPT,
    COMPRESS_RESEARCH_USER_MESSAGE,
    FINAL_REPORT_PROMPT,
    PLANNING_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from deep_research.common.utils import get_today_str
from motus.openai_agents import Agent, Runner, function_tool
from motus.tools.providers.brave.tool_provider import WebSearch

# ── Config ────────────────────────────────────────────────────────

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_CONCURRENT_RESEARCH = 3

_web_search = WebSearch(BRAVE_API_KEY)
_today = get_today_str()


# ── Tool (OAI needs @function_tool + str return) ─────────────────


@function_tool
async def web_search(query: str) -> str:
    """Search the web for information. Returns JSON results with title, url, description.

    Args:
        query: Search query string. Be specific for better results.
    """
    if not BRAVE_API_KEY:
        return json.dumps([{"error": "BRAVE_API_KEY not set"}])
    results = await _web_search(query)
    return json.dumps(results)


# ── Sub-agents ────────────────────────────────────────────────────

planner = Agent(
    name="planner",
    model=MODEL,
    instructions="You are a research planner. Follow the user's instructions to create a research plan.",
)
researcher = Agent(
    name="researcher",
    model=MODEL,
    instructions=RESEARCH_SYSTEM_PROMPT.format(date=_today),
    tools=[web_search],
)
synthesizer = Agent(
    name="synthesizer",
    model=MODEL,
    instructions=COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=_today),
)
report_writer = Agent(
    name="report_writer",
    model=MODEL,
    instructions="You write research reports. Follow the user's instructions exactly.",
)


# ── Helpers ───────────────────────────────────────────────────────


async def _run(a: Agent, prompt: str) -> str:
    result = await Runner.run(a, prompt, max_turns=25)
    return result.final_output or ""


# ── Pipeline ──────────────────────────────────────────────────────


async def deep_research(
    question: str, user_context: str = "No additional context provided by user."
) -> str:
    # Step 1: Plan
    plan_text = await _run(
        planner, PLANNING_PROMPT.format(question=question, user_answer=user_context)
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
        results = await asyncio.gather(*[_run(researcher, p) for p in batch])
        findings.extend(results)

    # Step 3: Compress
    findings_text = "\n\n---\n\n".join(findings)
    compressed = await _run(
        synthesizer, f"{findings_text}\n\n{COMPRESS_RESEARCH_USER_MESSAGE}"
    )

    # Step 4: Report
    report = await _run(
        report_writer,
        FINAL_REPORT_PROMPT.format(
            date=_today,
            question=question,
            user_context=user_context,
            findings=compressed,
        ),
    )
    return report
