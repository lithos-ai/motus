"""
Deep Research using Google ADK.

Flow: Plan → Research (parallel) → Compress → Report

Usage:
    motus serve start deep_research.google_adk.serve:agent
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
from motus.google_adk.agents.llm_agent import Agent
from motus.models import ChatMessage
from motus.tools.providers.brave.tool_provider import WebSearch

# ── Config ────────────────────────────────────────────────────────

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
MAX_CONCURRENT_RESEARCH = 3

_web_search = WebSearch(BRAVE_API_KEY)
_today = get_today_str()


# ── Tool (ADK expects plain async functions) ─────────────────────


async def web_search(query: str) -> list[dict]:
    """Search the web for information. Returns results with title, url, description.

    Args:
        query: Search query string. Be specific for better results.
    """
    if not BRAVE_API_KEY:
        return [{"error": "BRAVE_API_KEY not set"}]
    return await _web_search(query)


# ── Sub-agents ────────────────────────────────────────────────────

planner = Agent(
    model=MODEL,
    name="planner",
    instruction="You are a research planner. Follow the user's instructions to create a research plan.",
    tools=[],
)
researcher = Agent(
    model=MODEL,
    name="researcher",
    instruction=RESEARCH_SYSTEM_PROMPT.format(date=_today),
    tools=[web_search],
)
synthesizer = Agent(
    model=MODEL,
    name="synthesizer",
    instruction=COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=_today),
    tools=[],
)
report_writer = Agent(
    model=MODEL,
    name="report_writer",
    instruction="You write research reports. Follow the user's instructions exactly.",
    tools=[],
)


# ── Helpers ───────────────────────────────────────────────────────


async def _run(agent: Agent, prompt: str) -> str:
    msg = ChatMessage.user_message(content=prompt)
    response, _ = await agent.run_turn(msg, [])
    return response.content or ""


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
