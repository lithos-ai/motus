"""
Deep Research - Main researcher implementation.

This module contains the core research logic including:
- ReAct-style iterative research loops
- Parallel research execution
- Research compression
- Final report generation
"""

import json
import logging
from typing import Any, Dict, List

import colorlog

from deep_research.common.prompts import (
    CLARIFICATION_PROMPT,
    COMPRESS_RESEARCH_SYSTEM_PROMPT,
    COMPRESS_RESEARCH_USER_MESSAGE,
    FINAL_REPORT_PROMPT,
    PLANNING_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from deep_research.common.utils import get_today_str
from motus.agent import ReActAgent
from motus.agent.tasks import model_serve_task
from motus.memory import CompactionMemory, CompactionMemoryConfig
from motus.models import BaseChatClient, ChatMessage
from motus.runtime.agent_task import agent_task
from motus.runtime.hooks import register_hook
from motus.tools import Tools

from .config import (
    ENABLE_RESEARCH_COMPRESSION,
    MAX_CONCURRENT_RESEARCH,
    MAX_REACT_ITERATIONS,
)
from .elicitation import question
from .schemas import ResearchPlan

# Logging setup
_handler = colorlog.StreamHandler()
_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s%(reset)s:%(name)s:%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

logging.basicConfig(level=logging.DEBUG, handlers=[_handler])
logger = logging.getLogger("DeepResearch")

# Silence noisy loggers
for noisy in (
    "agent_task",
    "Message",
    "httpx",
    "httpcore",
    "mcp",
    "mcp.client.streamable_http",
    "openai",
    "asyncio",
    "urllib3",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Hooks: log phase transitions ──────────────────────────────


def _on_phase_start(event):
    logger.info(f"▶ phase started: {event.name}")


def _on_phase_end(event):
    logger.info(f"✓ phase completed: {event.name}")


def _on_phase_error(event):
    logger.error(f"✗ phase failed: {event.name} — {event.error}")


register_hook("task_start", _on_phase_start)
register_hook("task_end", _on_phase_end)
register_hook("task_error", _on_phase_error)


# Agent tasks


@agent_task
def clarification_task(
    client: BaseChatClient, model: str, question_str: str
) -> str | None:
    """
    Get clarification question from the question.
    """
    logger.debug(f"Begin clarification task: {question_str}")

    prompt = CLARIFICATION_PROMPT.format(question=question_str)
    messages = [ChatMessage(role="user", content=prompt)]

    completion = model_serve_task(client, model, messages, None).af_result()
    assistant_msg = completion.to_message()

    logger.debug(f"Finish clarification task, assistant_msg: {assistant_msg}")
    return assistant_msg.content


@agent_task
async def user_answer_task(clarification_question: str) -> str:
    """Get user's answer to clarification questions."""
    return await question(f"\n{clarification_question}\nYour answer: ")


@agent_task
def planning_task(
    client: BaseChatClient, model: str, question: str, user_answer: str
) -> List[Dict[str, Any]]:
    """
    Generate research plan with topics and queries using structured tool output.
    """
    logger.debug("Begin planning task")

    prompt = PLANNING_PROMPT.format(question=question, user_answer=user_answer)
    messages = [ChatMessage(role="user", content=prompt)]

    completion = model_serve_task(
        client, model, messages=messages, tools=None, response_format=ResearchPlan
    ).af_result()

    # If user refused to provide more information, return a default plan
    if hasattr(completion, "refusal") and completion.refusal:
        logger.warning(f"Planning refused: {completion.refusal}")
        return [{"topic": "General overview", "query": user_answer or question}]

    # If the plan is not valid, return a default plan
    plan: ResearchPlan = completion.parsed
    if not plan or not plan.topics:
        return [{"topic": "General overview", "query": user_answer or question}]

    result: List[Dict[str, Any]] = []
    for topic in plan.topics:
        if topic.search_queries:
            for query in topic.search_queries:
                result.append({"topic": topic.name, "query": query})
        else:
            result.append({"topic": topic.name, "query": topic.name})

    logger.debug(f"Finish planning task: {len(result)} topics")
    return result


@agent_task
def research_task(
    client: BaseChatClient,
    model: str,
    topic: str,
    query: str,
    tools: Tools | None = None,
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> str:
    """
    Research task using ReActAgent.

    Creates a ReActAgent with the research system prompt and tools,
    runs it on the query, and returns the final response.
    """
    logger.info(f"Begin ReAct research: topic={topic!r}, query={query!r}")

    memory = CompactionMemory(
        config=CompactionMemoryConfig(safety_ratio=0.75),
        on_compact=lambda info: logger.info(
            f"Memory compacted for topic={topic!r}: "
            f"{info.get('old_count', '?')} msgs → {info.get('new_count', '?')}"
        ),
    )

    agent = ReActAgent(
        client=client,
        model_name=model,
        system_prompt=RESEARCH_SYSTEM_PROMPT.format(date=get_today_str()),
        tools=tools,
        memory=memory,
        max_steps=max_iterations,
    )

    user_prompt = (
        f"Research Topic: {topic}\n\nResearch Query: {query}\n\n"
        f"Please use the available tools to gather comprehensive information."
    )

    result = agent(user_prompt).af_result()

    logger.info(f"Finish ReAct research: topic={topic!r}")
    return result or ""


@agent_task
def compress_research_task(
    client: BaseChatClient,
    model: str,
    findings: List[str],
) -> str:
    """
    Compress and synthesize research findings into a clean summary.
    """
    logger.debug(f"Begin compress research: {len(findings)} findings")

    findings_text = "\n\n---\n\n".join(findings)

    system_content = COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())
    user_content = f"{findings_text}\n\n{COMPRESS_RESEARCH_USER_MESSAGE}"

    messages = [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]

    completion = model_serve_task(client, model, messages, None).af_result()
    assistant_msg = completion.to_message()

    logger.debug("Finish compress research")
    return assistant_msg.content or ""


@agent_task
def final_report_task(
    client: BaseChatClient,
    model: str,
    question: str,
    user_context: str,
    compressed_findings: str,
) -> str:
    """
    Generate the final comprehensive research report.
    """
    logger.debug("Begin final report generation")

    prompt = FINAL_REPORT_PROMPT.format(
        date=get_today_str(),
        question=question,
        user_context=user_context,
        findings=compressed_findings,
    )

    messages = [ChatMessage(role="user", content=prompt)]
    completion = model_serve_task(client, model, messages, None).af_result()
    assistant_msg = completion.to_message()

    logger.debug("Finish final report generation")
    return assistant_msg.content or ""


# Main workflow


async def deep_research(
    client: BaseChatClient,
    model: str,
    question: str,
    tools: Tools,
    user_context: str = "No additional context provided by user.",
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> str:
    """
    Non-interactive deep research pipeline: planning → research → compress → report.

    Returns:
        The final research report as a string.
    """
    planning_results = await planning_task(client, model, question, user_context)
    logger.info(f"Planning completed: {len(planning_results)} topics")

    findings = []
    for i in range(0, len(planning_results), MAX_CONCURRENT_RESEARCH):
        batch = planning_results[i : i + MAX_CONCURRENT_RESEARCH]
        futures = [
            research_task(
                client, model, pr["topic"], pr["query"], tools, max_iterations
            )
            for pr in batch
        ]
        for f in futures:
            findings.append(await f)

    findings_json = json.dumps(findings)
    if len(findings_json) > 100000:
        compressed_findings = findings_json[:100000] + "...[truncated]"
    elif ENABLE_RESEARCH_COMPRESSION:
        compressed_findings = await compress_research_task(client, model, findings)
    else:
        compressed_findings = json.dumps(findings, indent=2)

    report = await final_report_task(
        client, model, question, user_context, compressed_findings
    )
    return report


async def deep_research_workflow(
    client: BaseChatClient,
    model: str,
    question_str: str,
    tools: Tools,
    max_iterations: int = MAX_REACT_ITERATIONS,
) -> str:
    """
    Interactive deep research workflow.

    Adds a clarification step before delegating to deep_research().

    Flow:
    1. Clarification - Ask user for more context
    2-5. Planning → Research → Compression → Report (via deep_research)
    """
    # Step 1: Clarification
    clarification_future = clarification_task(client, model, question_str)
    user_answer = await user_answer_task(clarification_future)
    clarification_text = await clarification_future
    user_context = f"Clarification Q: {clarification_text}\nUser Answer: {user_answer}"

    # Steps 2-5: Core pipeline
    return await deep_research(
        client=client,
        model=model,
        question=question_str,
        tools=tools,
        user_context=user_context,
        max_iterations=max_iterations,
    )
