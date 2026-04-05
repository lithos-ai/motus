"""Workflow triage — conditional parallel execution with callbacks.

A root agent analyzes the user's request and sets an execution plan in
session state. A ParallelAgent runs specialized workers concurrently,
but each worker checks the plan via ``before_agent_callback`` and skips
itself if not selected. A summary agent collects the results.

Deploy: motus serve start examples.google_adk.workflow_triage:root_agent
"""

from typing import Optional

from google.adk.agents.llm_agent import Agent as _ADKAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from motus.google_adk.agents.llm_agent import Agent


def update_execution_plan(
    execution_agents: list[str], tool_context: ToolContext
) -> str:
    """Set which worker agents should execute.

    Args:
        execution_agents: List of agent names to activate (e.g. ["research_agent", "math_agent"]).
    """
    tool_context.state["execution_agents"] = execution_agents
    return f"Execution plan updated: {execution_agents}"


def _skip_if_not_planned(agent_name: str):
    """Factory for before_agent_callback that skips agents not in the plan."""

    def callback(callback_context) -> Optional[types.Content]:
        planned = callback_context.state.get("execution_agents", [])
        if agent_name not in planned:
            return types.Content(
                parts=[types.Part(text=f"[{agent_name} skipped — not in plan]")]
            )
        return None

    return callback


research_agent = _ADKAgent(
    model="gemini-2.5-flash",
    name="research_agent",
    instruction=(
        "You are a research specialist. Answer factual questions concisely. "
        "Provide key facts and cite sources when possible."
    ),
    before_agent_callback=_skip_if_not_planned("research_agent"),
    output_key="research_output",
)

math_agent = _ADKAgent(
    model="gemini-2.5-flash",
    name="math_agent",
    instruction=(
        "You are a math specialist. Solve calculations step by step. "
        "Show your work clearly."
    ),
    before_agent_callback=_skip_if_not_planned("math_agent"),
    output_key="math_output",
)

creative_agent = _ADKAgent(
    model="gemini-2.5-flash",
    name="creative_agent",
    instruction=(
        "You are a creative writing specialist. Write engaging, "
        "creative content. Keep responses concise but vivid."
    ),
    before_agent_callback=_skip_if_not_planned("creative_agent"),
    output_key="creative_output",
)

worker_pool = ParallelAgent(
    name="worker_pool",
    sub_agents=[research_agent, math_agent, creative_agent],
)


def _summary_instruction(readonly_context) -> str:
    """Build summary instruction from whatever outputs are available."""
    agents = readonly_context.state.get("execution_agents", [])
    parts = ["Summarize the results from the worker agents.\n"]
    for name in agents:
        key = f"{name}_output"
        output = readonly_context.state.get(key, "(no output)")
        parts.append(f"{name}: {output}")
    parts.append("\nPresent a unified response to the user.")
    return "\n".join(parts)


summary_agent = _ADKAgent(
    model="gemini-2.5-flash",
    name="summary_agent",
    instruction=_summary_instruction,
    include_contents="none",
)

execution_pipeline = SequentialAgent(
    name="execution_pipeline",
    sub_agents=[worker_pool, summary_agent],
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="triage_manager",
    description="Analyzes requests and delegates to specialized workers.",
    instruction=(
        "You are a triage manager. Analyze the user's request and decide "
        "which specialist agents are needed:\n"
        "- research_agent: for factual questions, lookups, comparisons\n"
        "- math_agent: for calculations, equations, number problems\n"
        "- creative_agent: for writing, stories, poems, creative tasks\n\n"
        "Call update_execution_plan with the list of agents to activate, "
        "then transfer to execution_pipeline."
    ),
    tools=[update_execution_plan],
    sub_agents=[execution_pipeline],
)
