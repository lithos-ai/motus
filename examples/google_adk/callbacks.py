# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Callbacks — demonstrates all six ADK callback hooks.

Shows before/after callbacks on agent, model, and tool lifecycle. Each
callback logs when it fires. The ``before_tool_callback`` modifies tool
args, and ``after_tool_callback`` enriches the tool response.

Deploy: motus serve start examples.google_adk.callbacks:root_agent
"""

from google.adk.tools.tool_context import ToolContext

from motus.google_adk.agents.llm_agent import Agent


def roll_die(sides: int, tool_context: ToolContext) -> dict:
    """Roll a die with the given number of sides.

    Args:
        sides: Number of sides on the die.
    """
    import random

    result = random.randint(1, sides)
    tool_context.state["last_roll"] = result
    return {"roll": result, "sides": sides}


async def check_prime(n: int) -> dict:
    """Check if a number is prime.

    Args:
        n: The number to check.
    """
    if n < 2:
        return {"number": n, "is_prime": False}
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return {"number": n, "is_prime": False}
    return {"number": n, "is_prime": True}


# ── Agent callbacks ──


async def before_agent(callback_context):
    """Called before the agent processes a turn."""
    print("[callback] before_agent")
    return None  # Return Content to skip agent entirely


async def after_agent(callback_context):
    """Called after the agent finishes a turn."""
    print("[callback] after_agent")
    return None  # Return Content to override the response


# ── Model callbacks ──


async def before_model(callback_context, llm_request):
    """Called before each LLM call."""
    print(f"[callback] before_model (messages: {len(llm_request.contents)})")
    return None  # Return LlmResponse to skip the model call


async def after_model(callback_context, llm_response):
    """Called after each LLM response."""
    print("[callback] after_model")
    return None  # Return LlmResponse to override


# ── Tool callbacks ──


def before_tool(tool, args, tool_context):
    """Called before a tool executes. Can modify args."""
    print(f"[callback] before_tool: {tool.name}({args})")
    return None  # Return dict to skip tool and use as result


def after_tool(tool, args, tool_context, tool_response):
    """Called after a tool returns. Can modify the response."""
    print(f"[callback] after_tool: {tool.name} -> {tool_response}")
    # Enrich the response with a note
    if isinstance(tool_response, dict):
        tool_response["callback_note"] = "processed by after_tool"
    return tool_response


root_agent = Agent(
    model="gemini-2.5-flash",
    name="callback_demo_agent",
    description="Demonstrates all six ADK callback hooks.",
    instruction=(
        "You are a helpful assistant. Use the roll_die tool to roll dice "
        "and check_prime to test if numbers are prime."
    ),
    tools=[roll_die, check_prime],
    before_agent_callback=before_agent,
    after_agent_callback=after_agent,
    before_model_callback=before_model,
    after_model_callback=after_model,
    before_tool_callback=before_tool,
    after_tool_callback=after_tool,
)
