"""Multi-agent — root agent delegates to specialized sub-agents.

The root agent triages user requests to either a math agent or a creative
writing agent. Each sub-agent has its own tools and instructions. The ADK
runner handles delegation and response routing automatically.

Deploy: motus serve start examples.google_adk.multi_agent:root_agent
"""

from google.adk.agents.llm_agent import Agent as _ADKAgent

from motus.google_adk.agents.llm_agent import Agent


def calculate(expression: str) -> dict:
    """Evaluate a math expression safely.

    Args:
        expression: A Python math expression to evaluate (e.g. "2 + 3 * 4").
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


def roll_die(sides: int = 6) -> dict:
    """Roll a die with the given number of sides.

    Args:
        sides: Number of sides on the die (default 6).
    """
    import random

    return {"result": random.randint(1, sides), "sides": sides}


def count_words(text: str) -> dict:
    """Count words in a text.

    Args:
        text: The text to count words in.
    """
    return {"word_count": len(text.split()), "char_count": len(text)}


# Sub-agents (vanilla ADK Agent — no run_turn needed for sub-agents)
math_agent = _ADKAgent(
    model="gemini-2.5-flash",
    name="math_agent",
    description="Handles math calculations, dice rolls, and number questions.",
    instruction=(
        "You are a math specialist. Use the calculate tool for arithmetic "
        "and the roll_die tool for random number generation. Show your work."
    ),
    tools=[calculate, roll_die],
)

writing_agent = _ADKAgent(
    model="gemini-2.5-flash",
    name="writing_agent",
    description="Handles creative writing, text analysis, and word tasks.",
    instruction=(
        "You are a creative writing assistant. Help with poems, stories, "
        "and text analysis. Use count_words when asked about text length."
    ),
    tools=[count_words],
)

# Root agent — motus Agent with run_turn for serve compatibility
root_agent = Agent(
    model="gemini-2.5-flash",
    name="triage_agent",
    description="Routes requests to math or writing specialists.",
    instruction=(
        "You are a triage agent. Analyze the user's request and delegate to "
        "the appropriate specialist:\n"
        "- math_agent: for calculations, dice rolls, number questions\n"
        "- writing_agent: for creative writing, poems, text analysis\n"
        "If the request doesn't fit either, answer directly."
    ),
    sub_agents=[math_agent, writing_agent],
)
