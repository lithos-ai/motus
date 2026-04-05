"""Token usage — agent that tracks and reports token consumption.

Demonstrates an agent with tools that return varying amounts of data,
showing how token usage accumulates. The tracing system captures token
counts per model call.

Deploy: motus serve start examples.google_adk.token_usage:root_agent
"""

from motus.google_adk.agents.llm_agent import Agent


def generate_text(topic: str, length: str = "short") -> dict:
    """Generate text about a topic with specified length.

    Args:
        topic: The topic to write about.
        length: One of 'short' (1-2 sentences), 'medium' (paragraph), or 'long' (multiple paragraphs).
    """
    lengths = {
        "short": f"A brief note about {topic}.",
        "medium": f"Here is a paragraph about {topic}. " * 5,
        "long": f"An extensive discussion about {topic}. " * 20,
    }
    text = lengths.get(length, lengths["short"])
    return {"topic": topic, "length": length, "text": text, "chars": len(text)}


def count_tokens_estimate(text: str) -> dict:
    """Estimate the token count for a piece of text.

    Args:
        text: The text to estimate tokens for.
    """
    # Rough estimate: ~4 chars per token
    estimated = len(text) // 4
    return {"estimated_tokens": estimated, "chars": len(text)}


root_agent = Agent(
    model="gemini-2.5-flash",
    name="token_tracker",
    description="Generates text and tracks token usage.",
    instruction=(
        "You are a text generation assistant that is mindful of token usage. "
        "When asked to generate text, use the generate_text tool. Use "
        "count_tokens_estimate to check how many tokens a response uses. "
        "Report the estimated token count to the user."
    ),
    tools=[generate_text, count_tokens_estimate],
)
