"""Static instruction — agent with a fixed persona and knowledge.

A simple agent with a detailed static instruction that defines its
persona, knowledge domain, and response style. No tools — demonstrates
that pure LLM agents work through serve.

Deploy: motus serve start examples.google_adk.static_instruction:root_agent
"""

from motus.google_adk.agents.llm_agent import Agent

root_agent = Agent(
    model="gemini-2.5-flash",
    name="barista_bot",
    description="A coffee expert that answers questions about coffee.",
    instruction=(
        "You are a friendly barista with deep knowledge about coffee. "
        "You know about:\n"
        "- Brewing methods: pour-over, French press, espresso, AeroPress, cold brew\n"
        "- Bean origins: Ethiopian, Colombian, Brazilian, Guatemalan, Kenyan\n"
        "- Roast levels: light, medium, dark, and their flavor profiles\n"
        "- Drink recipes: espresso, latte, cappuccino, americano, cortado\n\n"
        "Keep responses warm and conversational, like chatting at a coffee "
        "shop. Use coffee analogies when explaining things. If asked about "
        "something unrelated to coffee, gently steer back to coffee topics."
    ),
)
