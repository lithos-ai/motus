"""Chatbot — minimal conversational agent with web search.

On Motus cloud, OPENAI_BASE_URL and OPENAI_API_KEY are auto-wired to the
model proxy, so no secrets are needed for the LLM. Pass BRAVE_API_KEY as
a deploy secret for web search.

Deploy:   motus deploy chatbot.motus_native.serve:agent
Serve:    motus serve start chatbot.motus_native.serve:agent
Chat:     motus serve chat <agent-url>
"""

import os

from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import DictTools, WebSearchTool

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

agent = ReActAgent(
    client=OpenAIChatClient(),
    model_name="gpt-5.2",
    system_prompt=(
        "You are MotusBot, an AI assistant built by LithosAI and powered by "
        "the Motus agent framework. You are knowledgeable, concise, and "
        "friendly.\n\n"
        "You have access to web search — use it when the user asks about "
        "recent events, real-time data, or anything you're unsure about. "
        "Cite sources when you search.\n\n"
        "When greeting users, introduce yourself as MotusBot."
    ),
    tools=DictTools({"web_search": WebSearchTool(BRAVE_API_KEY)}),
)
