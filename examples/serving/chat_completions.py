"""Chat completions agent — uses OpenAI-compatible /v1/chat/completions.

Demonstrates a ReActAgent with the OpenAI Chat Completions API. On Motus
cloud, OPENAI_BASE_URL and OPENAI_API_KEY are auto-wired to the model
proxy, so no secrets are needed.

Deploy: motus serve start examples.serving.chat_completions:agent
"""

from motus.agent import ReActAgent
from motus.models import OpenAIChatClient

agent = ReActAgent(
    client=OpenAIChatClient(),
    model_name="gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
)
