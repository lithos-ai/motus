"""Deep Research — local server and cloud deployment.

Serve:    motus serve start deep_research.motus_native.serve:server --port 8000
Deploy:   motus deploy deep_research.motus_native.serve:server

Client example:
    # Create session
    curl -X POST http://localhost:8000/sessions

    # Send research question
    curl -X POST http://localhost:8000/sessions/{id}/messages \\
      -H "Content-Type: application/json" \\
      -d '{"role": "user", "content": "What are promising applications of agentic AI?"}'

    # Poll for result
    curl http://localhost:8000/sessions/{id}?wait=true
"""

import logging

from motus.models import ChatMessage, OpenAIChatClient
from motus.serve import AgentServer
from motus.tools import DictTools, WebSearchTool

from .config import (
    API_KEY,
    BASE_URL,
    BRAVE_API_KEY,
    MAX_REACT_ITERATIONS,
    MODEL,
)
from .researcher import deep_research

logger = logging.getLogger("deep_research.server")


async def agent(
    message: ChatMessage, state: list[ChatMessage]
) -> tuple[ChatMessage, list[ChatMessage]]:
    """Deep research agent — serve contract."""
    client = OpenAIChatClient(api_key=API_KEY, base_url=BASE_URL)
    tools = DictTools({"web_search": WebSearchTool(BRAVE_API_KEY)})

    report = await deep_research(
        client=client,
        model=MODEL,
        question=message.content,
        tools=tools,
        max_iterations=MAX_REACT_ITERATIONS,
    )

    response = ChatMessage.assistant_message(content=report)
    return response, state + [message, response]


server = AgentServer(agent, timeout=600)
