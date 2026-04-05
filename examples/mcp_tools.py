"""
MCP integration examples — seven patterns for using get_mcp() with Motus.

Usage:
    python examples/mcp_tools.py <pattern>

Patterns:
    lazy              Agent manages lifecycle (lazy connect)
    context_manager   User manages lifecycle (async with)
    agent_task_cm     async with inside @agent_task
    agent_task_lazy   @agent_task creates Agent with lazy MCP
    guardrails        Prefix, blocklist, and input guardrails
    sandbox           Docker sandbox + MCP
    remote            Remote HTTP MCP server (Jina AI)

Prerequisites:
    - Node.js / npx (for @modelcontextprotocol/server-filesystem)
    - OPENROUTER_API_KEY or OPENAI_API_KEY
    - Docker Desktop (sandbox pattern only)
    - JINA_API_KEY (remote pattern only)

Examples:
    OPENROUTER_API_KEY=sk-... python examples/mcp.py lazy
    OPENROUTER_API_KEY=sk-... python examples/mcp.py guardrails
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.runtime.agent_task import agent_task
from motus.tools import get_mcp, tools

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARN").upper())

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

ALLOWED_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL = os.getenv("MODEL", "google/gemini-2.5-flash")
SYSTEM_PROMPT = f"You are a helpful coding assistant.\nAllowed directory: {ALLOWED_DIR}"

MCP_COMMAND = "npx"
MCP_ARGS = ["-y", "@modelcontextprotocol/server-filesystem", ALLOWED_DIR]


def make_client() -> OpenAIChatClient:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY")
    return OpenAIChatClient(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def _chat_loop(agent, label):
    """Synchronous input loop — calls agent().result()."""
    print(f"\n{label}")
    print("Type /quit to exit.\n")
    while True:
        try:
            prompt = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() == "/quit":
            break
        result = agent(prompt).result()
        print(f"\n[Agent]: {result}\n")
    print("Goodbye!")


async def _async_chat_loop(agent, label):
    """Async input loop."""
    print(f"\n{label}")
    print("Type /quit to exit.\n")
    while True:
        try:
            prompt = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() == "/quit":
            break
        try:
            result = await agent(prompt)
            print(f"\n[Agent]: {result}\n")
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n(interrupted)")
            break
    print("Goodbye!")


# ---------------------------------------------------------------------------
# Pattern 1: Agent manages lifecycle (lazy connect)
# ---------------------------------------------------------------------------


def lazy():
    """Pass an unconnected get_mcp() session to the Agent.

    The Agent connects it automatically when _execute() is first called.
    Cleanup: agent.tools.close() or let DictTools.__del__ handle it.
    """
    client = make_client()
    session = get_mcp(command=MCP_COMMAND, args=MCP_ARGS)
    agent = ReActAgent(
        client=client,
        model_name=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[session],
    )
    _chat_loop(
        agent, f"MCP filesystem agent (lazy connect). Allowed dir: {ALLOWED_DIR}"
    )


# ---------------------------------------------------------------------------
# Pattern 2: User manages lifecycle (async with)
# ---------------------------------------------------------------------------


async def context_manager():
    """Use async with get_mcp(...) to connect before passing to Agent.

    The session is already connected when the Agent receives it.
    Cleanup: __aexit__ closes the MCP server gracefully.
    """
    client = make_client()
    async with get_mcp(command=MCP_COMMAND, args=MCP_ARGS) as session:
        agent = ReActAgent(
            client=client,
            model_name=MODEL,
            system_prompt=SYSTEM_PROMPT,
            tools=[session],
        )
        print(f"Available tools: {', '.join(agent.tools or [])}")
        await _async_chat_loop(
            agent, f"MCP filesystem agent (context manager). Allowed dir: {ALLOWED_DIR}"
        )


# ---------------------------------------------------------------------------
# Pattern 3: async with inside @agent_task
# ---------------------------------------------------------------------------


def agent_task_cm():
    """Use async with get_mcp(...) inside an @agent_task.

    Since @agent_task runs on the AgentEngine loop, __aenter__ detects
    the same loop and awaits the connection directly.
    """

    @agent_task
    async def run_with_mcp(question: str) -> str:
        client = make_client()
        async with get_mcp(command=MCP_COMMAND, args=MCP_ARGS) as session:
            agent = ReActAgent(
                client=client,
                model_name=MODEL,
                system_prompt=SYSTEM_PROMPT,
                tools=[session],
            )
            return await agent(question)

    future = run_with_mcp("List all Python files in your allowed directory.")
    print(future.result())


# ---------------------------------------------------------------------------
# Pattern 4: @agent_task creates Agent with lazy MCP
# ---------------------------------------------------------------------------


def agent_task_lazy():
    """Build an Agent with lazy MCP inside an @agent_task, then run it.

    _execute() lazily connects the MCP session before the first _run().
    """

    @agent_task
    async def ask_filesystem(question: str) -> str:
        client = make_client()
        session = get_mcp(command=MCP_COMMAND, args=MCP_ARGS)
        agent = ReActAgent(
            client=client,
            model_name=MODEL,
            system_prompt=SYSTEM_PROMPT,
            tools=[session],
        )
        result = await agent(question)
        agent.tools.close()
        return result

    future = ask_filesystem(
        "What files are in your allowed directory? Summarize them briefly."
    )
    print(future.result())


# ---------------------------------------------------------------------------
# Pattern 5: Prefix, blocklist, and guardrails
# ---------------------------------------------------------------------------


async def guardrails():
    """Wrap an MCPSession with tools(session, prefix=..., ...) to:

    - Add a prefix to every tool name (avoid collisions)
    - Filter tools via blocklist (remove write tools entirely)
    - Attach input guardrails to validate tool arguments
    """

    def block_underscore_files(path: str = ""):
        if path:
            basename = os.path.basename(path)
            if basename.startswith("_"):
                raise ValueError(
                    f"Access denied: files starting with '_' are private ({basename})"
                )

    client = make_client()
    async with get_mcp(command=MCP_COMMAND, args=MCP_ARGS) as session:
        wrapped = tools(
            session,
            prefix="fs_",
            blocklist={"write_file", "create_directory", "move_file", "edit_file"},
            input_guardrails=[block_underscore_files],
        )
        agent = ReActAgent(
            client=client,
            model_name=MODEL,
            system_prompt=f"{SYSTEM_PROMPT}\nAll tool names are prefixed with 'fs_'.",
            tools=wrapped,
        )
        print(f"Available tools: {', '.join(agent.tools or [])}")
        print("(write_file, create_directory, move_file, edit_file are blocked)")
        print("\nTry: 'List all tools available to you.'       → no write tools")
        print("Try: 'Read _common.py'                         → blocked by guardrail")
        print("Try: 'List files in this directory'             → works")
        await _async_chat_loop(agent, "MCP with prefix + guardrails")


# ---------------------------------------------------------------------------
# Pattern 6: Docker sandbox + MCP
# ---------------------------------------------------------------------------


async def sandbox():
    """Use get_mcp(image=...) to run an MCP server inside a Docker sandbox.

    Requires Docker Desktop running.
    """
    from motus.tools import get_sandbox

    client = make_client()
    sys_prompt = (
        "You are a helpful assistant with tools from the MCP 'Everything' test "
        "server running in a Docker container. Use the tools to answer questions."
    )

    # Auto sandbox + manual close
    print("=== Auto sandbox + manual close ===\n")
    session = get_mcp(
        image="node:20",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything", "streamableHttp"],
        env={"PORT": "3000"},
        port=3000,
    )
    agent = ReActAgent(
        client=client, model_name=MODEL, system_prompt=sys_prompt, tools=[session]
    )
    result = agent("Use the echo tool to echo 'Hello from sandbox!'")
    print(f"[Agent]: {result.result()}\n")
    session.close()

    # Explicit sandbox + async with
    print("=== Explicit sandbox + async with ===\n")
    with get_sandbox(image="node:20", ports={3000: None}) as sb:
        async with get_mcp(
            sandbox=sb,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything", "streamableHttp"],
            env={"PORT": "3000"},
            port=3000,
        ) as session:
            agent = ReActAgent(
                client=client,
                model_name=MODEL,
                system_prompt=sys_prompt,
                tools=[session, sb],
            )
            result = await agent("Use the echo tool to echo 'Hello from sandbox!'")
            print(f"[Agent]: {result}\n")

    print("Done!")


# ---------------------------------------------------------------------------
# Pattern 7: Remote HTTP MCP server (Jina AI)
# ---------------------------------------------------------------------------


async def remote():
    """Connect to a remote MCP server over HTTP using get_mcp(url=...).

    Requires JINA_API_KEY.
    """
    jina_url = "https://mcp.jina.ai/v1"
    jina_key = os.getenv("JINA_API_KEY", "")
    client = make_client()

    headers = {"Authorization": f"Bearer {jina_key}"} if jina_key else None
    async with get_mcp(url=jina_url, headers=headers) as session:
        wrapped = tools(
            session,
            blocklist={
                "parallel_search_web",
                "parallel_search_arxiv",
                "parallel_search_ssrn",
                "parallel_read_url",
                "search_jina_blog",
                "show_api_key",
                "deduplicate_strings",
                "deduplicate_images",
            },
        )
        agent = ReActAgent(
            client=client,
            model_name=MODEL,
            system_prompt=(
                "You are a research assistant with web search and reading capabilities. "
                "Use search_web to find information, and read_url to read web pages. "
                "Give concise answers with sources."
            ),
            tools=wrapped,
        )
        print(f"Connected to Jina AI MCP server: {jina_url}")
        print(f"Available tools: {', '.join(agent.tools or [])}")
        await _async_chat_loop(agent, "Remote MCP agent (Jina AI)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PATTERNS = {
    "lazy": (lazy, False),
    "context_manager": (context_manager, True),
    "agent_task_cm": (agent_task_cm, False),
    "agent_task_lazy": (agent_task_lazy, False),
    "guardrails": (guardrails, True),
    "sandbox": (sandbox, True),
    "remote": (remote, True),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP integration examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name:20s} {fn.__doc__.strip().splitlines()[0]}"
            for name, (fn, _) in PATTERNS.items()
        ),
    )
    parser.add_argument("pattern", choices=PATTERNS.keys(), help="Pattern to run")
    args = parser.parse_args()

    fn, is_async = PATTERNS[args.pattern]
    if is_async:
        asyncio.run(fn())
    else:
        fn()
