# Motus End-to-End Examples

Complete applications showing how multiple Motus features work together.

---

## Example 1: Customer Support Bot

A production-ready customer support agent with memory, guardrails, and multiple tools.

```python
import asyncio
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import tool, get_mcp
from motus.memory import CompactionMemory, CompactionMemoryConfig
from motus.guardrails import InputGuardrailTripped

# --- Tools ---

@tool
async def lookup_order(order_id: str) -> str:
    """Look up order status by order ID."""
    # Replace with real DB call
    return f"Order {order_id}: Shipped on 2024-03-15, arriving 2024-03-18"

@tool
async def refund(order_id: str, reason: str) -> str:
    """Issue a refund for an order."""
    return f"Refund initiated for {order_id}. Reason: {reason}. Confirmation #RF-{order_id[-4:]}"

@tool
async def escalate(summary: str) -> str:
    """Escalate to a human agent with a summary of the issue."""
    return f"Escalated. A human agent will follow up within 24 hours."

# --- Guardrails ---

def block_prompt_injection(value: str) -> str | None:
    patterns = ["ignore previous", "system prompt", "you are now"]
    if any(p in value.lower() for p in patterns):
        raise InputGuardrailTripped("Suspicious input blocked")
    return None

def redact_pii(value: str) -> str | None:
    import re
    value = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", value)
    value = re.sub(r"\b\d{16}\b", "[CARD]", value)
    return value

# --- Agent ---

async def main():
    client = OpenAIChatClient()

    memory = CompactionMemory(
        config=CompactionMemoryConfig(
            session_id="customer-session-001",
            log_base_path="./support_logs",
        ),
    )

    agent = ReActAgent(
        client=client,
        model_name="gpt-4o",
        system_prompt="""You are a customer support agent for an e-commerce store.
Be polite, concise, and helpful. Use tools to look up orders and process refunds.
Escalate if you cannot resolve the issue in 3 exchanges.""",
        tools=[lookup_order, refund, escalate],
        memory=memory,
        max_steps=8,
        input_guardrails=[block_prompt_injection],
        output_guardrails=[redact_pii],
    )

    # Simulate conversation
    for prompt in [
        "Hi, where is my order ORD-7821?",
        "It's been delayed 5 days. I want a refund.",
        "Thank you!",
    ]:
        print(f"User: {prompt}")
        response = await agent(prompt)
        print(f"Agent: {response}\n")

asyncio.run(main())
```

---

## Example 2: Research Pipeline (Task Graph + ReAct)

A workflow that researches multiple topics in parallel, then synthesizes a report.

```python
import asyncio
from motus.runtime import init, shutdown, agent_task
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import tool

init()

client = OpenAIChatClient()  # Cloud proxy auto-provides API keys

@tool
async def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [simulated results]"

@agent_task(retries=2, timeout=60.0)
async def research_topic(topic: str) -> str:
    """Research a single topic using a ReAct agent."""
    agent = ReActAgent(
        client, "anthropic/claude-sonnet-4.5",
        system_prompt="You are a research assistant. Search and summarize findings.",
        tools=[web_search],
        max_steps=5,
    )
    return await agent(f"Research the latest developments in: {topic}")

@agent_task(timeout=120.0)
async def write_report(findings: list[str]) -> str:
    """Synthesize research findings into a report."""
    agent = ReActAgent(
        client, "anthropic/claude-sonnet-4.5",
        system_prompt="You are a technical writer. Create clear, structured reports.",
    )
    combined = "\n\n---\n\n".join(findings)
    return await agent(f"Write a report synthesizing these findings:\n\n{combined}")

async def main():
    # Research 3 topics in parallel
    topics = ["quantum computing 2024", "AI regulation EU", "fusion energy progress"]
    research_futures = [research_topic(t) for t in topics]

    # Write report waits for all research to complete
    report = write_report(research_futures)
    print(report.af_result(timeout=300))

    shutdown()

asyncio.run(main())
```

---

## Example 3: Code Assistant with Sandbox

An agent that writes and executes Python code in a sandbox. Same code runs locally (Docker) and in cloud deploy (platform-managed).

```python
import asyncio
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import get_sandbox, FunctionTool, tool

async def main():
    client = OpenAIChatClient()

    with get_sandbox(image="python:3.12") as sb:

        @tool
        async def install_package(package: str) -> str:
            """Install a Python package in the sandbox."""
            return await sb.exec("pip", "install", package)

        agent = ReActAgent(
            client=client,
            model_name="gpt-4o",
            system_prompt="""You are a Python coding assistant with access to a sandbox.
Write code, execute it, and iterate until you get the correct result.
Always test your code before giving the final answer.""",
            tools={
                "run_python": FunctionTool(sb.python),
                "run_shell": FunctionTool(sb.sh),
                "install": install_package,
            },
            max_steps=15,
        )

        result = await agent(
            "Create a matplotlib chart showing the Fibonacci sequence up to F(20). "
            "Save it as /tmp/fib.png and tell me the values."
        )
        print(result)

asyncio.run(main())
```

---

## Example 4: Multi-Agent System with Delegation

An orchestrator that delegates to specialized sub-agents.

```python
import asyncio
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import tool

async def main():
    client = OpenAIChatClient()

    # Specialist agents
    analyst = ReActAgent(
        client, "gpt-4o",
        system_prompt="You analyze data and extract insights. Be quantitative.",
        tools=[],
    )

    @tool
    async def generate_chart(spec: str) -> str:
        """Generate a chart based on a specification."""
        return f"Chart generated: {spec}"

    visualizer = ReActAgent(
        client, "gpt-4o",
        system_prompt="You create data visualizations based on descriptions.",
        tools=[generate_chart],
    )

    # Orchestrator uses specialists as tools
    orchestrator = ReActAgent(
        client, "gpt-4o",
        system_prompt="""You are a data team lead.
Break down data analysis requests into steps:
1. Use the analyst to interpret the data
2. Use the visualizer to create charts
Synthesize their outputs into a final answer.""",
        tools=[
            analyst.as_tool(name="analyst", description="Analyze data and extract insights"),
            visualizer.as_tool(name="visualizer", description="Create charts and visualizations"),
        ],
        max_steps=10,
    )

    result = await orchestrator(
        "Analyze the trend of global AI investment from 2020-2024 and create a chart."
    )
    print(result)

asyncio.run(main())
```

---

## Example 5: Deploy to Cloud (Framework Support)

All Motus framework agents can be deployed directly — no wrapper function needed. The serve runtime detects `ServableAgent` instances (objects with `run_turn`) and OpenAI Agent instances automatically.

```python
# agent.py — deploy with: motus deploy --name my-support-bot agent:support_agent
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import tool

@tool
async def lookup(query: str) -> str:
    """Search knowledge base."""
    return f"Answer for: {query}"

client = OpenAIChatClient()  # Cloud proxy auto-provides API keys

support_agent = ReActAgent(
    client=client,
    model_name="gpt-4o",
    system_prompt="You are a support agent. Be helpful and concise.",
    tools=[lookup],
    max_steps=5,
)
# Deploy target: agent:support_agent — run_turn is inherited from AgentBase
```

```bash
# Local test
motus serve start agent:support_agent --port 8000
motus serve chat http://localhost:8000 "What is your return policy?"

# Cloud deploy (first time — creates project, saves to motus.toml)
motus login        # provision credentials (once)
motus deploy --name my-support-bot agent:support_agent

# Cloud redeploy (subsequent — reads from motus.toml)
motus deploy
```

### Same pattern with other frameworks

```python
# Anthropic ToolRunner — deploy with: motus deploy --name my-bot agent:my_runner
from anthropic.lib.tools import BetaAsyncFunctionTool
from motus.anthropic import ToolRunner

async def lookup(query: str) -> str:
    """Search knowledge base."""
    return f"Answer for: {query}"

my_runner = ToolRunner(model="claude-sonnet-4-20250514", max_tokens=1024, tools=[BetaAsyncFunctionTool(lookup)])
```

```python
# OpenAI Agents — deploy with: motus deploy --name my-bot agent:my_agent
from motus.openai_agents import Agent, function_tool

@function_tool
def lookup(query: str) -> str:
    """Search knowledge base."""
    return f"Answer for: {query}"

my_agent = Agent(name="support", model="gpt-4.1", instructions="You are a support agent.", tools=[lookup])
```

```python
# Google ADK — deploy with: motus deploy --name my-bot agent:my_agent
from motus.google_adk.agents.llm_agent import Agent

def lookup(query: str) -> str:
    """Search knowledge base."""
    return f"Answer for: {query}"

my_agent = Agent(model="gemini-2.0-flash", name="support", description="Support agent", instruction="Help users.", tools=[lookup])
```
