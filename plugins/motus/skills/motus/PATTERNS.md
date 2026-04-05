# Motus Code Patterns

Proven patterns for common scenarios. Each pattern is copy-pasteable and runnable.

---

## Pattern 1: Minimal agent

```python
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import tool

@tool
async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

client = OpenAIChatClient()
agent = ReActAgent(client, "gpt-4o", tools=[greet])
response = await agent("Say hello to Alice")
```

---

## Pattern 2: Multiple tools with class

```python
from motus.tools import tools

@tools(prefix="db_")
class DatabaseTools:
    def __init__(self, connection_string: str):
        self.conn = connection_string

    async def query(self, sql: str) -> str:
        """Execute a read-only SQL query."""
        return f"Results for: {sql}"

    async def tables(self) -> str:
        """List all tables in the database."""
        return "users, orders, products"

agent = ReActAgent(client, "gpt-4o", tools=[DatabaseTools("postgres://...")])
```

---

## Pattern 3: Agent composing other agents

```python
researcher = ReActAgent(client, "gpt-4o", tools=[web_search], system_prompt="You research topics.")
writer = ReActAgent(client, "gpt-4o", system_prompt="You write clear summaries.")

orchestrator = ReActAgent(
    client, "gpt-4o",
    system_prompt="You coordinate research and writing tasks.",
    tools=[
        researcher.as_tool(name="research", description="Research a topic"),
        writer.as_tool(name="write", description="Write a summary from research notes"),
    ],
)

result = await orchestrator("Write a summary about quantum computing advances in 2024")
```

---

## Pattern 4: MCP with guardrails and filtering

```python
from motus.tools import get_mcp, tools
from motus.guardrails import ToolInputGuardrailTripped

def block_writes(path: str = ""):
    if any(path.endswith(ext) for ext in [".env", ".key", ".pem"]):
        raise ToolInputGuardrailTripped(f"Write to {path} blocked")

async with get_mcp(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]) as session:
    safe_fs = tools(
        session,
        prefix="fs_",
        blocklist={"write_file", "move_file"},
        input_guardrails=[block_writes],
    )

    agent = ReActAgent(client, "gpt-4o", tools=safe_fs)
    result = await agent("List all CSV files in /data")
```

---

## Pattern 5: Task-graph workflow with parallel execution

```python
from motus.runtime import init, agent_task

init()

@agent_task(retries=2, timeout=30.0)
async def scrape(url: str) -> str:
    async with httpx.AsyncClient() as c:
        return (await c.get(url)).text

@agent_task
async def extract_links(html: str) -> list[str]:
    ...

@agent_task
async def summarize_all(texts: list) -> str:
    ...

# scrape runs 3x in parallel; summarize_all waits for all three
pages = [scrape(url) for url in ["https://a.com", "https://b.com", "https://c.com"]]
links = [extract_links(p) for p in pages]
summary = summarize_all(links)
print(summary.af_result(timeout=60))
```

---

## Pattern 6: Mixed ReAct + task graph

```python
from motus.runtime import agent_task

@agent_task
async def research(topic: str) -> str:
    """Use a ReAct agent inside a task-graph step."""
    agent = ReActAgent(client, "gpt-4o", tools=[web_search])
    return await agent(f"Research: {topic}")

@agent_task
async def compile_report(research_a: str, research_b: str) -> str:
    agent = ReActAgent(client, "gpt-4o")
    return await agent(f"Compile a report from:\n1. {research_a}\n2. {research_b}")

# Two research tasks run in parallel, report waits for both
a = research("topic A")
b = research("topic B")
report = compile_report(a, b)
```

---

## Pattern 7: Compaction memory with session restore

```python
from motus.memory import CompactionMemory, CompactionMemoryConfig

memory = CompactionMemory(
    config=CompactionMemoryConfig(
        session_id="user-123-session",
        log_base_path="./conversation_logs",
        safety_ratio=0.75,
    ),
)

agent = ReActAgent(client, "gpt-4o", memory=memory, tools=[...])

# ... many turns later, process restarts ...

# Restore the session
restored = CompactionMemory.restore_from_log(
    session_id="user-123-session",
    log_base_path="./conversation_logs",
)
agent = ReActAgent(client, "gpt-4o", memory=restored, tools=[...])
```

---

## Pattern 8: Structured output with guardrails

```python
from pydantic import BaseModel, Field
from motus.guardrails import OutputGuardrailTripped

class Analysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    summary: str

def validate_confidence(confidence: float):
    if confidence < 0.5:
        raise OutputGuardrailTripped("Low confidence — retry")

agent = ReActAgent(
    client, "gpt-4o",
    response_format=Analysis,
    output_guardrails=[validate_confidence],
)

result = await agent("Analyze the sentiment of: 'This product is amazing!'")
# result is an Analysis instance
```

---

## Pattern 9: Docker sandbox for code execution

```python
from motus.tools import get_sandbox, FunctionTool

with get_sandbox(image="python:3.12") as sb:
    agent = ReActAgent(
        client, "gpt-4o",
        system_prompt="You can execute Python code to answer questions.",
        tools={
            "run_python": FunctionTool(sb.python),
            "run_shell": FunctionTool(sb.sh),
        },
    )
    result = await agent("Calculate the first 20 Fibonacci numbers")
```

---

## Pattern 10: Lifecycle hooks for monitoring

```python
from motus.runtime.hooks import register_hook, register_type_hook

# Log all task completions
register_hook("task_end", lambda e: print(f"✓ {e.name} done"))
register_hook("task_error", lambda e: print(f"✗ {e.name} failed: {e.error}"))

# Track LLM costs
token_total = 0
def track_tokens(event):
    global token_total
    if hasattr(event.result, "usage"):
        token_total += event.result.usage.get("total_tokens", 0)

register_type_hook("MODEL_CALL", "task_end", track_tokens)
```

---

## Pattern 11: Serving as HTTP API

```python
# myapp.py
from motus.serve import AgentServer
from motus.models.base import ChatMessage

def my_agent(message: ChatMessage, state: list) -> tuple[ChatMessage, list]:
    # Your agent logic here
    response = ChatMessage.assistant_message(content="Processed: " + message.content)
    return response, state + [message, response]

server = AgentServer(
    my_agent,
    max_workers=4,
    ttl=3600,        # Session expires after 1 hour
    timeout=120,     # Max 2 minutes per request
)
```

```bash
motus serve start myapp:server --port 8000
```

---

## Pattern 12: Full production setup

```python
from motus.agent import ReActAgent
from motus.models import AnthropicChatClient
from motus.tools import tool, get_mcp
from motus.memory import CompactionMemory, CompactionMemoryConfig
from motus.guardrails import InputGuardrailTripped
from motus.runtime.hooks import register_hook

# Guardrails
def block_injection(value: str) -> str | None:
    if "ignore previous instructions" in value.lower():
        raise InputGuardrailTripped("Prompt injection detected")
    return None

def redact_pii(value: str) -> str | None:
    import re
    return re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", value)

# Tools
@tool
async def lookup_customer(customer_id: str) -> str:
    """Look up customer details by ID."""
    return f"Customer {customer_id}: John Doe, Premium plan"

mcp_session = get_mcp(command="npx", args=["-y", "@modelcontextprotocol/server-postgres", "postgres://..."])

# Memory
memory = CompactionMemory(
    config=CompactionMemoryConfig(session_id="prod-session", safety_ratio=0.75),
)

# Monitoring
register_hook("task_error", lambda e: print(f"ERROR: {e.name}: {e.error}"))

# Agent
client = AnthropicChatClient()
agent = ReActAgent(
    client=client,
    model_name="claude-sonnet-4-20250514",
    system_prompt="You are a customer support agent. Be helpful and concise.",
    tools=[lookup_customer, mcp_session],
    memory=memory,
    max_steps=10,
    input_guardrails=[block_injection],
    output_guardrails=[redact_pii],
    cache_policy="auto",
)
```
