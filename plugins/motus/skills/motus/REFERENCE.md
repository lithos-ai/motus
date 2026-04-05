# Motus API Reference

## Model Clients

All clients implement `BaseChatClient` with `create()` and `parse()` methods.

### OpenAIChatClient

```python
from motus.models import OpenAIChatClient

client = OpenAIChatClient(
    api_key: str | None = None,        # Falls back to OPENAI_API_KEY env
    base_url: str | None = None,       # Custom endpoint (e.g., local Ollama)
    http_client: httpx.AsyncClient | None = None,
)
```

### AnthropicChatClient

```python
from motus.models import AnthropicChatClient

client = AnthropicChatClient(
    api_key: str | None = None,        # Falls back to ANTHROPIC_API_KEY env
    auth_token: str | None = None,     # OAuth token (overrides api_key)
)
```

### GeminiChatClient

```python
from motus.models import GeminiChatClient
client = GeminiChatClient()
```

### OpenRouterChatClient

```python
from motus.models import OpenRouterChatClient
client = OpenRouterChatClient(
    api_key: str | None = None,        # Falls back to OPENROUTER_API_KEY env
)
```

### CachePolicy

```python
from motus.models.base import CachePolicy

CachePolicy.NONE    # No caching
CachePolicy.STATIC  # Cache system prompt + tool definitions
CachePolicy.AUTO    # STATIC + cache conversation prefix (default)
```

---

## ReActAgent

```python
from motus.agent import ReActAgent

agent = ReActAgent(
    client: BaseChatClient,                    # REQUIRED — model client instance
    model_name: str,                           # REQUIRED — "gpt-4o", "claude-sonnet-4-20250514", etc.
    name: str | None = None,                   # Agent name (auto-inferred if omitted)
    system_prompt: str | None = None,          # System instructions
    tools: Any | None = None,                  # Tools: list, dict, @tools class, MCPSession, etc.
    response_format: type[BaseModel] | None = None,  # Structured output (Pydantic model)
    max_steps: int = 20,                       # Max reasoning iterations before stopping
    timeout: float | None = None,              # Total timeout in seconds
    memory_type: str = "basic",                # "basic" or "compact"
    memory: BaseMemory | None = None,          # Custom memory instance (overrides memory_type)
    input_guardrails: list | None = None,      # Guardrails applied to user input
    output_guardrails: list | None = None,     # Guardrails applied to agent output
    include_reasoning: bool = True,            # Include model reasoning in response
    cache_policy: CachePolicy | str = "auto",  # Prompt caching strategy
    step_callback: Callable | None = None,     # async fn(content, tool_calls) after each step
)
```

### Key methods

```python
response = await agent("user message")         # Run agent (async)
response = resolve(agent("user message"))    # Run agent (sync via future)
response = agent("user message").af_result()    # Run agent (sync via future)
forked = agent.fork()                           # Independent copy with forked memory
tool = agent.as_tool(name=..., description=...) # Wrap as tool for another agent
await agent.compact_memory()                    # Trigger memory compaction
trace = agent.get_execution_trace()             # Get full execution trace
agent.reset()                                   # Clear conversation history
```

---

## Tools

### @tool decorator

```python
from motus.tools import tool

@tool                                            # Minimal — name and schema from function
@tool(name="custom_name")                        # Custom name
@tool(description="Override description")        # Custom description
@tool(schema=MyInputSchema)                      # Pydantic input schema
@tool(schema={"type": "object", ...})            # Raw JSON schema
@tool(input_guardrails=[...])                    # Tool-level input guardrails
@tool(output_guardrails=[...])                   # Tool-level output guardrails
@tool(on_start=callback)                         # Per-tool lifecycle hooks
@tool(on_end=callback)
@tool(on_error=callback)
async def my_tool(param: str) -> str:
    """Docstring becomes tool description for the LLM."""
    return result
```

### @tools class decorator

```python
from motus.tools import tools

@tools(
    prefix: str = "",                            # Prefix all tool names: "db_query", "db_insert"
    include_private: bool = False,               # Include _private methods
    method_schemas: dict | None = None,          # Per-method Pydantic schemas
    method_aliases: dict | None = None,          # Rename methods: {"inc": "increase"}
    allowlist: Container[str] | None = None,     # Only include these methods
    blocklist: Container[str] | None = None,     # Exclude these methods
    input_guardrails: list | None = None,        # Applied to all methods
    output_guardrails: list | None = None,
)
class MyTools:
    async def query(self, sql: str) -> str: ...
    async def insert(self, table: str, data: dict) -> str: ...
```

### InputSchema (clean Pydantic base)

```python
from motus.tools import InputSchema
from pydantic import Field

class MyInput(InputSchema):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
```

### FunctionTool (manual wrapping)

```python
from motus.tools import FunctionTool

tool = FunctionTool(
    func: Callable,                              # Sync or async function to wrap
    name: str | None = None,
    schema: dict | type[BaseModel] | None = None,
    input_guardrails: list | None = None,
    output_guardrails: list | None = None,
)
```

### AgentTool (agent as tool)

```python
from motus.tools import AgentTool

# Via convenience method (preferred)
tool = agent.as_tool(
    name="researcher",
    description="Research a topic thoroughly",
    output_extractor: Callable | None = None,    # Extract specific part of response
    stateful: bool = False,                      # True = share memory across calls
    max_steps: int | None = None,
    input_guardrails: list | None = None,
    output_guardrails: list | None = None,
)

# Via class
tool = AgentTool(agent, name="researcher", description="...")
```

### MCPSession (external tool servers)

```python
from motus.tools import get_mcp

# Stdio transport (local process)
session = get_mcp(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    env: dict | None = None,
)

# HTTP transport (remote server)
session = get_mcp(
    url="https://mcp.example.com/v1",
    headers={"Authorization": "Bearer ..."},
)

# In Docker sandbox
session = get_mcp(
    image="node:20",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-everything"],
    port=3000,
)
```

**Usage patterns:**

```python
# Lazy — pass to agent, connects on first call
agent = ReActAgent(..., tools=[session])

# Explicit lifecycle
async with get_mcp(...) as session:
    agent = ReActAgent(..., tools=[session])

# With prefix and filtering
from motus.tools import tools
wrapped = tools(session, prefix="fs_", blocklist={"write_file"})
```

### Sandbox (Docker code execution)

```python
from motus.tools import get_sandbox

sandbox = get_sandbox(
    image: str = "python:3.12",
    dockerfile: str | None = None,               # Build from Dockerfile
    name: str | None = None,                     # Container name
    env: dict[str, str] | None = None,
    mounts: dict[str, str] | None = None,        # Host:container path mapping
    connect: str | None = None,                  # Connect to existing container
    ports: dict[int, int | None] | None = None,  # Port mapping
)
```

**Usage:**

```python
# Context manager
with get_sandbox(image="python:3.12") as sb:
    result = await sb.exec("python", "-c", "print('hello')")

# As tools for agent
from motus.tools import FunctionTool
agent = ReActAgent(..., tools={
    "python": FunctionTool(sandbox.python),
    "sh": FunctionTool(sandbox.sh),
})
```

---

## Memory

### BasicMemory (default)

```python
from motus.memory import BasicMemory

memory = BasicMemory()
# Or via agent:
agent = ReActAgent(..., memory_type="basic")
```

No compaction. Conversations grow until context limit is hit.

### CompactionMemory

```python
from motus.memory import CompactionMemory, CompactionMemoryConfig

memory = CompactionMemory(
    config=CompactionMemoryConfig(
        safety_ratio: float = 0.75,              # Compact at 75% of context window
        token_threshold: int | None = None,      # Or set explicit token count
        compact_model_name: str | None = None,   # Model for summarization (defaults to agent model)
        session_id: str | None = None,           # For session persistence (auto-UUID if omitted)
        log_base_path: str | None = None,        # Path for conversation logs
    ),
    on_compact: Callable[[dict], None] | None = None,  # Callback on compaction
)

# Restore from saved session
restored = CompactionMemory.restore_from_log(
    session_id="abc-123",
    log_base_path="./logs",
)
```

---

## Runtime & Task Graph

### Initialization

```python
from motus.runtime import init, shutdown, is_initialized

init()          # Explicit init (or auto-inits on first @agent_task call)
shutdown()      # Clean up resources
```

### @agent_task

```python
from motus.runtime import agent_task

@agent_task                                      # Simple
@agent_task(retries=3, timeout=10.0)             # With policy
@agent_task(retry_delay=1.0)                     # Delay between retries
@agent_task(num_returns=2)                       # Multi-return (returns tuple of futures)
@agent_task(on_start=cb, on_end=cb, on_error=cb) # Per-task hooks
async def my_task(arg): ...

# Call returns AgentFuture immediately
future = my_task("arg")
result = future.af_result(timeout=30)               # Block until done

# Override policy at call site
future = my_task.policy(retries=5)("arg")

# Dependency tracking — pass futures as arguments
a = fetch("url1")
b = fetch("url2")        # a and b run in parallel
c = merge(a, b)          # c waits for both
```

### Hooks

```python
from motus.runtime.hooks import (
    register_hook,           # Global hook
    register_task_hook,      # Per-task-name hook
    register_type_hook,      # Per-task-type hook (TOOL_CALL, MODEL_CALL, etc.)
    global_hook,             # Decorator form
    task_hook,               # Decorator form
)

# Events: "task_start", "task_end", "task_error", "task_cancelled"

@global_hook("task_end")
def log_completion(event):
    print(f"{event.name} completed with {event.result}")
```

---

## Guardrails

### Agent-level

```python
# Input guardrail: receives user prompt, returns modified or None
def validate(value: str) -> str | None:
    if "bad" in value:
        raise InputGuardrailTripped("Blocked")
    return None  # pass through unchanged

# Output guardrail: receives agent response
def redact(value: str) -> str | None:
    return value.replace("secret", "***")

agent = ReActAgent(..., input_guardrails=[validate], output_guardrails=[redact])
```

### Tool-level

```python
# Input guardrail — declare only the params you want to inspect
def check_query(query: str):
    if "DROP" in query:
        raise ToolInputGuardrailTripped("DROP forbidden")

# Output guardrail — receives the typed return value
def redact_result(result: str) -> str:
    return result.replace("password", "***")

@tool(input_guardrails=[check_query], output_guardrails=[redact_result])
async def db_query(query: str, limit: int = 10) -> str: ...
```

### Structured output guardrails

```python
# Declare fields from the response_format BaseModel
def validate_score(score: float):
    if not 0 <= score <= 1:
        raise OutputGuardrailTripped("Invalid score")

agent = ReActAgent(..., response_format=MyModel, output_guardrails=[validate_score])
```

---

## Serving & Deployment

For serving and cloud deployment, run `/motus deploy`. It handles the full workflow interactively.

---

## Key Import Paths

```python
# Agents
from motus.agent import ReActAgent

# Models
from motus.models import OpenAIChatClient, AnthropicChatClient, GeminiChatClient, OpenRouterChatClient
from motus.models.base import ChatMessage, ChatCompletion, CachePolicy

# Tools
from motus.tools import tool, tools, FunctionTool, AgentTool, InputSchema
from motus.tools import get_mcp, get_sandbox

# Memory
from motus.memory import BasicMemory, CompactionMemory, CompactionMemoryConfig

# Runtime
from motus.runtime import init, shutdown, agent_task
from motus.runtime.hooks import register_hook, global_hook, task_hook

# Guardrails
from motus.guardrails import (
    InputGuardrailTripped, OutputGuardrailTripped,
    ToolInputGuardrailTripped, ToolOutputGuardrailTripped,
)
```
