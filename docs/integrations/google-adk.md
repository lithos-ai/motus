# Google ADK

Run Google Agent Development Kit agents through Motus with full tracing, model proxying, and cloud deployment. Import `Agent` from `motus.google_adk.agents.llm_agent` instead of `google.adk.agents`.

## Installation

```bash
uv sync --extra google-adk
```

This installs `google-adk>=1.27.2` and its dependencies.

## Basic usage

```python
from motus.google_adk.agents.llm_agent import Agent

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction="You are a helpful assistant that tells the current time.",
    tools=[get_current_time],
)
```

The `Agent` class subclasses the ADK `Agent` and adds a `run_turn()` method for serve compatibility. You do not need to change your tool definitions, callbacks, or sub-agent wiring.

## What Motus adds

You get the following without any code changes:

### Tracing

`MotusSpanProcessor` implements an OpenTelemetry `SpanProcessor` that receives Google ADK's OTEL spans (`invoke_agent`, `generate_content`, `execute_tool`) and converts them to Motus `task_meta` format for `TraceManager.ingest_external_span()`. Every agent invocation, LLM call, and tool execution is captured with timing, token usage, and input/output payloads.

Tracing is auto-registered when you import `motus.google_adk.agents.llm_agent`. The processor attaches to the existing `TracerProvider` if one is set, or creates one via ADK's `maybe_set_otel_providers()`. You can also call `register_tracing()` explicitly:

```python
from motus.google_adk.agents.llm_agent import register_tracing

register_tracing()
```

### Model proxy

When deployed to Motus cloud, the platform automatically routes Gemini API calls through the model proxy. No `GOOGLE_API_KEY` is needed in the deployed environment. The proxy handles authentication, rate limiting, and cost tracking transparently.

### Session replay

Each `run_turn()` creates a fresh `InMemoryRunner` and replays the conversation state into the ADK session, so the model sees full context across turns. This is handled automatically, and you do not need to manage session state manually.

## Deployment

### Local serving

```bash
motus serve start myapp:root_agent --port 8000
```

### Cloud deployment

```bash
cd my_project
motus deploy --name my-adk-agent agent:root_agent
```

When deploying to Motus cloud, include `requirements.txt` with `google-adk>=1.27.2`. No API key secrets are needed because the platform routes Gemini API calls through the model proxy.

Session state (conversation history) is persisted in DynamoDB and survives backend restarts, failovers, and scaling events.

## Agent subclass

`motus.google_adk.agents.llm_agent.Agent` subclasses the Google ADK `Agent` and adds `run_turn()`:

```python
async def run_turn(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]
```

Only the **root agent** needs to be the Motus `Agent` subclass. Sub-agents (i.e., `SequentialAgent`, `ParallelAgent`, and vanilla `google.adk.agents.Agent`) work natively without any modification.

```python
from motus.google_adk.agents.llm_agent import Agent
from google.adk.agents import Agent as ADKAgent

# Sub-agents can be vanilla ADK agents
math_agent = ADKAgent(
    model="gemini-2.5-flash",
    name="math_agent",
    instruction="You solve math problems.",
)

# Only the root agent needs the Motus subclass
root_agent = Agent(
    model="gemini-2.5-flash",
    name="triage",
    instruction="Route requests to the appropriate specialist.",
    sub_agents=[math_agent],
)
```

## What works

All Google ADK features are supported:

- Function tools (plain functions, Pydantic model args)
- All 6 callback hooks (`before_agent_callback`, `after_agent_callback`, `before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback`)
- Multi-agent composition (triage, sequential, parallel)
- Structured output with `output_schema` and `output_key`
- Parallel tool calls
- Multimodal tool outputs
- Token usage tracking

## Motus-specific exports

| Export                | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `Agent`               | ADK Agent subclass with `run_turn()` for serve                   |
| `MotusSpanProcessor`  | OTEL SpanProcessor bridging ADK spans to `TraceManager`          |
| `register_tracing()`  | Installs `MotusSpanProcessor` (called on import)                 |
| `get_tracer()`        | Returns the `TraceManager` instance                              |

## Import mapping

```python
from motus.google_adk.agents.llm_agent import Agent
```

The underlying PyPI package is `google-adk` (v1.27.2+). Only the root agent import changes. All other ADK imports (`google.adk.agents`, `google.genai.types`, etc.) remain the same.

## Trace export

Traces are auto-exported on process exit when `TraceManager.config.export_enabled` is `True`. You can also export manually:

```python
from motus.google_adk.agents.llm_agent import get_tracer

tracer = get_tracer()
if tracer:
    tracer.export_trace()
```

## Traced span types

The integration produces three span types in `TraceManager`, mapped from ADK's OTEL semantic conventions:

- **`model_call`**: From `generate_content` spans. Contains model name, input/output tokens, LLM request/response payloads, and finish reasons.
- **`tool_call`**: From `execute_tool` spans. Contains tool name, call arguments, tool response, and error type if failed.
- **`agent_call`**: From `invoke_agent` spans. Contains agent name. Parents model and tool spans within the agent invocation.
