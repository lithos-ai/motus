# Anthropic SDK

Run Anthropic Python SDK code through Motus with full tracing, model proxying, and cloud deployment. Import from `motus.anthropic` instead of `anthropic.lib.tools` — everything else stays the same.

## Installation

The Anthropic SDK is included as a core Motus dependency. No extra install is needed. The tool runner integration requires `anthropic>=0.49.0`:

```bash
# Upgrade if needed
uv pip install 'anthropic>=0.49.0'
```

## Basic usage

```python
from motus.anthropic import ToolRunner, beta_async_tool

@beta_async_tool
async def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location."""
    return '{"temperature": "20C", "condition": "Sunny"}'

runner = ToolRunner(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[get_weather],
)
```

The `ToolRunner` wraps the SDK's Beta Tool Runner with tracing and a serve-compatible interface. You do not need to change your tool definitions or message handling.

## What Motus adds

You get the following without any code changes:

### Tracing

`MotusBetaAsyncToolRunner` (and its sync/streaming variants) subclasses the SDK's `BetaAsyncToolRunner` and overrides `_handle_request()` and `_generate_tool_call_response()` to emit `model_call` and `tool_call` spans into `TraceManager`. Every model generation and tool invocation is captured with timing, token usage, input/output payloads, and parent-child relationships.

Tracing is auto-registered when you import `motus.anthropic`. You can also call `register_tracing()` explicitly:

```python
from motus.anthropic import register_tracing

register_tracing()
```

### Model proxy

When deployed to Motus cloud, the platform automatically routes Anthropic API calls through the model proxy. No `ANTHROPIC_API_KEY` is needed in the deployed environment — the proxy handles authentication, rate limiting, and cost tracking transparently.

### Tool wrapping

Every tool invocation is intercepted and traced. Each `@beta_async_tool` call produces a span with the tool name, input arguments, output, timing, and error status. Failed tool calls are recorded with the exception details.

## Deployment

### Local serving

```bash
motus serve start myapp:runner --port 8000
```

### Cloud deployment

```bash
cd my_project
motus deploy --name my-agent tools_runner:runner
```

When deploying to Motus cloud, include `requirements.txt` with `anthropic>=0.49.0` (the SDK is not in the base image). No API key secrets are needed — the platform routes Anthropic API calls through the model proxy.

Session state (conversation history) is persisted in DynamoDB and survives backend restarts, failovers, and scaling events.

## ToolRunner

`ToolRunner` holds configuration (model, tools, system prompt, max iterations) and provides a `run_turn()` method satisfying the serve agent contract:

```python
async def run_turn(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]
```

Each `run_turn()` creates a fresh `MotusBetaAsyncToolRunner` — tool runners are single-use generators that cannot be re-iterated. State is managed by the serve layer.

```python
from motus.anthropic import ToolRunner, beta_async_tool

@beta_async_tool
async def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

runner = ToolRunner(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[search],
    system="You are a helpful research assistant.",
    max_iterations=10,
)
```

## What works

All Anthropic SDK Beta Tool Runner features are supported:

- `@beta_async_tool` and `@beta_tool` decorated functions
- `ToolRunner` with system prompts and iteration limits
- Multi-tool agents with parallel tool calls
- Deferred tool loading (tool-search beta)
- Sync, async, streaming, and async-streaming runners

## Motus-specific exports

In addition to re-exporting tool types from `anthropic.lib.tools`, `motus.anthropic` provides:

| Export                              | Description                                          |
|-------------------------------------|------------------------------------------------------|
| `ToolRunner`                        | Serve-compatible wrapper with `run_turn()`           |
| `MotusBetaToolRunner`               | Sync tool runner with tracing                        |
| `MotusBetaStreamingToolRunner`      | Sync streaming tool runner with tracing              |
| `MotusBetaAsyncToolRunner`          | Async tool runner with tracing                       |
| `MotusBetaAsyncStreamingToolRunner` | Async streaming tool runner with tracing             |
| `beta_async_tool`                   | Re-exported from `anthropic.lib.tools`               |
| `beta_tool`                         | Re-exported from `anthropic.lib.tools`               |
| `BetaAsyncFunctionTool`             | Re-exported from `anthropic.lib.tools`               |
| `BetaFunctionTool`                  | Re-exported from `anthropic.lib.tools`               |
| `register_tracing()`               | Creates the `TraceManager` (called on import)        |
| `get_tracer()`                      | Returns the `TraceManager` instance                  |

## Import mapping

```python
from motus.anthropic import ToolRunner, beta_async_tool
```

The underlying PyPI package is `anthropic` (v0.49.0+ required for tool runner support). `motus.anthropic` re-exports tool-related classes from `anthropic.lib.tools` and adds the Motus `ToolRunner` and tracing-instrumented runner subclasses.

## Trace export

Traces are auto-exported on process exit when `TraceManager.config.export_enabled` is `True`. You can also export manually:

```python
from motus.anthropic import get_tracer

tracer = get_tracer()
if tracer:
    tracer.export_trace()
```

## Traced span types

The integration produces three span types in `TraceManager`:

- **`model_call`** — Created from `_handle_request()` override. Contains model name, input messages, conversation snapshot, token usage, and timing.
- **`tool_call`** — Created from `_generate_tool_call_response()` override. Contains tool name, input arguments, output, error status, and timing.
- **`agent_call`** — Root span created per `run_turn()`. Parents all model and tool spans within the turn. Updated with total duration on completion.
