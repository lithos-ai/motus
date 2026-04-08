# OpenAI Agents SDK

Run OpenAI Agents SDK code through Motus with full tracing and cloud deployment. Import from `motus.openai_agents` instead of `agents` — everything else stays the same.

## Installation

```bash
uv sync --extra openai-agents
```

## Basic usage

```python
from motus.openai_agents import Agent, Runner

agent = Agent(name="assistant", instructions="You are helpful.")
result = await Runner.run(agent, "Hello!")
print(result.final_output)
```

The `Runner` returned by `motus.openai_agents` wraps every call with tracing and model interception. You do not need to change your agent definitions, tool functions, or run logic.

## What Motus adds

You get the following without any code changes:

### Tracing

Every agent turn, tool call, and model generation is captured by `TraceManager`. The `MotusTracingProcessor` replaces the SDK's default `BackendSpanExporter` (which posts traces to `api.openai.com`) on import. Traces flow into the Motus trace viewer, Jaeger export, and analytics pipeline.

Tracing is auto-registered when you import `motus.openai_agents`. You can also call `register_tracing()` explicitly:

```python
from motus.openai_agents import register_tracing

register_tracing()
```

### Model wrapping

`MotusOpenAIProvider` and `MotusMultiProvider` sit in the model call path as transparent pass-throughs. Future releases will add hooks for caching, routing, and cost control at this layer.

### Tool wrapping

Tool invocations are intercepted before execution. Each `function_tool` call produces a traced span with input arguments and output. Future releases will add tool-level optimization and caching.

## Deployment

The same agent can be served via Motus:

```bash
motus serve start myapp:agent --port 8000
```

Where `agent` is an OpenAI `Agent` instance — Motus auto-detects it.

## Runner methods

`Runner` exposes the same three methods as the SDK's original `Runner`:

```python
# Async
result = await Runner.run(agent, "Hello!")

# Sync
result = Runner.run_sync(agent, "Hello!")

# Streaming
stream = Runner.run_streamed(agent, "Hello!")
```

Each method registers tracing, wraps tools, and injects a `MotusOpenAIProvider` into the `RunConfig` before delegating to the original SDK runner.

## Run configuration

You can pass a custom `RunConfig`. Motus upgrades the default `OpenAIProvider` or `MultiProvider` to their Motus counterparts. If you supply your own custom provider, Motus preserves it:

```python
from motus.openai_agents import Runner, RunConfig, MotusOpenAIProvider

config = RunConfig(model_provider=MotusOpenAIProvider())
result = await Runner.run(agent, "Hello!", run_config=config)
```

## What works

All OpenAI Agents SDK features are supported:

- `function_tool` definitions
- `Agent` with instructions, tools, and handoffs
- `Runner.run()`, `Runner.run_sync()`, `Runner.run_streamed()`
- Handoffs between agents
- Guardrails (input and output)
- Custom tools and MCP tools
- Multi-provider routing (OpenAI, LiteLLM)

## Motus-specific exports

In addition to re-exporting the full `agents` package, `motus.openai_agents` provides:

| Export                        | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `MotusModel`                 | Base model wrapper                                       |
| `MotusResponsesModel`        | Responses API model wrapper                              |
| `MotusChatCompletionsModel`  | Chat Completions API model wrapper                       |
| `MotusLitellmModel`          | LiteLLM model wrapper                                   |
| `MotusOpenAIProvider`         | Provider that returns Motus model wrappers              |
| `MotusMultiProvider`          | Multi-provider with Motus interception                  |
| `MotusLitellmProvider`       | LiteLLM provider with Motus interception                |
| `MotusTracingProcessor`      | Bridges OAI SDK spans into `TraceManager`                |
| `register_tracing()`          | Registers the tracing processor (called on import)       |
| `get_tracer()`                | Returns the `TraceManager` instance                      |

## Import mapping

```python
from motus.openai_agents import X
```

re-exports everything from the `agents` package. The underlying PyPI package is `openai-agents` (installed as `agents`). Motus overrides `Runner`, `OpenAIProvider`, `MultiProvider`, and model classes with its own wrappers at import time.

## Trace export

Traces are auto-exported on process exit when `TraceManager.config.export_enabled` is `True`. You can also export manually:

```python
from motus.openai_agents import get_tracer

tracer = get_tracer()
if tracer:
    tracer.export_trace()
```
