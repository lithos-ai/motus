# OpenAI Agents SDK on Motus

Motus provides compatibility with the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) (v0.13.4+).
Examples in this directory are deployable as HTTP APIs via `motus serve`
and run with full tracing on the Motus cloud platform.

## Quick start

```python
# tools.py — upstream example with one import change
from motus.openai_agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(name="Weather agent", tools=[get_weather])
```

```sh
# Run directly
OPENAI_API_KEY=sk-... python examples/openai_agents/tools.py

# Deploy as HTTP API (local)
motus serve start examples.openai_agents.tools:agent --port 8000

# Deploy to Motus cloud (from examples/openai_agents/)
cd examples/openai_agents
motus deploy --name my-agent tools:agent
```

## Examples

All examples are adapted from
[openai/openai-agents-python](https://github.com/openai/openai-agents-python)
with `from agents import` changed to `from motus.openai_agents import`.
Each defines a module-level `Agent` instance deployable via serve.

| Example | Pattern | Agent var | serve | Cloud | Notes |
|---------|---------|-----------|--------|-------|-------|
| [tools.py](tools.py) | Function tools | `agent` | Yes | Yes | -- |
| [tool_guardrails.py](tool_guardrails.py) | Input/output guardrails on tools | `agent` | Yes | Yes | -- |
| [routing.py](routing.py) | Language-routing triage with handoffs | `triage_agent` | Yes | Yes | -- |
| [agents_as_tools.py](agents_as_tools.py) | Orchestrator calling sub-agents as tools | `orchestrator_agent` | Yes | Yes | -- |
| [agents_as_tools_structured.py](agents_as_tools_structured.py) | Structured agent-tool inputs | `orchestrator` | Yes | Yes | -- |
| [llm_as_a_judge.py](llm_as_a_judge.py) | Evaluation feedback loop | `story_outline_generator` | Yes | Yes | -- |
| [output_guardrails.py](output_guardrails.py) | Output guardrail tripwires | `agent` | Yes | Yes | Guardrail refusal returned as message, not exception |
| [message_filter.py](message_filter.py) | Handoff with message filtering | `first_agent` | Yes | Yes | -- |

## How it works

`motus.openai_agents` re-exports the entire agents SDK, overriding:

- **`Runner`** — registers `MotusTracingProcessor`, wraps tools, injects `MotusOpenAIProvider`
- **`OpenAIProvider`** → `MotusOpenAIProvider` — returns traced model instances
- **`MotusTracingProcessor`** — bridges OAI SDK `Span` events into `TraceManager.ingest_external_span()`

The serve worker detects OAI `Agent` instances via `isinstance` check and
wraps them with `_adapt_openai_agent()`, which converts motus `ChatMessage`
state to OAI message format and calls `Runner.run()`.

### Cloud deployment

When deploying to Motus cloud, include `requirements.txt` with
`openai-agents>=0.13.4` (the SDK is not in the base image). No secrets
are needed — the platform routes Responses API calls through the model
proxy:

```sh
motus deploy --name my-agent tools:agent
```

Session state (conversation history) is persisted in DynamoDB and survives
backend restarts, failovers, and scaling events.
