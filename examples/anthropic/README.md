# Anthropic SDK on Motus

Motus provides compatibility with the
[Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) (v0.49.0+).
Examples in this directory are deployable as HTTP APIs via `motus serve`
and run with full tracing on the Motus cloud platform.

## Quick start

```python
# tools_runner.py — tool runner with one import change
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

```sh
# Run directly
ANTHROPIC_API_KEY=sk-... python examples/anthropic/tools_runner.py

# Deploy as HTTP API (local)
motus serve start examples.anthropic.tools_runner:runner --port 8000

# Deploy to Motus cloud (from examples/anthropic/)
cd examples/anthropic
motus deploy --name my-agent tools_runner:runner
```

## Examples

All examples are adapted from
[anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)
and use `motus.anthropic.ToolRunner` for serve compatibility and tracing.
Each defines a module-level `runner` instance deployable via serve.

| Example | Pattern | serve | Cloud | Notes |
|---------|---------|-------|-------|-------|
| [tools_runner.py](tools_runner.py) | Basic tool runner with `@beta_async_tool` | Yes | Yes | -- |
| [search_tool.py](search_tool.py) | Deferred tool loading with tool-search meta-tool | Yes | Yes | Requires `tool-search-tool-2025-10-19` beta |

## How it works

`motus.anthropic` wraps the Anthropic SDK's Beta Tool Runner:

- **`ToolRunner`** -- holds configuration and provides a `run_turn(message, state)` method satisfying the serve agent contract
- **`MotusBetaAsyncToolRunner`** -- subclass of the SDK's `BetaAsyncToolRunner` that emits `model_call` and `tool_call` spans into `TraceManager`
- **Tracing** -- auto-registered on import; model calls, tool calls, and agent spans are captured with timing, token usage, and parent-child relationships

The serve worker detects `ToolRunner` instances via `run_turn` duck-typing.
Each `run_turn()` creates a fresh single-use runner (tool runners are
generators that cannot be re-iterated).

### Cloud deployment

When deploying to Motus cloud, include `requirements.txt` with
`anthropic>=0.49.0` (the SDK is not in the base image). No secrets are
needed — the platform automatically routes Anthropic API calls through
the model proxy:

```sh
motus deploy --name my-agent tools_runner:runner
```

Session state (conversation history) is persisted in DynamoDB and survives
backend restarts, failovers, and scaling events.
