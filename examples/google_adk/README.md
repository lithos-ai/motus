# Google ADK on Motus

Motus provides compatibility with the
[Google Agent Development Kit](https://github.com/google/adk-python) (v1.27.2+).
Examples in this directory are deployable as HTTP APIs via `motus serve`
and run with full tracing on the Motus cloud platform.

## Quick start

```python
# agent.py — upstream example with one import change
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

```sh
# Run locally
GOOGLE_API_KEY=... motus serve start examples.google_adk.agent:root_agent

# Deploy to Motus cloud (from examples/google_adk/)
cd examples/google_adk
motus deploy --name my-adk-agent agent:root_agent
```

## Examples

All examples use `motus.google_adk.agents.llm_agent.Agent` (which adds
`run_turn` for serve compatibility) as the root agent. Sub-agents can be
vanilla `google.adk` agents.

| Example | Pattern | serve | Cloud | Notes |
|---------|---------|-------|-------|-------|
| [agent.py](agent.py) | Basic agent with a single tool | Yes | Yes | -- |
| [callbacks.py](callbacks.py) | All 6 callback hooks (before/after agent, model, tool) | Yes | Yes | -- |
| [multi_agent.py](multi_agent.py) | Root triage agent delegates to math and writing sub-agents | Yes | Yes | -- |
| [structured_output.py](structured_output.py) | Pydantic `output_schema` with tool-gathered data | Yes | Yes | Response is a JSON string |
| [parallel_functions.py](parallel_functions.py) | Multiple async tools called concurrently | Yes | Yes | -- |
| [fields_output_schema.py](fields_output_schema.py) | List-typed Pydantic output with `output_key` | Yes | Yes | Response is a JSON string |
| [pydantic_argument.py](pydantic_argument.py) | Tools with Pydantic model args (Optional, nested) | Yes | Yes | -- |
| [workflow_triage.py](workflow_triage.py) | Triage → conditional ParallelAgent + summary | Yes | Yes | -- |
| [static_instruction.py](static_instruction.py) | Pure LLM agent with detailed persona (no tools) | Yes | Yes | -- |
| [token_usage.py](token_usage.py) | Text generation with token usage tracking | Yes | Yes | -- |
| [multimodal_tools.py](multimodal_tools.py) | Rich tool outputs: charts, reports, comparisons | Yes | Yes | -- |
| [mcp.py](mcp.py) | MCP tools over HTTP via `MCPToolset` with motus `create_auth` | Yes | Yes | Set `MCP_SERVER_URL`; uses `ConsoleAuth` locally, `DaprAuth` when deployed |

## How it works

`motus.google_adk.agents.llm_agent.Agent` subclasses the Google ADK
`Agent` and adds a `run_turn()` method satisfying the serve agent contract:

- Creates an `InMemoryRunner` per turn
- Replays conversation state into the ADK session
- Runs the agent and collects the final response
- Auto-registers `MotusSpanProcessor` with OTEL for tracing

Sub-agents (`SequentialAgent`, `ParallelAgent`, vanilla `Agent`) work
natively — only the root agent needs to be the motus `Agent` subclass.

### Cloud deployment

When deploying to Motus cloud, include `requirements.txt` with
`google-adk>=1.27.2`. No secrets are needed — the platform automatically
routes Gemini API calls through the model proxy:

```sh
motus deploy --name my-agent agent:root_agent
```
