# Examples

Feature examples for the Motus framework. Each demonstrates a specific
capability in isolation.

## Quick start

[`agent.py`](agent.py) — Minimal ReAct agent with a stateless chat loop. Start here.

```sh
python examples/agent.py
```

## Features

| Feature | Examples |
|---------|----------|
| MCP tools | [`mcp_tools.py`](mcp_tools.py) — Lazy init, context managers, prefixed tools, sandbox MCP, remote HTTP |
| Memory | [`memory.py`](memory.py) — Compaction memory, session restore |
| Runtime | [`runtime/`](runtime/) — Agent-as-tool, guardrails, hooks, resilient tasks, task graphs |
| Serving | [`serving/`](serving/) — Agent and custom entrypoints for `motus serve` |

## Third-party SDK support

| SDK | Examples | serve | Cloud deploy |
|-----|----------|-------|--------------|
| Anthropic SDK | [`anthropic/`](anthropic/) — Tool runner, deferred tool search | Yes | Yes |
| OpenAI Agents SDK | [`openai_agents/`](openai_agents/) — Tools, routing, agents-as-tools, guardrails, message filtering | Yes | Yes |
| Google ADK | [`google_adk/`](google_adk/) — Tools, callbacks, multi-agent, structured output, parallel tools, triage workflows | Yes | Yes |
