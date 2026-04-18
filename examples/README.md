# Examples

These Motus examples demonstrate its various features, including support for
third-party agent-programming frameworks (SDKs). They are organized into
Motus-specific features, third-party SDK support, and a full Deep Research
implementation.

## Quick start

[`agent.py`](agent.py) — Minimal ReAct agent with a stateless chat loop. Start here.
Deployable via `motus deploy agent:agent`.

```sh
python agent.py
```

## Features

| Feature | Examples | Deployable |
|---------|----------|------------|
| MCP tools | [`mcp_tools.py`](mcp_tools.py) — Lazy init, context managers, prefixed tools, sandbox MCP, remote HTTP | No |
| Memory | [`memory.py`](memory.py) — Compaction memory with auto-compact, session restore | No |
| Skills | [`skills/`](skills/) — On-demand agent instructions loaded from SKILL.md directories | No |
| Runtime | [`runtime/`](runtime/) — Agent-as-tool, guardrails, hooks, resilient tasks, task graphs | No |
| Serving | [`serving/`](serving/) — Agent and custom entrypoints for `motus serve` | Yes |

## Third-party SDK support

Motus supports third-party agent SDKs by wrapping agent abstractions to expose
a serve-compatible stateless interface. Furthermore, the cloud platform
transparently proxies model API requests obviating the need for third-party API
keys unless serving locally.

| SDK | Examples | Deployable |
|-----|----------|------------|
| Anthropic SDK | [`anthropic/`](anthropic/) — Tool runner, deferred tool search, MCP | Yes |
| OpenAI Agents SDK | [`openai_agents/`](openai_agents/) — Tools, routing, agents-as-tools, guardrails, message filtering, MCP | Yes |
| Google ADK | [`google_adk/`](google_adk/) — Tools, callbacks, multi-agent, structured output, parallel tools, triage workflows, MCP | Yes |

## Multi-SDK example

[`deep_research/`](deep_research/) — A deep research agent implemented across
three SDK backends (motus native, Google ADK, OpenAI Agents). Each variant
includes a `serve.py` entrypoint and is deployable.
