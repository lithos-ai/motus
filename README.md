<!-- # Motus -->

<!-- TODO: commit logo to assets/ and replace with repo-relative or raw.githubusercontent path -->
<p align="center">
  <img alt="Motus" src="assets/motus.png" />
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
  <a href="https://github.com/lithos-ai/motus/releases"><img alt="Release" src="https://img.shields.io/github/v/release/lithos-ai/motus" /></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg" /></a>
  <a href="https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q"><img alt="Slack" src="https://img.shields.io/badge/Slack-community-purple?logo=slack" /></a>
  <!-- TODO: add CI badge once URL is live -->
  <!-- <a href="https://github.com/lithos-ai/motus/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/lithos-ai/motus/ci.yml?branch=main" /></a> -->
</p>

<h3 align="center">
  Higher capability. Lower cost. Faster agents.<br/>
  Deploy locally or to the cloud in one command. Same code, any scale.
</h3>

<p align="center">
  <a href="https://www.lithosai.com/">LithosAI</a> &middot;
  <a href="http://console.lithosai.com/">Cloud</a> &middot;
  <a href="https://motus.readthedocs.io/">Docs</a> &middot;
  <a href="https://motus.readthedocs.io/getting-started/quickstart/">Quickstart</a> &middot;
  <a href="https://motus.readthedocs.io/examples/">Examples</a> &middot;
  <a href="https://motus.readthedocs.io/contributing/development-setup/">Contributing</a> &middot;
  <a href="https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q">Slack</a>
</p>

## About

Agentic inference is exploding. Motus is an open-source agent serving project that enables higher capability, lower cost, and faster agents---while making deployment simple across local and cloud environments at any scale.

## Use with your coding agent

The fastest way to get started is to let your coding agent handle building, serving, and deploying with Motus.

Motus works out of the box with any coding agent (e.g., Claude Code, Codex, or Cursor). Install the plugin with one command:

```sh
curl -fsSL https://www.lithosai.com/install.sh | sh
```

Then use it directly in your workflow:

```
/motus                          # activate Motus skills

build your agent                # start building your agent

/motus serve                    # serve locally

/motus deploy                   # deploy to the cloud
```

See [`plugins/motus/README.md`](plugins/motus/README.md) for marketplace installs and more details.



## Serve & deploy any agent

Install Motus to serve agents locally and deploy them to [Motus Cloud](http://console.lithosai.com/). Motus supports agents built with:

* Motus
*  OpenAI Agents SDK
*  Claude Agent SDK
*  Google ADK
*  Plain Python

### Install the Motus Python library and CLI tool

Using uv:

```bash
uv add motus
```

Or with pip:

```bash
pip install motus
```

### Serve locally and deploy to the cloud

```bash
# Serve locally
motus serve start myapp:agent --port 8000

# Chat with your local agent
motus serve chat http://localhost:8000 "Hello!"

# Deploy to Motus Cloud
motus deploy --name myapp myapp:agent

# Chat with your deployed agent
motus serve chat https://myapp.lithosai.com "Hello!"
```

## Build with Motus

Motus provides a complete agent toolkit---including agents, tools, memory, guardrails, and tracing---powered by a runtime that automatically converts Python code into parallel, resilient workflows. Everything is designed to be simple, intuitive, and customizable.

### Build an agent

```python
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.runtime import resolve
from motus.tools import tool

@tool  # define a simple tool
async def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# define a ReAct agent
agent = ReActAgent(client=OpenAIChatClient(), model_name="gpt-4o", tools=[search])
print(resolve(agent("Hello World!")))
```

Start simple, and explore the [agents documentation](docs/user-guide/agents.md) for more advanced usage.

### Build a workflow

Example: fetch an article, summarize it, extract hashtags in parallel, then publish:

```python
from motus.runtime import resolve
from motus.runtime.agent_task import agent_task

@agent_task # wrap functions as tasks in your workflow
async def summarize(article): ... # just a normal function

@agent_task
async def extract(article): ... # extract hashtags

@agent_task(retries=3, timeout=10.0) # augment tasks with retries and timeouts
async def fetch(url): ...

@agent_task
async def publish(summary, hashtags): ... # publish on LinkedIn

# Your logic becomes your code directly:
article = fetch("https://www.lithosai.com")
summary = summarize(article)            # Motus infers the dependency graph from data flow.
hashtags = extract(article)             # Both depend on `article`, run in parallel.
post = publish(summary, hashtags)       # Waits for both upstream tasks.

print(resolve(post)) # get final result
```

No explicit DAGs—just Python. Motus leverages `@agent_task` decorators to turn Python functions into asynchrous tasks.
Motus handles scheduling, parallelism, caching, resilience, tracing, and so on. [Learn more about the Motus runtime](docs/user-guide/runtime.md).

### Examples

Run the included examples:

```bash
# Basic ReAct agent — interactive console chat
uv run python examples/agent.py

# Task graph demo — parallelism, dependency tracking, multi-return
uv run python examples/runtime/task_graph_demo.py
```

Learn more from our [comprehensive examples](examples/).

### Motus features

#### Start simple

| | |
|---|---|
| **[Agents](docs/user-guide/agents.md)** | `ReActAgent` runs the reasoning loop, tool dispatch, and conversation state. Multi-turn memory, structured output via Pydantic, and input/output guardrails — all built in. A working agent in under 10 lines. |
| **[Tools](docs/user-guide/tools.md)** | Write a function, get a tool. Expose class methods with `@tools`, wrap an MCP server with `get_mcp()`, nest another agent with `as_tool()`, or run untrusted code in a Docker sandbox — everything composes through the same `tools=[...]` interface. Built-in utilities: skills, `bash`, file ops, `glob` / `grep`, todo tracking. |
| **[Task-graph runtime](docs/user-guide/runtime.md)** | `@agent_task` turns any function into a node in a dependency graph — automatic parallel execution, multi-return futures, non-blocking operators. Retries, timeouts, and backoff are declarative on the task and overridable per call site with `.policy()`. |
| **[Multi-provider models](docs/user-guide/models.md)** | Unified client for OpenAI, Anthropic, Gemini, and OpenRouter. Switch providers by changing one line — agent logic stays the same. Local models (Ollama, vLLM) work through `base_url`. |
| **[Tracing & debugging](docs/user-guide/tracing.md)** | Every LLM call, tool invocation, and task dependency traced automatically. Interactive HTML viewer, Jaeger export, or cloud dashboard — enabled with one env var. |
| **[Local serving](docs/user-guide/serving.md)** | `motus serve` exposes any agent as a session-based HTTP API locally. Test the full serving stack before deploying to the cloud. |

#### Go deeper

| | |
|---|---|
| **[Memory](docs/user-guide/memory.md)** | Provided memory solutions: `basic` (append-only), `compact` (auto-summarizes when token budget runs thin). Session save/restore built in. |
| **[Guardrails](docs/user-guide/guardrails.md)** | Input and output validation on both agents and individual tools. Declare the parameters you care about — return a dict to modify, raise to block. Structured output guardrails match fields on Pydantic models. |
| **[Multi-agent composition](docs/user-guide/agents.md)** | `agent.as_tool()` wraps any agent as a tool. The supervisor doesn't know whether it's calling a function or another agent — the interface is identical. `fork()` creates independent conversation branches. |
| **[MCP integration](docs/user-guide/mcp-integration.md)** | Connect any MCP-compatible server with `get_mcp()` — local via stdio, remote via HTTP, or inside a Docker container. Filter and rename tools with `prefix`, `blocklist`, and guardrails. |
| **[Docker sandboxes](docs/user-guide/tools.md)** | Run untrusted code in isolated containers. Mount volumes, expose ports, execute shell and Python — attach to any agent as a tool provider. |
| **[Prompt caching](docs/user-guide/models.md)** | Prompt caching via `CachePolicy` — `STATIC` (system + tools) or `AUTO` (+ conversation prefix). Reduce latency and cost on long conversations. |
| **SDK compatibility** | Drop-in for [OpenAI Agents SDK](docs/integrations/openai-agents.md), [Claude Agent SDK](docs/integrations/claude-agent.md), and Google ADK. Change the import, keep your code — get tracing and cloud deployment for free. |
| **[Lifecycle hooks](docs/user-guide/tracing.md)** | Three-level hook system (global, per-task name, per-task type). Tap into `task_start`, `task_end`, `task_error` for logging, metrics, or custom logic. |

---

## Contributing

**Open source from Day 1.** We believe the infrastructure for agentic inference should be open.
See the **[Contributing Guide](docs/contributing/development-setup.md)** to get started, or come say hi on [Slack](https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q). Let's build together!

## License

Apache 2.0 — see [LICENSE](LICENSE).
