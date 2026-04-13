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
  Self-managed or cloud deployment, agent serving in one command. Same code, any scale.
</h3>

<p align="center">
  <a href="https://www.lithosai.com/">LithosAI</a> &middot;
  <a href="https://console.lithosai.cloud/">Cloud</a> &middot;
  <a href="https://docs.motus.lithosai.com/">Docs</a> &middot;
  <a href="https://docs.motus.lithosai.com/getting-started/quickstart">Quickstart</a> &middot;
  <a href="https://github.com/lithos-ai/motus/tree/main/examples">Examples</a> &middot;
  <a href="https://docs.motus.lithosai.com/contributing/development-setup">Contributing</a> &middot;
  <a href="https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q">Slack</a>
</p>

## About

Motus is an open-source agent serving project that enables higher capability, lower cost, and faster agents. As building agents has never been easier, Motus takes a no-framework approach and provides the infrastructure needed for efficient agent serving. Deploy simply across self-managed and cloud environments at any scale.

## Use with your coding agent

The fastest way to get started is to let your coding agent handle building, serving, and deploying with Motus.

Motus works out of the box with any coding agent (e.g., Claude Code, Codex, or Cursor). Install the plugin and CLI with one command:

```sh
curl -fsSL https://www.lithosai.com/motus/install.sh | sh
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

Install Motus to serve agents locally and deploy them to [Motus Cloud](https://console.lithosai.cloud/). Motus supports agents built with:

* Motus
*  OpenAI Agents SDK
*  Anthropic SDK
*  Google ADK
*  Plain Python

### Install Motus in your project

Using uv:

```bash
uv add lithosai-motus
```

Or with pip:

```bash
pip install lithosai-motus
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

Motus is powered by a serving runtime that automatically converts Python code into parallel, resilient workflows. Everything is designed to be simple, intuitive, and customizable.

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

Start simple, and explore the [agents documentation](https://docs.motus.lithosai.com/concepts/agents) for more advanced usage.

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

No explicit DAGs, just Python. Motus leverages `@agent_task` decorators to turn Python functions into asynchronous tasks.
Motus sits under your agents, providing scheduling, parallelism, caching, resilience, observability, and tracing. [Learn more about the Motus runtime](https://docs.motus.lithosai.com/concepts/workflow).

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
| **[Agents](https://docs.motus.lithosai.com/concepts/agents)** | `ReActAgent` runs the reasoning loop, tool dispatch, and conversation state. Multi-turn memory, structured output via Pydantic, and input/output guardrails. All built in. A working agent in under 10 lines. |
| **[Tools](https://docs.motus.lithosai.com/concepts/tools)** | Write a function, get a tool. Expose class methods with `@tools`, wrap an MCP server with `get_mcp()`, nest another agent with `as_tool()`, or run untrusted code in a Docker sandbox. Everything composes through the same `tools=[...]` interface. Built-in utilities: skills, `bash`, file ops, `glob` / `grep`, todo tracking. |
| **[Task-graph runtime](https://docs.motus.lithosai.com/concepts/workflow)** | `@agent_task` turns any function into a node in a dependency graph with automatic parallel execution, multi-return futures, non-blocking operators. Retries, timeouts, and backoff are declarative on the task and overridable per call site with `.policy()`. |
| **[Observability & debugging](https://docs.motus.lithosai.com/guides/tracing)** | Every LLM call, tool invocation, and task dependency traced automatically. Interactive HTML viewer, Jaeger export, or cloud dashboard. Enabled with one env var. |
| **[Multi-provider models](https://docs.motus.lithosai.com/concepts/models)** | Unified client for OpenAI, Anthropic, Gemini, and OpenRouter. Switch providers by changing one line, agent logic stays the same. Local models (Ollama, vLLM, SGLang) work through `base_url`. |
| **[Local serving](https://docs.motus.lithosai.com/guides/serving)** | `motus serve` exposes any agent as a session-based HTTP API locally. Test the full serving stack before deploying to the cloud. |

#### Go deeper

| | |
|---|---|
| **[Memory](https://docs.motus.lithosai.com/concepts/memory)** | Provided memory solutions: `basic` (append-only), `compact` (auto-summarizes when token budget runs thin). Session save/restore built in. |
| **[Guardrails](https://docs.motus.lithosai.com/guides/guardrails)** | Input and output validation on both agents and individual tools. Declare the parameters you care about — return a dict to modify, raise to block. Structured output guardrails match fields on Pydantic models. |
| **[Multi-agent composition](https://docs.motus.lithosai.com/guides/multi-agent)** | `agent.as_tool()` wraps any agent as a tool. The supervisor doesn't know whether it's calling a function or another agent — the interface is identical. `fork()` creates independent conversation branches. |
| **[MCP integration](https://docs.motus.lithosai.com/guides/mcp-integration)** | Connect any MCP-compatible server with `get_mcp()`. Local via stdio, remote via HTTP, or inside a Docker container. Filter and rename tools with `prefix`, `blocklist`, and guardrails. |
| **[Docker sandboxes](https://docs.motus.lithosai.com/concepts/tools)** | Run untrusted code in isolated containers. Mount volumes, expose ports, execute shell and Python — attach to any agent as a tool provider. |
| **[Prompt caching](https://docs.motus.lithosai.com/concepts/models)** | Prompt caching via `CachePolicy` — `STATIC` (system + tools) or `AUTO` (+ conversation prefix). Reduce latency and cost on long conversations. |
| **SDK compatibility** | Drop-in for [OpenAI Agents SDK](https://docs.motus.lithosai.com/integrations/openai-agents), [Anthropic SDK](https://docs.motus.lithosai.com/integrations/anthropic-sdk), and [Google ADK](https://docs.motus.lithosai.com/integrations/google-adk). Change the import, keep your code. |
| **[Human-in-the-loop](https://docs.motus.lithosai.com/guides/human-in-the-loop)** | Built-in support for interactive approval, clarification, and feedback during agent execution. Pause the agent, ask for human input, and resume. Works in both local serving and cloud deployment. |


---

## Contributing

See the **[Contributing Guide](https://docs.motus.lithosai.com/contributing/development-setup)** to get started, or come say hi on [Slack](https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q). Let's build together!

## License

Apache 2.0 — see [LICENSE](LICENSE).
