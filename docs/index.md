# Motus

Build AI agents in Python. Deploy them to the cloud in one command.

Motus handles the full lifecycle — from local prototyping to production-scale serving — so you focus on agent logic, not infrastructure.

=== "ReAct Agent"

    ```python
    @tool
    async def lookup(query: str) -> str:
        """Look up information about a topic."""
        return f"Result for: {query}"

    agent = ReActAgent(client, "gpt-4o", tools=[lookup])
    response = await agent("What can you tell me about Motus?")
    ```

=== "Task-Graph Workflow"

    ```python
    @agent_task(retries=3, timeout=10.0)
    async def fetch(url): ...

    @agent_task
    async def summarize(text_a, text_b): ...

    # No DAG definition, no edge wiring — motus infers the graph from data flow.
    a = fetch("https://source-a.com")  # ╮
    b = fetch("https://source-b.com")  # ├─ parallel (independent args)
    result = summarize(a, b)            # ╰─ waits for both (a, b are deps)
    ```


```bash
# One command to go from local code to cloud endpoint
motus deploy my-project myapp:agent
```

## Why Motus

### Ship to production, not just a notebook

Most agent frameworks stop at local execution. Motus gives you `motus serve` to expose agents as session-based HTTP APIs and `motus deploy` to push them to managed cloud infrastructure — auto-scaling, load balancing, zero Docker knowledge required.

### See what your agents are doing

Built-in distributed tracing captures every LLM call, tool invocation, and task dependency in real time. View execution traces in a live HTML viewer, export to OpenTelemetry/Jaeger, or stream to the Motus cloud dashboard. No blind spots.

### Two ways to build: ReAct or workflow

Use `ReActAgent` when the LLM should decide which tools to call. Use `@agent_task` to compose deterministic multi-step pipelines — no DAG definitions, no edge wiring, no scaffolding. Decorate your functions and pass outputs as inputs; Motus infers the dependency graph automatically. Mix both in the same application — an agent task can invoke a ReAct agent, and a ReAct agent can delegate to a task graph.

### Production-ready from day one

Guardrails validate and transform agent inputs and outputs — block unsafe content, redact PII, enforce schemas — without touching your agent logic. Tools are first-class: wrap any Python function with `@tool`, connect MCP servers for external capabilities, hook into task lifecycle events, or run untrusted code in Docker sandboxes. Everything composes through the same interface.

### Scale without rearchitecting

The async task-graph runtime tracks dependencies, retries failures, and parallelizes where possible. Go from a single agent to multi-agent orchestration without rewriting your code.

### Meta-agent optimization (coming soon)

A built-in meta-agent continuously analyzes your agent's traces, user interactions, and feedback to suggest concrete improvements — better prompts, tighter tool schemas, cheaper model routing, fewer redundant steps. It debugs failures, identifies cost hotspots, and recommends configuration changes so your agents get better over time without manual tuning.

### Agent-aware serving infrastructure (coming soon)

The Motus serving layer is designed around agent workloads, not generic HTTP traffic. Adaptive batching, intelligent request routing, and auto-scaling policies that respond to agent-level metrics — token throughput, tool latency, step count — rather than raw request volume.

## Key Features

- **Cloud deployment** — `motus deploy` packages and ships your agent to production with one command
- **Production serving** — session-based HTTP APIs with worker pools, TTL management, and webhook callbacks
- **Observability** — distributed tracing with live viewer, OpenTelemetry export, and configurable collection levels
- **Two agent paradigms** — ReAct loop for autonomous tool use; `@agent_task` for lightweight dependency-tracked workflows with zero boilerplate
- **Guardrails** — input/output validation, PII redaction, schema enforcement on agents and individual tools
- **Composable tools** — `@tool` decorator wraps any function; MCP sessions connect external tool servers; Docker sandboxes isolate untrusted code
- **Lifecycle hooks** — tap into `task_start`, `task_end`, `task_error` for custom logging, metrics, or side effects
- **Multi-provider models** — unified client for OpenAI, Anthropic, Gemini, and OpenRouter
- **Persistent memory** — basic and compaction strategies with boundary-aware auto-compaction and session restore
- **Meta-agent** :material-clock-fast:{ title="Coming soon" } — analyzes traces and user feedback to optimize prompts, reduce cost, debug failures, and strengthen agent configurations
- **Agent-aware infra** :material-clock-fast:{ title="Coming soon" } — serving infrastructure with adaptive batching, intelligent routing, and auto-scaling tuned to agent-level metrics

## Getting Started

<div class="grid cards" markdown>

- **Installation**

    Install Motus and set up your environment.

    [:octicons-arrow-right-24: Install](getting-started/installation.md)

- **Quickstart**

    Build and run your first agent in under 5 minutes.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

- **Deploy to Cloud**

    Go from local agent to production endpoint.

    [:octicons-arrow-right-24: Deployment](user-guide/deployment.md)

- **Observability**

    Trace and monitor agent execution in real time.

    [:octicons-arrow-right-24: Tracing](user-guide/tracing.md)

</div>

## Already using another agent runtime?

Motus integrates with existing agent SDKs. Bring your OpenAI Agents or Claude Agent SDK agents and deploy them through Motus with full observability and cloud serving — no rewrite needed.

=== "OpenAI Agents SDK"

    ```python
    from motus.openai_agents import Agent, Runner

    agent = Agent(name="my-agent", instructions="You are helpful.")
    result = await Runner.run(agent, "Hello!")
    ```

=== "Claude Agent SDK"

    ```python
    from motus.claude_agent import query

    async for message in query(prompt="Summarize this document."):
        print(message)
    ```

[:octicons-arrow-right-24: OpenAI Agents integration](integrations/openai-agents.md) | [:octicons-arrow-right-24: Anthropic SDK integration](integrations/anthropic.md) | [:octicons-arrow-right-24: Google ADK integration](integrations/google-adk.md)

## Learn More

| Section | What you will find |
|---------|-------------------|
| [User Guide](user-guide/overview.md) | Architecture, agents, tools, models, memory, serving |
| [Integrations](integrations/openai-agents.md) | Bridges to OpenAI Agents SDK, Anthropic SDK, and Google ADK |
| [Examples](examples/index.md) | Runnable demos: runtime patterns, MCP, multi-agent bots |
| [API Reference](api/index.md) | Auto-generated reference for every public class and function |
| [Contributing](contributing/development-setup.md) | Dev environment, tests, PRs |
