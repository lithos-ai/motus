---
name: motus
description: Build, configure, and deploy AI agents using the Motus framework. Use when user wants to create agents, define tools, set up workflows, configure memory or guardrails, or deploy agents locally or to the cloud. Triggers on mentions of motus, ReActAgent, agent_task, tool creation, MCP integration, motus deploy, motus serve.
argument-hint: "[deploy [--name name] [import-path]] or [deploy [--project-id id] [import-path]] or [serve [import-path]] or [description of agent to build]"
---

# Motus

You are an expert in the Motus AI agent framework. You help users build and deploy agent applications.

## Command routing

Parse the user's arguments to determine the mode:

- **First argument is `deploy`** → go to [Cloud Deploy](#cloud-deploy), pass remaining arguments
- **First argument is `serve`** → go to [Local Serve](#local-serve), pass remaining arguments
- **Anything else** (no args, or a description of what to build) → go to [Build](#build)

There are **three distinct ways to run an agent** — make sure you and the user are aligned on which one:

| Mode | What it does | When to use |
|------|-------------|-------------|
| **CLI interaction** | Run the agent directly in the terminal (`uv run python agent.py`) | Quick testing, development, one-off conversations |
| **Local serve** | Start an HTTP server on the user's machine (`motus serve start`) | Local API testing, multi-session usage, integration testing |
| **Cloud deploy** | Deploy to LITHOSAI cloud (`motus deploy`) | Production, sharing with others, persistent hosting |

Examples:
- `/motus` → Build (interactive)
- `/motus I need a customer support agent` → Build
- `/motus deploy` → Cloud Deploy (auto-detect; uses motus.toml if available)
- `/motus deploy myapp:my_agent` → Cloud Deploy with import path
- `/motus deploy --name my-app myapp:my_agent` → First cloud deploy (creates new project)
- `/motus deploy --project-id abc123 myapp:my_agent` → Cloud Deploy to existing project by ID
- `/motus serve` → Local Serve (auto-detect)
- `/motus serve myapp:my_agent` → Local Serve with import path

---

# Build

Your job is to understand the user's requirements and help them build a fully functional agent application using Motus.

## Before writing any code

Infer as much as possible from what the user already said and from the project context (existing code, dependencies, env vars). **Do not ask questions you can answer from context.** Start building and let the user course-correct.

**Choosing a framework** — pick based on context, don't ask:

- If the user mentions a specific SDK (Anthropic, OpenAI Agents, Google ADK), use that SDK's wrapper.
- Otherwise, default to `motus.agent.ReActAgent` with `OpenAIChatClient`.

| User says | Framework |
|-----------|-----------|
| "Anthropic SDK" | `motus.anthropic.ToolRunner` |
| "OpenAI SDK" / "OpenAI Agents" | `motus.openai_agents.Agent` |
| "Google ADK" / "Gemini" | `motus.google_adk.agents.llm_agent.Agent` |
| No preference / "Motus" | `motus.agent.ReActAgent` (use `OpenAIChatClient` with any model like `"anthropic/claude-sonnet-4.5"` or `"gpt-4o"`) |

**Important:** `ReActAgent` is a generic agent loop that uses model clients (`OpenAIChatClient`, etc.) as backends. It is *not* the same as using a provider's native SDK. When the user asks for a specific provider's SDK, use that provider's dedicated wrapper above — not `ReActAgent` with the provider's client.

When the agent is ready to **deploy**: if the user's original request includes deploying, proceed directly to [Cloud Deploy](#cloud-deploy) — do not stop and ask the user to run `/motus deploy` separately. "Deploy" always means cloud deployment. If the user asks to "test" or "try" the agent without specifying deploy, suggest CLI interaction or local serve instead.

**Always prefer `uv`** for package management (`uv add`, `uv sync`). Use `uv run` for running user scripts (e.g. `uv run python agent.py`), but not for the `motus` CLI — it is installed globally via `uv tool install` and available directly as `motus`.

**Python version** — When creating a new project, pin Python 3.12 (e.g. `uv init --python 3.12` or `requires-python = ">=3.12"` in pyproject.toml). The cloud runtime supports up to Python 3.13. Do not use Python 3.14, which uv may select by default.

## Environment check

Before writing any agent code, verify the user's environment is ready. Run these checks silently and only report problems:

1. **Motus installed** — Check that `lithosai-motus` is installed as a project dependency. If not, install it:
   ```bash
   uv add lithosai-motus
   ```
2. **API keys** — LLM provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.) are **only needed for local testing**, not for cloud deployment. If no keys are set, **do not block** — note it and continue. The user can still build the agent code and deploy to the cloud, where the platform's model proxy provides all LLM credentials automatically. Only ask about API keys if the user explicitly wants to run the agent locally.
3. **Optional: Docker** — Only check if the user needs sandbox or MCP-in-container features. `docker info` should succeed.

If everything is fine, proceed without mentioning the checks.

## Quick reference

See these files for detailed API reference and patterns:

- [REFERENCE.md](REFERENCE.md) — Complete API signatures, constructors, all parameters
- [PATTERNS.md](PATTERNS.md) — Proven code patterns for common scenarios
- [EXAMPLES.md](EXAMPLES.md) — End-to-end example applications

## Core concepts (brief)

| Concept | What it is | Key import |
|---------|-----------|------------|
| `ReActAgent` | Autonomous agent with reasoning + tool-use loop (has `run_turn` for deploy) | `from motus.agent import ReActAgent` |
| `ToolRunner` | Anthropic SDK wrapper with tool execution (has `run_turn` for deploy) | `from motus.anthropic import ToolRunner` |
| Google ADK `Agent` | Google ADK wrapper (has `run_turn` for deploy) | `from motus.google_adk.agents.llm_agent import Agent` |
| OpenAI Agents `Agent` | OpenAI Agents SDK wrapper (auto-adapted for deploy) | `from motus.openai_agents import Agent` |
| `@agent_task` | Decorator turning functions into dependency-tracked async tasks | `from motus.runtime import agent_task` |
| `@tool` | Universal tool decorator (works with ReActAgent + Anthropic ToolRunner) | `from motus.tools import tool` |
| `MCPSession` | Connect to external MCP tool servers | `from motus.tools import get_mcp` |
| `Sandbox` | Docker container for code execution (**local-only**) | `from motus.tools import get_sandbox` |
| Guardrails | Input/output validators on agents and tools | `from motus.guardrails import *` |
| Memory | Conversation history management (basic or compaction) | `from motus.memory import *` |
| Hooks | Task lifecycle callbacks (start/end/error) | `from motus.runtime.hooks import register_hook` |

## Deployable agent types

Not all agent configurations can be deployed to the cloud. Use this to guide what you build:

| Agent type | CLI | Local serve | Cloud deploy | Notes |
|-----------|-----|-------------|-------------|-------|
| **Conversational** (customer support, Q&A, assistants) | Yes | Yes | Yes | Most common. Uses tools for API calls, lookups, etc. |
| **Research / pipeline** (web search, multi-step reasoning) | Yes | Yes | Yes | Uses `@tool` functions and `@agent_task` workflows |
| **Multi-agent** (orchestrator + specialists) | Yes | Yes | Yes | Uses `agent.as_tool()` for delegation |
| **MCP-connected** (external tool servers) | Yes | Yes | Yes | Uses `get_mcp()` — MCP server must be network-accessible from cloud |
| **Coding / sandbox** (code execution in Docker) | Yes | Yes | **No** | `get_sandbox()` requires local Docker — not available in cloud |

When the user asks to build an agent, **infer the type from their description** — do not ask them to pick from this list. If they describe something that requires a sandbox (code execution, running scripts, etc.), note that it will only work locally and offer to proceed with CLI or local serve mode.

## Workflow: building an agent application

### Step 1: Choose your framework and define tools

Pick the framework based on the user's preferred SDK (see guidance in "Before writing any code" above). Each framework has its own tool format:

**Motus ReActAgent** — uses `@tool` decorator:
```python
from motus.tools import tool

@tool
async def my_tool(param: str) -> str:
    """Description the LLM sees."""
    return result
```

**Anthropic ToolRunner** — uses `BetaAsyncFunctionTool` from the Anthropic SDK:
```python
from anthropic.lib.tools import BetaAsyncFunctionTool

async def my_tool(param: str) -> str:
    """Description the LLM sees.

    Args:
        param: Description of the parameter.
    """
    return result

tools = [BetaAsyncFunctionTool(my_tool)]
```

**OpenAI Agents** — uses `@function_tool` decorator:
```python
from motus.openai_agents import function_tool

@function_tool
def my_tool(param: str) -> str:
    """Description the LLM sees."""
    return result
```

**Google ADK** — uses plain functions:
```python
def my_tool(param: str) -> str:
    """Description the LLM sees."""
    return result
```

For complex inputs, see [PATTERNS.md](PATTERNS.md). For external tool servers, use `get_mcp()`. For local code execution, use `get_sandbox()` (local-only — not available in cloud deploy).

### Step 2: Create the agent

**Motus ReActAgent** (generic loop, any provider via OpenAI-compatible client):
```python
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient

client = OpenAIChatClient()  # Cloud proxy auto-provides API keys
agent = ReActAgent(client=client, model_name="gpt-4o", system_prompt="You are ...", tools=[my_tool], max_steps=10)
```

**Anthropic ToolRunner** (native Anthropic SDK tool-use):
```python
from motus.anthropic import ToolRunner

agent = ToolRunner(model="claude-sonnet-4-20250514", max_tokens=1024, tools=tools, system="You are ...")
```

**OpenAI Agents** (OpenAI Agents SDK):
```python
from motus.openai_agents import Agent

agent = Agent(name="my_agent", model="gpt-4.1", instructions="You are ...", tools=[my_tool])
```

**Google ADK** (Google ADK):
```python
from motus.google_adk.agents.llm_agent import Agent

agent = Agent(model="gemini-2.0-flash", name="my_agent", description="...", instruction="You are ...", tools=[my_tool])
```

### Step 3: Run

All of the above agent instances support all three run modes. See [Serve Contract & Framework Support](#serve-contract--framework-support) for details.

**CLI interaction** (quickest way to test):
```bash
uv run python agent.py
```
Add a `__main__` block to the agent file for interactive CLI usage (see patterns below).

**Local serve** (HTTP server):
```bash
motus serve start agent:my_agent --port 8000
```

**Cloud deploy** (production):
```bash
motus deploy --name my-app agent:my_agent    # first deploy (creates project)
motus deploy                                  # subsequent deploys (reads motus.toml)
```

## Critical rules

- **For `ReActAgent`: always pass `client` as the first argument** — Not a model name string.
- **Use `await` for agent calls in async context** — `response = await agent("prompt")`.
- **Guardrail functions declare only the parameters they inspect** — They don't need to match the full tool signature.
- **MCP sessions are lazy by default** — They connect on first tool call when passed to an agent.

---

# Local Serve

Start a local HTTP server for the agent. This is useful for API testing, multi-session usage, and integration testing before cloud deployment.

Usage:
- `/motus serve` — auto-detect agent from files
- `/motus serve myapp:my_agent` — serve with import path

## S0. Detect & Confirm

**No arguments provided:**

Scan `.py` files in the current directory for likely agent entry points. Look for:
1. Agent instances (ReActAgent, ToolRunner, ADK Agent, OpenAI Agent)
2. Functions matching the serve contract: `def xxx(message, state)` or `async def xxx(message, state)` (see [Agent Function Contract](#agent-function-contract))
3. Files named `agent.py`, `app.py`, `server.py`, `main.py`

If candidates are found, you MUST call the `AskUserQuestion` tool so the user gets clickable options:

```json
{"question": "I found these agent entry points. Which one to serve?", "options": ["agent:my_agent", "app:serve", "Let me specify manually"]}
```

## S1. Validate

- Verify import path contains `:` (format: `module:callable`)
- Verify the referenced module file exists in the current directory
- **Validate the agent function signature** — see [Agent Function Contract](#agent-function-contract) below. If the function does not conform, help the user fix it before proceeding.

## S2. Start Server

```bash
motus serve start $IMPORT_PATH --port 8000
```

Optional flags to offer the user:
- `--port <N>` — bind port (default 8000)
- `--workers <N>` — worker processes (default CPU count)
- `--ttl <seconds>` — idle session TTL
- `--timeout <seconds>` — max seconds per agent turn
- `--log-level debug` — verbose logging

## S3. Test

Once the server is running, suggest testing in another terminal:

```bash
motus serve chat http://localhost:8000 "hello"   # single message
motus serve chat http://localhost:8000            # interactive REPL
motus serve health http://localhost:8000          # health check
```

For post-serve interaction details, see [POST-DEPLOY.md](POST-DEPLOY.md).

---

# Cloud Deploy

Deploy the agent to LITHOSAI cloud for production hosting. **"Deploy" always means cloud deployment** — for local usage, see [Local Serve](#local-serve) or use CLI interaction.

Usage:
- `/motus deploy` — auto-detect agent from files, deploy to cloud
- `/motus deploy myapp:my_agent` — deploy specific import path to cloud
- `/motus deploy --project-id my-project myapp:my_agent` — deploy to specific cloud project

For deploy troubleshooting, see [DEPLOY-REFERENCE.md](DEPLOY-REFERENCE.md). For post-deploy interaction, see [POST-DEPLOY.md](POST-DEPLOY.md).

## C0. Detect & Confirm

**No arguments provided:**

Scan `.py` files in the current directory for likely agent entry points. Look for:
1. Agent instances (ReActAgent, ToolRunner, ADK Agent, OpenAI Agent)
2. Functions matching the serve contract: `def xxx(message, state)` or `async def xxx(message, state)` (see [Agent Function Contract](#agent-function-contract))
3. Files named `agent.py`, `app.py`, `server.py`, `main.py`

If candidates are found, you MUST call the `AskUserQuestion` tool so the user gets clickable options:

```json
{"question": "I found these agent entry points. Which one to deploy?", "options": ["agent:my_agent", "app:serve", "Let me specify manually"]}
```

If no candidates found, you MUST call `AskUserQuestion` tool with just a question (no options) to ask for the import path.

### C1. Validate

- Verify that either `--name`, `--project-id`, or a `project_id` in `motus.toml` is available
- Verify import path is provided (via argument or `motus.toml`) and contains `:` (format: `module:callable`)
- Verify the referenced module file exists in the current directory
- **Validate the agent function signature** — see [Agent Function Contract](#agent-function-contract) below. If the function does not conform, help the user fix it before proceeding.
- Ensure the user is authenticated by running `motus whoami`. If the output indicates the user is not logged in, run `motus login` directly (do **not** ask the user to run it themselves). The command will print a URL for the user to open if browser auth is needed, then wait and return once credentials are saved. After `motus login` completes, proceed with the deploy — no further env-var setup is required.

### C1.5. Cloud Model Proxy — No Code Changes Needed

The Motus cloud platform provides a **transparent model proxy** that automatically routes API calls for all supported SDKs. When an agent is deployed, the platform auto-wires the necessary environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `GOOGLE_API_KEY`, `GOOGLE_GEMINI_BASE_URL`) so that SDK clients work without any code changes.

**This means:**
- No need to hardcode API keys or base URLs
- No need to pass `--secret` flags during deploy
- Code that works locally with env vars works identically in the cloud
- All three SDKs (OpenAI, Anthropic, Google) are supported transparently

**Supported proxy endpoints:**
- `/v1/chat/completions` — OpenAI Chat Completions API (via OpenRouter)
- `/v1/responses` — OpenAI Responses API (direct to OpenAI)
- `/v1/messages` — Anthropic Messages API (via OpenRouter)
- `/v1beta/` — Google GenAI API (direct to Google)

**The only check needed:** Ensure the agent code does not hardcode API keys or base URLs that would conflict with the auto-wired values. Standard SDK patterns (`OpenAIChatClient()`, `AsyncAnthropic()`, Gemini via `google.genai`) all pick up the env vars automatically.

### C1.6. Local Smoke Test Before Cloud Deploy (Optional)

If LLM provider API keys are available locally, you can catch errors early by running the agent with `motus serve start` before uploading. **Skip this step if no LLM provider keys are set** — the cloud platform provides them automatically, so a missing local key is not a reason to block deployment.

```bash
motus serve start $IMPORT_PATH --port 8000
```

Then in another terminal:

```bash
motus serve chat http://localhost:8000 "hello"
```

If the agent responds correctly, proceed to cloud deploy. If it fails with an import error or signature mismatch, fix the issue locally first. If it fails only due to a missing API key, that's fine — proceed to cloud deploy.

Common failures at this stage:
- Import errors (missing dependencies, typos) — fix before deploying
- Handler signature mismatches — fix before deploying
- LLM client connection errors (missing API key env var) — **OK to skip, cloud provides these**

### C1.7. Check Project Dependencies (Third-Party Packages)

The cloud build installs the base motus package (**no extras**), then installs project deps from `requirements.txt`, `pyproject.toml`, `uv.lock`, or `pylock.toml`. If none of these exist, only base motus is available.

The motus SDK integrations are **optional extras** that pull in separate packages:

| Import found | Underlying package needed | Add to requirements.txt |
|---|---|---|
| `from motus.openai_agents import ...` or `from agents import ...` | `openai-agents` (motus[openai-agents] extra) | `openai-agents` |
| Any other third-party import (e.g. `requests`, `pandas`) | that package | the package name |

> **Note**: Do NOT write `motus[openai-agents]` in requirements.txt — motus is already installed by the cloud build; you just need the underlying packages (e.g. `openai-agents`).

Scan `.py` files in the project for these imports. Then check if a dependency file exists:

1. **`requirements.txt` exists** — check that the required packages are listed. If missing, warn the user and offer to add them.
2. **`pyproject.toml` exists** — check `[project.dependencies]`. If missing, warn and offer to add.
3. **No dependency file exists** — warn the user and use `AskUserQuestion`:
   ```json
   {"question": "No requirements.txt found. Create one with the needed dependencies?", "options": ["Yes, create it for me", "No, I'll handle it myself"]}
   ```

   If yes, create `requirements.txt` with the detected packages. Record the file creation so it can optionally be cleaned up after deploy.

### C2. Rewrite SDK Imports (if needed)

Scan `.py` files for direct SDK imports that should use motus wrappers (the cloud build installs motus, so wrappers are available). Motus wrappers are drop-in — they re-export all symbols via `*` import.

| Direct import | Replacement |
|---|---|
| `from agents import ...` | `from motus.openai_agents import ...` |
| `import agents` | `import motus.openai_agents as agents` |

If found: list the files/lines, explain the change is temporary for deploy, wait for user confirmation, then make replacements and record originals for revert.

### C3. Deploy

**First deploy** (no `motus.toml` yet):
```bash
motus deploy --name $PROJECT_NAME $IMPORT_PATH
```

**Subsequent deploys** (`motus.toml` has `project_id` from a previous deploy):
```bash
motus deploy
```

**Targeting a specific project by ID:**
```bash
motus deploy --project-id $PROJECT_ID $IMPORT_PATH
```

> **Important:** After the first deploy, `motus.toml` is created with `project_id` and `import_path`. Always check for `motus.toml` before choosing which form to use — if it exists with a `project_id`, just run `motus deploy` with no flags. Do NOT pass `--name` on every deploy, as this creates a new project each time.

No `--secret` flags are needed — the platform's model proxy automatically provides API keys for all supported LLM providers. Add `--secret KEY=VALUE` only for non-LLM secrets the agent needs (e.g., database credentials, external API tokens).

### C4. Revert Imports

If imports were rewritten in C2, ask the user whether to revert. If no, the motus imports are fine to keep (identical behavior + tracing).

### C5. Report

- Extract build ID from output
- Report: project ID, build ID, import path
- Show expected agent URL: `https://{project-id}.agent.lithosai.cloud`
- Suggest: `motus serve health https://{project-id}.agent.lithosai.cloud`

If deploy fails, see [DEPLOY-REFERENCE.md](DEPLOY-REFERENCE.md) for troubleshooting. For post-deploy interaction, testing, and debugging, see [POST-DEPLOY.md](POST-DEPLOY.md).

---

## Cloud Agent REST API

The deployed agent exposes the same serve REST API via the agent router:

```
POST   /sessions                          — create session
GET    /sessions                          — list sessions
GET    /sessions/{id}                     — get session (add ?wait=true for long-poll)
DELETE /sessions/{id}                     — delete session
POST   /sessions/{id}/messages            — send message (returns 202, async)
GET    /sessions/{id}/messages            — get message history
```

Authentication: `Authorization: Bearer <api-key>` (from `motus login` credentials or `LITHOSAI_API_KEY` env var)

## SDK Import Mapping

The motus wrapper (`motus.openai_agents`) is a transparent drop-in replacement:

- Re-exports all symbols from the original SDK via `from <sdk> import *`
- Wraps key entry points (`Runner`) to inject tracing
- Traces are automatically uploaded to the cloud for viewing in the LITHOSAI console
- No behavior change — identical API surface, just with observability added

---

## Serve Contract & Framework Support

`motus serve start <module>:<name>` accepts **three kinds of targets** (checked in this order):

1. **`ServableAgent` instance** — any object with a `run_turn(message, state)` method. The serve runtime calls `run_turn` directly. **No wrapper function needed.**
2. **OpenAI Agents SDK `Agent` instance** — detected automatically and wrapped by the runtime. **No wrapper function needed.**
3. **Bare callable** — a function matching `(ChatMessage, list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]`.

### Built-in framework support (preferred)

Several Motus framework integrations implement the `ServableAgent` protocol. When using these, **point your import path directly at the agent instance** — do not write a manual wrapper function.

| Framework | Class | Has `run_turn`? | Deploy target |
|-----------|-------|----------------|---------------|
| **Motus ReActAgent** | `motus.agent.ReActAgent` | Yes (inherited from `AgentBase`) | `module:my_react_agent` |
| **Google ADK** | `motus.google_adk.agents.llm_agent.Agent` | Yes | `module:my_adk_agent` |
| **Anthropic SDK** | `motus.anthropic.ToolRunner` | Yes | `module:my_tool_runner` |
| **OpenAI Agents SDK** | `motus.openai_agents.Agent` | No — but auto-adapted by the serve runtime | `module:my_oai_agent` |

**Example — ReActAgent (no wrapper needed):**

```python
# agent.py — deploy with: motus deploy --name myapp agent:my_agent
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.tools import tool

@tool
async def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

client = OpenAIChatClient()  # Cloud proxy auto-provides API keys

my_agent = ReActAgent(
    client=client,
    model_name="anthropic/claude-sonnet-4.5",
    system_prompt="You are a research assistant.",
    tools=[search],
    max_steps=10,
)
# Deploy target: agent:my_agent — run_turn is inherited from AgentBase
```

**Example — Google ADK (no wrapper needed):**

```python
# agent.py — deploy with: motus deploy --name myapp agent:my_agent
from motus.google_adk.agents.llm_agent import Agent

def get_time(city: str) -> dict:
    """Return current time for a city."""
    return {"city": city, "time": "10:30 AM"}

my_agent = Agent(
    model="gemini-2.0-flash",
    name="time_agent",
    description="Returns the time in a city.",
    instruction="Call get_time when asked about time.",
    tools=[get_time],
)
# Deploy target: agent:my_agent — run_turn is built into the Agent class
```

**Example — Anthropic ToolRunner (no wrapper needed):**

```python
# agent.py — deploy with: motus deploy --name myapp agent:my_runner
from motus.anthropic import ToolRunner
from motus.tools import tool

@tool
async def lookup(query: str) -> str:
    """Look up information."""
    return f"Info about {query}"

my_runner = ToolRunner(
    model="claude-sonnet-4-20250514",
    tools=[lookup],
    system_prompt="You are a helpful assistant.",
)
# Deploy target: agent:my_runner — run_turn is built into ToolRunner
```

**Example — OpenAI Agents SDK (no wrapper needed):**

```python
# agent.py — deploy with: motus deploy --name myapp agent:my_agent
from motus.openai_agents import Agent, function_tool

@function_tool
def get_status() -> str:
    """Return system status."""
    return "All systems operational"

my_agent = Agent(
    name="assistant",
    model="gpt-4.1",
    instructions="You are a concise assistant.",
    tools=[get_status],
)
# Deploy target: agent:my_agent — the serve runtime auto-wraps OAI Agent instances
```

### Manual wrapper (fallback)

Only write a manual wrapper function when:
- Using a framework that Motus doesn't auto-support (e.g. LangGraph, CrewAI)
- You need custom state management or pre/post-processing logic

The function **must** conform to this signature:

```python
from motus.models import ChatMessage

async def my_agent(
    message: ChatMessage,
    state: list[ChatMessage],
) -> tuple[ChatMessage, list[ChatMessage]]:
    # your logic here
    response = ChatMessage(role="assistant", content="...")
    return response, state + [message, response]
```

`def` (non-async) is also accepted.

### Validation checklist (for manual wrappers)

| # | Check | Fail example |
|---|-------|-------------|
| 1 | **Module-level function** — not a method, nested function, or class | `class Agent: def run(self, ...)` |
| 2 | **Exactly 2 parameters** — `(message, state)` | `def agent(query):` |
| 3 | **`message` is a single `ChatMessage`** — not `str`, `dict`, or `list` | `def agent(message: str, state)` |
| 4 | **`state` is `list[ChatMessage]`** — the conversation history | `def agent(message, state: dict)` |
| 5 | **Returns `tuple[ChatMessage, list[ChatMessage]]`** — (response, updated_state) | `return response` (missing state) |
| 6 | **Response has `role="assistant"`** | `ChatMessage(role="user", ...)` |
| 7 | **Updated state includes both the incoming message and the response** | `return response, state` (forgot to append) |

### Common manual wrapper patterns

**Pattern A: Function that takes a string and returns a string**

```python
def my_agent(query: str) -> str:
    return call_llm(query)

# Wrapper
from motus.models import ChatMessage

def my_agent_motus(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]:
    result = my_agent(message.content)
    response = ChatMessage(role="assistant", content=result)
    return response, state + [message, response]
```

**Pattern B: Class-based agent (e.g. LangGraph, CrewAI)**

```python
from motus.models import ChatMessage

_agent = MyAgent()  # instantiate once at module level

def my_agent(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]:
    result = _agent.run(message.content)
    response = ChatMessage(role="assistant", content=result)
    return response, state + [message, response]
```

### How to fix non-conforming code

When you detect a non-conforming function:

1. Explain which check(s) failed
2. Show the user a wrapper function that adapts their code to the contract (use the patterns above)
3. Offer two options via `AskUserQuestion`:

```json
{
  "question": "Your function `agent:run` takes (query: str) instead of (message: ChatMessage, state: list[ChatMessage]). I can add a wrapper. Where should I put it?",
  "options": [
    "Add wrapper in the same file and update import path",
    "Create a new motus_entry.py with the wrapper",
    "I'll fix it myself"
  ]
}
```

4. If the user agrees, write the wrapper and update the import path accordingly (e.g. `agent:run` → `agent:run_motus` or `motus_entry:my_agent`)

---

## When something doesn't work

If the user hits a bug, needs a feature that Motus does not currently support, or encounters any limitation that blocks their use case, **offer to file a GitHub issue on their behalf**. Draft the issue with a clear title, reproduction steps, and expected behavior; show it to the user, requesting approval; and submit it, pending that approval. Use the following template for the issue body:

```bash
gh issue create --repo lithos-ai/motus \
  --title "Bug: <concise description>" \
  --body "$(cat <<'EOF'
## Description
<what happened>

## Steps to reproduce
<minimal code or commands>

## Expected behavior
<what should happen>

## Environment
- Python: <version>
- Motus: <version>
- OS: <os>
- Context: build / local deploy / cloud deploy
EOF
)"
```

Do not bother the user to file it themselves. Write the issue, show them the draft for approval, and submit it.

---

# Updates

If any `motus` command prints a message matching `motus update available: ... Run: uv tool upgrade lithosai-motus`, run the upgrade command before continuing. This updates both the CLI and the plugin files for Codex, Cursor, and Gemini.
