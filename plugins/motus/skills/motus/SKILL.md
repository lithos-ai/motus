---
name: motus
description: Build, configure, and deploy AI agents using the Motus framework. Use when user wants to create agents, define tools, set up workflows, configure memory or guardrails, or deploy agents locally or to the cloud. Triggers on mentions of motus, ReActAgent, agent_task, tool creation, MCP integration, motus deploy, motus serve.
argument-hint: "[deploy [import-path]] or [deploy [--project-id project-id] [import-path]] or [description of agent to build]"
---

# Motus

You are an expert in the Motus AI agent framework. You help users build and deploy agent applications.

## Command routing

Parse the user's arguments to determine the mode:

- **First argument is `deploy`** → go to [Deploy](#deploy), pass remaining arguments
- **Anything else** (no args, or a description of what to build) → go to [Build](#build)

Examples:
- `/motus` → Build (interactive)
- `/motus I need a customer support agent` → Build
- `/motus deploy` → Deploy (auto-detect)
- `/motus deploy myapp:my_agent` → Deploy with import path
- `/motus deploy --project-id my-project myapp:my_agent` → Deploy to cloud

---

# Build

Your job is to understand the user's requirements and help them build a fully functional agent application using Motus.

## Before writing any code

Gather requirements, but **do not bombard the user with a list of questions**. Instead:

- Infer as much as possible from what the user already said and from the project context (existing code, dependencies, env vars).
- Use a single concise multi-choice question to confirm the remaining unknowns. Prefer selection-based prompts (e.g., "Which provider? 1) OpenAI 2) Anthropic 3) Gemini 4) OpenRouter") over open-ended questions.
- Only ask follow-ups for things you truly cannot infer.

You need to understand:

1. **Agent type** — ReAct agent, task-graph workflow, or both?
2. **LLM provider & model** — OpenAI, Anthropic, Gemini, or OpenRouter?
3. **Tools needed** — Custom functions, web search, sandbox, MCP servers, or nested agents?
4. **Memory strategy** — Basic (short) or compaction (long conversations)?
5. **Safety requirements** — Any guardrails needed?
6. **Observability needs** — Tracing, hooks?

When the agent is ready to **deploy**, direct the user to run `/motus deploy`.

**Always prefer `uv`** for package management and running commands (`uv sync`, `uv pip install`, `uv run`). Only fall back to `pip`/`python` if the user explicitly asks or `uv` is unavailable.

## Environment check

Before writing any agent code, verify the user's environment is ready. Run these checks silently and only report problems:

1. **Python version** — Must be 3.12+. Check with `python3 --version`.
2. **Motus installed** — Try `python3 -c "import motus"`. If it fails, install as a dependency:
   ```bash
   uv add git+ssh://git@github.com/lithos-ai/motus.git
   ```
3. **API keys** — For local development, check if at least one LLM provider key is set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.). If none are found, ask which provider they plan to use and help them set the key. Note: API keys are **not needed for cloud deployment** — the platform's model proxy handles authentication automatically.
4. **Optional: Docker** — Only check if the user needs sandbox or MCP-in-container features. `docker info` should succeed.

If everything is fine, proceed without mentioning the checks.

## Quick reference

See these files for detailed API reference and patterns:

- [REFERENCE.md](REFERENCE.md) — Complete API signatures, constructors, all parameters
- [PATTERNS.md](PATTERNS.md) — Proven code patterns for common scenarios
- [EXAMPLES.md](EXAMPLES.md) — End-to-end example applications

## Core concepts (brief)

| Concept | What it is | Key import |
|---------|-----------|------------|
| `ReActAgent` | Autonomous agent with reasoning + tool-use loop | `from motus.agent import ReActAgent` |
| `@agent_task` | Decorator turning functions into dependency-tracked async tasks | `from motus.runtime import agent_task` |
| `@tool` | Decorator wrapping functions as LLM-callable tools | `from motus.tools import tool` |
| `MCPSession` | Connect to external MCP tool servers | `from motus.tools import get_mcp` |
| `Sandbox` | Docker container for code execution | `from motus.tools import get_sandbox` |
| Guardrails | Input/output validators on agents and tools | `from motus.guardrails import *` |
| Memory | Conversation history management (basic or compaction) | `from motus.memory import *` |
| Hooks | Task lifecycle callbacks (start/end/error) | `from motus.runtime.hooks import register_hook` |

## Workflow: building an agent application

### Step 1: Set up the client

```python
# Pick one provider
from motus.models import OpenAIChatClient      # or
from motus.models import AnthropicChatClient    # or
from motus.models import GeminiChatClient       # or
from motus.models import OpenRouterChatClient

client = OpenAIChatClient(api_key="sk-...")
```

### Step 2: Define tools

```python
from motus.tools import tool

@tool
async def my_tool(param: str) -> str:
    """Description the LLM sees."""
    return result
```

Both `async def` and regular `def` are supported. For complex inputs, use `InputSchema` (Pydantic). For class-based tools, use `@tools`. For external servers, use `get_mcp()`. For code execution, use `get_sandbox()`. See [PATTERNS.md](PATTERNS.md) for all patterns.

### Step 3: Create the agent

```python
from motus.agent import ReActAgent

agent = ReActAgent(
    client=client,
    model_name="gpt-4o",
    system_prompt="You are ...",
    tools=[my_tool],
    max_steps=10,
)
```

### Step 4: Add guardrails (if needed)

```python
agent = ReActAgent(
    ...,
    input_guardrails=[validate_input],
    output_guardrails=[redact_pii],
)
```

### Step 5: Run

```python
response = await agent("user message")
```

## Critical rules

- **Always pass `client` as the first argument to `ReActAgent`** — Not a model name string.
- **Use `await` for agent calls in async context** — `response = await agent("prompt")`.
- **Guardrail functions declare only the parameters they inspect** — They don't need to match the full tool signature.
- **MCP sessions are lazy by default** — They connect on first tool call when passed to an agent.

---

# Deploy

Usage:
- `/motus deploy` — auto-detect agent from files, ask user to confirm
- `/motus deploy myapp:my_agent` — will ask local or cloud
- `/motus deploy --project-id my-project myapp:my_agent` — `--project-id` flag implies cloud deploy

For deploy troubleshooting, see [DEPLOY-REFERENCE.md](DEPLOY-REFERENCE.md). For post-deploy interaction, see [POST-DEPLOY.md](POST-DEPLOY.md).

## 0. Detect & Confirm

**No arguments provided:**

Scan `.py` files in the current directory for likely agent entry points. Look for:
1. Functions matching the serve contract: `def xxx(message, state)` or `async def xxx(message, state)` (see [Agent Function Contract](#agent-function-contract))
2. Files named `agent.py`, `app.py`, `server.py`, `main.py`
3. Variables/functions that look like agent callables — if found but don't match the contract, note them as candidates that need a wrapper

If candidates are found, you MUST call the `AskUserQuestion` tool (not just print text) so the user gets clickable options. Example tool call:

```json
{"question": "I found these agent entry points. Which one to deploy?", "options": ["agent:my_agent", "app:serve", "Let me specify manually"]}
```

If no candidates found, you MUST call `AskUserQuestion` tool with just a question (no options) to ask for the import path.

**One argument provided (import-path):**

You MUST call the `AskUserQuestion` tool to ask deploy target. Do NOT just print the options as text — the user needs clickable buttons. Example:

```json
{"question": "Where would you like to deploy agent:my_agent?", "options": ["Local — start serve on this machine", "Cloud — deploy to LITHOSAI"]}
```

**`--project-id` flag provided (project-id + import-path):**

Assume cloud deploy. No need to ask.

---

## Local Deploy

### L1. Validate

- Verify import path contains `:` (format: `module:callable`)
- Verify the referenced module file exists in the current directory
- **Validate the agent function signature** — see [Agent Function Contract](#agent-function-contract) below. If the function does not conform, help the user fix it before proceeding.

### L2. Start Server

```bash
motus serve start $IMPORT_PATH --port 8000
```

Optional flags to offer the user:
- `--port <N>` — bind port (default 8000)
- `--workers <N>` — worker processes (default CPU count)
- `--ttl <seconds>` — idle session TTL
- `--timeout <seconds>` — max seconds per agent turn
- `--log-level debug` — verbose logging

### L3. Test

Once the server is running, suggest testing in another terminal:

```bash
motus serve chat http://localhost:8000 "hello"   # single message
motus serve chat http://localhost:8000            # interactive REPL
motus serve health http://localhost:8000          # health check
```

For post-deploy interaction details, see [POST-DEPLOY.md](POST-DEPLOY.md).

---

## Cloud Deploy

### C1. Validate

- Verify project-id and import-path are provided
- Verify import path contains `:` (format: `module:callable`)
- Verify the referenced module file exists in the current directory
- **Validate the agent function signature** — see [Agent Function Contract](#agent-function-contract) below. If the function does not conform, help the user fix it before proceeding.
- Verify env vars `LITHOSAI_API_URL` and `LITHOSAI_API_KEY` are set; if missing, tell user and stop

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

### C1.6. Local Smoke Test Before Cloud Deploy

After rewriting the code for cloud (C1.5) and before uploading, run the agent locally with `motus serve start` to catch errors early. Cloud builds take 1-2 minutes per iteration — a local test takes seconds.

```bash
motus serve start $IMPORT_PATH --port 8000
```

Then in another terminal:

```bash
motus serve chat http://localhost:8000 "hello"
```

If the agent responds correctly, proceed to cloud deploy. If it fails, fix the issue locally first. Common failures at this stage:
- Import errors (missing dependencies, typos)
- LLM client connection errors (missing API key env var)
- Handler signature mismatches

> **Note:** The local test uses the same serve runtime as the cloud, so most issues caught here will also be issues in cloud.

### C1.7. Check Project Dependencies (Third-Party Packages)

The cloud build installs the base motus package (**no extras**), then installs project deps from `requirements.txt`, `pyproject.toml`, `uv.lock`, or `pylock.toml`. If none of these exist, only base motus is available.

The motus SDK integrations are **optional extras** that pull in separate packages:

| Import found | Underlying package needed | Add to requirements.txt |
|---|---|---|
| `from motus.claude_agent import ...` or `from claude_agent_sdk import ...` | `claude-agent-sdk` (motus[claude-agent] extra) | `claude-agent-sdk` |
| `from motus.openai_agents import ...` or `from agents import ...` | `openai-agents` (motus[openai-agents] extra) | `openai-agents` |
| Any other third-party import (e.g. `requests`, `pandas`) | that package | the package name |

> **Note**: Do NOT write `motus[claude-agent]` or `motus[openai-agents]` in requirements.txt — motus is already installed by the cloud build; you just need the underlying packages (`claude-agent-sdk` or `openai-agents`).

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
| `from claude_agent_sdk import ...` | `from motus.claude_agent import ...` |
| `import claude_agent_sdk` | `import motus.claude_agent as claude_agent_sdk` |
| `from agents import ...` | `from motus.openai_agents import ...` |
| `import agents` | `import motus.openai_agents as agents` |

If found: list the files/lines, explain the change is temporary for deploy, wait for user confirmation, then make replacements and record originals for revert.

### C3. Deploy

```bash
motus deploy --name $PROJECT_NAME $IMPORT_PATH
```

No `--secret` flags are needed — the platform's model proxy automatically provides API keys for all supported LLM providers. Add `--secret KEY=VALUE` only for non-LLM secrets the agent needs (e.g., database credentials, external API tokens).

### C4. Revert Imports

If imports were rewritten in C2, ask the user whether to revert. If no, the motus imports are fine to keep (identical behavior + tracing).

### C5. Report

- Extract build ID from output
- Report: project ID, build ID, import path
- Show expected agent URL: `https://{project-id}.agent.{subdomain}`
- Suggest: `motus serve health https://{project-id}.agent.{subdomain}`

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

Authentication: `Authorization: Bearer $LITHOSAI_API_KEY`

## SDK Import Mapping

The motus wrappers (`motus.claude_agent`, `motus.openai_agents`) are transparent drop-in replacements:

- Re-export all symbols from the original SDK via `from <sdk> import *`
- Wrap key entry points (`query()`, `ClaudeSDKClient`, `Runner`) to inject tracing
- Traces are automatically uploaded to the cloud for viewing in the LITHOSAI console
- No behavior change — identical API surface, just with observability added

---

## Agent Function Contract

The Motus platform runs user code via `motus serve start <module>:<callable>`. The callable **must** conform to this exact signature:

```python
from motus.models import ChatMessage

def my_agent(
    message: ChatMessage,
    state: list[ChatMessage],
) -> tuple[ChatMessage, list[ChatMessage]]:
    # message: the incoming user message
    # state: full conversation history up to this point
    response = ChatMessage(role="assistant", content="...")
    return response, state + [message, response]
```

`async def` is also accepted.

### Validation checklist

Read the target function and verify **all** of the following. If any check fails, explain the issue and offer to fix it.

| # | Check | Fail example |
|---|-------|-------------|
| 1 | **Module-level function** — not a method, nested function, or class | `class Agent: def run(self, ...)` |
| 2 | **Exactly 2 parameters** — `(message, state)` | `def agent(query):`, `def agent(message, state, config):` |
| 3 | **`message` is a single `ChatMessage`** — not `str`, `dict`, or `list` | `def agent(message: str, state)` |
| 4 | **`state` is `list[ChatMessage]`** — the conversation history | `def agent(message, state: dict)` |
| 5 | **Returns `tuple[ChatMessage, list[ChatMessage]]`** — (response, updated_state) | `return response` (missing state), `return {"text": "..."}` |
| 6 | **Response has `role="assistant"`** | `ChatMessage(role="user", ...)` |
| 7 | **Updated state includes both the incoming message and the response** | `return response, state` (forgot to append) |

Type annotations are optional (the runtime doesn't enforce them), but the actual values must conform.

### Common patterns that need wrapping

Users often write agent code that doesn't match this contract. Here's how to wrap common patterns:

**Pattern A: Function that takes a string and returns a string**

```python
# User's original code
def my_agent(query: str) -> str:
    return call_llm(query)

# Wrapped for motus
from motus.models import ChatMessage

def my_agent_motus(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]:
    result = my_agent(message.content)
    response = ChatMessage(role="assistant", content=result)
    return response, state + [message, response]
```

**Pattern B: Class-based agent (e.g. LangGraph, CrewAI)**

```python
# User's original code
class MyAgent:
    def __init__(self):
        self.graph = build_graph()
    def run(self, input: str) -> str:
        return self.graph.invoke(input)

# Wrapped for motus — instantiate once at module level
from motus.models import ChatMessage

_agent = MyAgent()

def my_agent(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]:
    result = _agent.run(message.content)
    response = ChatMessage(role="assistant", content=result)
    return response, state + [message, response]
```

**Pattern C: Agent that needs full conversation history (not just latest message)**

```python
from motus.models import ChatMessage

def my_agent(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]:
    # Convert state to the format your LLM expects
    history = [{"role": m.role, "content": m.content} for m in state]
    history.append({"role": message.role, "content": message.content})
    result = call_llm(messages=history)
    response = ChatMessage(role="assistant", content=result)
    return response, state + [message, response]
```

**Pattern D: Motus ReActAgent for cloud deploy**

```python
# Cloud deploy pattern (confirmed working)
# Deploy with: motus deploy --name myapp agent_module:agent
# No API keys or base URLs needed — the platform auto-wires them.
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient, ChatMessage

client = OpenAIChatClient()  # Picks up OPENAI_API_KEY and OPENAI_BASE_URL from env

async def agent(message, state):
    react_agent = ReActAgent(
        client=client,
        model_name="anthropic/claude-sonnet-4.5",
        tools=[...],
    )
    for msg in state:
        await react_agent.add_message(msg)
    result = await react_agent(message.content)
    response = ChatMessage.assistant_message(content=result)
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
