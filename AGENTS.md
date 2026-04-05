# Motus — Agent Guide

Motus is a full-stack AI agent framework built on a custom async task-graph runtime. It provides agent definitions (ReAct loop), multi-provider LLM clients, a composable tool system (function tools, MCP, Docker sandboxes), persistent memory, a skills system, and FastAPI-based serving infrastructure.

## Commands

```bash
# Install all dependencies (core + dev)
uv sync --all-extras --dev

# Run unit tests (fast, no API keys needed)
uv run pytest -v tests/unit -m "not slow"

# Run integration tests with VCR replay (no API keys needed)
uv run pytest -v tests/integration -m "integration"

# Re-record VCR cassettes (requires real API keys in .env)
uv run pytest tests/integration/examples/ -v --vcr-record=all

# Format code
uv run ruff format

# Lint and auto-fix
uv run ruff check --fix

# Run all pre-commit hooks on all files
uv run pre-commit run --all-files

# Install git hooks (first-time setup)
uv run pre-commit install

# Start the agent server
uv run motus-serve start myapp:server --port 8000

# Interactive chat with an agent
uv run motus-serve chat
```

## Tech Stack

- **Python** 3.12+ (uses modern syntax: `X | None`, `dict[K, V]`, `TYPE_CHECKING`)
- **Async runtime**: asyncio + `concurrent.futures` for cross-thread coordination
- **Validation**: pydantic >= 2.12.5
- **Web**: fastapi >= 0.128.0, uvicorn >= 0.40.0, httpx >= 0.28.1
- **LLM clients**: anthropic >= 0.40.0, openai >= 2.15.0, google-genai >= 1.0.0
- **Tool execution**: docker >= 7.1.0, kubernetes >= 34.1.0, mcp[cli] >= 1.25.0, paramiko >= 4.0.0
- **Testing**: pytest >= 9.0.2, pytest-asyncio >= 0.24.0, vcrpy >= 8.1.1
- **Linting**: ruff (rules: E, F, I, W; ignores: E501, E402)
- **Package manager**: uv (workspace with members `.` and `services/*`)

## Project Structure

```
src/motus/
├── __init__.py              # LithOS, ModelClient — high-level API
├── runtime/                 # Core task-graph engine (owner: @NorthmanPKU)
│   ├── agent_runtime.py     #   GraphScheduler (event loop, dependency DAG, retries/timeouts)
│   │                        #   AgentRuntime (thread-safe wrapper, cross-thread task submission)
│   ├── agent_task.py        #   @agent_task decorator, AgentTaskDefinition (descriptor protocol)
│   ├── agent_future.py      #   AgentFuture — lazy future with operator overloading & sync barriers
│   ├── task_instance.py     #   TaskInstance (COMPUTE/RESOLVE), TaskPolicy, TaskStatus, stack stitching
│   ├── hooks.py             #   HookManager — task_start / task_end / task_error lifecycle events
│   ├── types.py             #   AgentTaskId, AgentFutureId counters
│   └── tracing/             #   Distributed tracing, OpenTelemetry export, live trace server
├── agent/                   # Agent definitions (owners: @NorthmanPKU, @JackFram)
│   ├── base_agent.py        #   AgentBase[T] — abstract base with tools, memory
│   ├── react_agent.py       #   ReActAgent — Reasoning + Acting loop
│   ├── skills.py            #   Skill loading (SKILL.md) and tool creation
│   └── tasks.py             #   model_serve_task — agent task for model calls
├── tools/                   # Tool ecosystem (owners: @eliotsolomon18, @coppock, @NorthmanPKU)
│   ├── core/
│   │   ├── tool.py          #   Tool protocol (description, json_schema, __call__)
│   │   ├── function_tool.py #   FunctionTool — wraps Python functions as LLM tools
│   │   ├── mcp_tool.py      #   MCP (Model Context Protocol) tool integration
│   │   ├── sandbox.py       #   Sandbox ABC (create/acreate/connect factory methods)
│   │   ├── tool_provider.py #   ToolProvider interface
│   │   ├── composite_tool_provider.py
│   │   ├── normalize.py     #   Schema generation and tool normalization
│   │   └── decorators.py    #   @tool and @tools convenience decorators
│   ├── providers/
│   │   ├── docker/          #   DockerSandbox — container-based code execution
│   │   └── brave/           #   Brave Search tool
│   └── runtime/
│       └── sync_manager.py  #   Sync/async coordination for tool calls
├── models/                  # Multi-provider LLM clients (owner: @yzhou442)
│   ├── base.py              #   BaseChatClient, ChatMessage, ToolCall, ChatCompletion
│   ├── anthropic_client.py  #   Claude API
│   ├── openai_client.py     #   OpenAI API
│   ├── gemini_client.py     #   Google Gemini API
│   ├── openrouter_client.py #   OpenRouter aggregator
│   └── message_schema.py    #   Unified message data models
├── memory/                  # Memory & context management (owners: @JackFram, @vasiliskyp)
│   ├── memory.py            #   Main Memory class
│   ├── interfaces.py        #   MemoryStore, VectorStore ABCs
│   ├── models.py            #   MemoryEntry, MemoryScope, MemoryType, MemoryPriority
│   ├── embedding/           #   Vector embedding support
│   ├── stores/              #   FileSystem and InMemory stores
│   └── summarization/       #   Message summarization utilities
├── serve/                   # Agent serving infrastructure
│   ├── server.py            #   AgentServer (FastAPI)
│   ├── cli.py               #   motus-serve CLI (start/chat/submit/status/run)
│   ├── worker/              #   Worker pool (executor, context, pool)
│   └── analytics/           #   Analytics collector and web dashboard
└── utils/                   # Shared utilities (owner: @coppock)
```

```
tests/
├── unit/                    # Fast tests, no external deps
│   ├── runtime/             #   agent_future, hooks, lifecycle, task_policy
│   ├── agent/               #   agent tests
│   ├── tools/               #   function_tool, sandbox, browser, web_search
│   ├── memory/              #   memory system
│   └── serve/               #   chat, server
├── integration/             # Uses VCR cassettes for HTTP replay
│   └── examples/            #   test_omni_integration.py + conftest.py
examples/                    # Runnable demos
├── runtime/                 #   task_graph_demo, resilient_tasks, hooks_demo, stacktrace_demo
├── deep_research/           #   Research agent workflow
├── code_agent/              #   Code generation agent
├── omni/                    #   Full-stack multi-tool demo
└── motusbot/                #   Multi-channel bot
```

## Critical Runtime Constraints

These are hard-won lessons. Violating them causes deadlocks or silent data corruption.

### 1. NEVER use `set()` with AgentFuture

`AgentFuture.__eq__` and `__hash__` are **sync barriers** — they block the calling thread to resolve the future's value. Putting futures in a `set` or using them as `dict` keys triggers these barriers and deadlocks the runtime.

```python
# WRONG — deadlocks the runtime event loop
deps = {future1, future2, future3}

# CORRECT — use dict keyed by id()
deps: dict[int, AgentFuture] = {id(f): f for f in futures}
```

### 2. Deadlock detection for sync barriers

`AgentFuture._wait_for_result()` detects when blocking happens on the runtime's own event loop and raises `RuntimeError`. In async code, always `await future` instead of calling `.af_result()` or `resolve()`.

## Testing

### Test markers

| Marker | Purpose | Command |
|--------|---------|---------|
| `unit` | Fast, no external deps | `uv run pytest tests/unit -m "not slow"` |
| `integration` | VCR cassette replay | `uv run pytest tests/integration -m "integration"` |
| `slow` | Real API calls, heavy ops | `uv run pytest -m "slow"` (needs API keys) |

### Async test configuration

`asyncio_mode = "auto"` in pyproject.toml — all `async def test_*` functions run automatically without `@pytest.mark.asyncio`. Use `unittest.IsolatedAsyncioTestCase` for class-based async tests.

### VCR (HTTP replay) system

Integration tests record HTTP interactions to YAML cassettes for deterministic offline replay:

- **Cassettes location**: `tests/integration/examples/cassettes_vcrpy/`
- **Replay mode** (CI, default): Just run pytest — no API keys needed, uses fake keys injected by `_fake_api_keys` fixture
- **Record mode**: `uv run pytest tests/integration/examples/ -v --vcr-record=all` — requires real API keys in `.env`
- **Scrubbing**: Large base64 blobs, API keys, and OpenAI reasoning fields are automatically stripped from cassettes
- **Body matching**: Custom JSON matcher normalizes whitespace for stable comparison

### Key fixtures (`tests/integration/examples/conftest.py`)

- `configured_vcr` — Module-scoped VCR instance respecting `--vcr-record` CLI flag
- `mock_sandbox` — Mock sandbox with `mock_sh()` and `mock_python()` to avoid Docker dependency
- `_fake_api_keys` — Autouse fixture that sets fake API keys during replay mode

### Writing a new test

```python
import pytest
from motus.runtime import init, shutdown

class TestMyFeature:
    def setup_method(self):
        # Ensure clean runtime state between tests
        if is_initialized():
            shutdown()

    def teardown_method(self):
        if is_initialized():
            shutdown()

    def test_basic_task(self):
        rt = init()
        @agent_task
        async def add(a, b):
            return a + b

        future = add(1, 2)
        assert resolve(future, timeout=5) == 3

    @pytest.mark.integration
    def test_with_vcr(self, configured_vcr, mock_sandbox):
        with configured_vcr.use_cassette("my_test.yaml"):
            # Test code using LLM clients — HTTP calls replayed from cassette
            ...
```

## Code Style

### Ruff rules (pyproject.toml)

- **Enabled**: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `W` (warnings)
- **Ignored**: `E501` (line length — no limit enforced), `E402` (module-level imports)
- Format: `uv run ruff format` — applies Black-compatible style automatically

### Import ordering

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

# 3. Third-party
from pydantic import BaseModel
import httpx

# 4. Local / relative imports
from ..runtime.agent_future import AgentFuture
from .react_agent import ReActAgent

# 5. TYPE_CHECKING block for circular imports
if TYPE_CHECKING:
    from .agent_future import AgentFuture
```

### Naming conventions

| Entity | Convention | Examples |
|--------|-----------|----------|
| Classes | PascalCase | `TaskPolicy`, `AgentFuture`, `GraphScheduler` |
| Functions / methods | snake_case | `register_agent_task`, `_deep_unwrap` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_POLICY`, `MAX_BLOB_SIZE` |
| Private internals | `_` prefix | `_prerequisites`, `_scan_deps` |
| Type aliases | PascalCase | `HookCallback`, `AgentTaskId` |

### Type annotations

Use modern Python 3.12 syntax throughout:

```python
# Union syntax (not Optional)
timeout: float | None = None

# Built-in generic collections (not typing.Dict)
tasks: dict[AgentTaskId, TaskInstance] = {}

# TYPE_CHECKING for forward references that cause circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent_future import AgentFuture
```

## Git Workflow

### Branch naming

- Feature: `feature/<name>` (e.g., `feature/meta-agent`)
- Bug fix: `fix/<name>` (e.g., `fix/tracer-fix`)
- Personal/dev: `<username>-<topic>` (e.g., `jjn_runtime`)

### Commit messages

Concise, present tense, action-oriented. Prefix with category when helpful:

```
feat: add memory summarization to ReActAgent
fix: resolve deadlock in sync barrier detection
refactor: mv agent/tool spec under agent module
test: add VCR cassettes for omni integration
```

### PR template

Every PR must include **Motivation**, **Modifications**, and **Best Practices Checklist**:
- [ ] Format your code with pre-commit
- [ ] Add unit tests if needed
- [ ] Update documentation if needed

### CI pipeline

On push/PR to `main`, GitHub Actions runs:
1. **Code Quality** — `pre-commit run --all-files` (ruff format + check, codespell, file checks)
2. **Build & Unit Test** — `pytest -v tests/unit -m "not slow"` + `pytest -v tests/integration -m "integration"`

Both must pass before merge.

## Environment Variables

```bash
# Logging
MOTUS_LOG_LEVEL=DEBUG|INFO|WARN|ERROR     # Runtime log verbosity
MOTUS_QUIET_SYNC=1                        # Suppress sync barrier warnings

# Tracing
MOTUS_TRACING=1                           # Enable trace collection
MOTUS_COLLECTION_LEVEL=disabled|basic|full

# Server
MOTUS_HOST=0.0.0.0
MOTUS_PORT=8000

# API keys (for examples and recording VCR cassettes)
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
BRAVE_API_KEY=...
OPENROUTER_API_KEY=...
```

## Key Architectural Patterns

### Task graph execution flow

```
@agent_task def my_func(x): return x * 2

future = my_func(5)
  └→ AgentTaskDefinition.__call__
  └→ AgentRuntime.submit_task_registration (thread-safe, cross-thread)
  └→ GraphScheduler.register_task
      ├─ Create TaskInstance (kind=COMPUTE)
      ├─ _scan_deps(): find AgentFuture args → build dependency edges
      ├─ No deps? → status=READY → _execute_task()
      └─ Has deps? → status=PENDING → wait for deps to resolve

_execute_task:
  ├─ Unwrap resolved args
  ├─ Emit task_start hook
  ├─ Run function (async natively, sync via run_in_executor)
  ├─ On success: _settle_or_defer (creates RESOLVE task if result has nested futures)
  ├─ On failure: retry (up to policy.retries) or propagate error + stitch creation chain
  └─ Emit task_end / task_error hook

await future → resolved value
```

### Tool definition patterns

```python
from motus.tools import FunctionTool, tool

# Decorator-based
@tool
async def search(query: str) -> dict:
    """Search the web."""
    return {"results": [...]}

# Class-based
tool = FunctionTool(my_async_function)

# Tools are called with JSON string args, return JSON string results
result: str = await tool('{"query": "motus framework"}')
```

### Agent task decorator

```python
from motus.runtime.agent_task import agent_task

# Basic usage — returns AgentFuture, not the direct result
@agent_task
async def fetch(url: str) -> dict:
    ...

# With policy overrides
@agent_task(retries=3, timeout=10.0)
async def resilient_fetch(url: str) -> dict:
    ...

# Multi-return — destructures into multiple futures
@agent_task(num_returns=2)
async def split(data):
    return part_a, part_b

fa, fb = split(data)

# Dependency chaining — futures as args auto-wire the DAG
result = process(fetch("http://api.example.com"))  # process waits for fetch
```

## Boundaries

- **Always do:**
  - Run `uv run pre-commit run --all-files` before committing
  - Run `uv run pytest -v tests/unit -m "not slow"` and verify tests pass
  - Use `dict[int, AgentFuture]` keyed by `id()` for AgentFuture collections
  - Follow the import ordering and naming conventions above
  - Add or update unit tests for code changes
  - Use `TYPE_CHECKING` guards for imports that cause circular dependencies
  - Respect CODEOWNERS — changes to a module should involve its owner

- **Ask first:**
  - Adding new dependencies to pyproject.toml
  - Modifying CI/CD workflows (`.github/workflows/`)
  - Changing the public API of `src/motus/__init__.py`
  - Schema changes to TaskPolicy or other core dataclasses
  - Modifying VCR cassettes or test infrastructure in conftest.py
  - Any changes to the runtime event loop or GraphScheduler scheduling logic

- **Never do:**
  - Use `set()`, `frozenset()`, or dict keys with AgentFuture objects
  - Commit API keys, secrets, or `.env` files
  - Modify `node_modules/`, `vendor/`, or generated files in `cassettes_vcrpy/` by hand
  - Force-push to `main`
  - Skip pre-commit hooks with `--no-verify`
  - Use `.af_result()` or `resolve()` on AgentFuture from within the runtime event loop (causes deadlock)
  - Remove failing tests instead of fixing them
