# motus.serve

Single-agent HTTP server with session-based conversations and process-per-request isolation.

## Overview

serve is designed around a single constraint, **one agent per server**, leaving a minimal session-based REST API backed by isolated worker processes.

Key properties:

- **One agent, one server** — register a single function, serve it over HTTP.
- **Session-based** — conversations are held in named sessions with full message history.
- **Process-per-request** — each turn spawns a fresh subprocess, providing memory isolation and true parallelism (no GIL).
- **Long-poll** — message sends return immediately; clients await results via an optional blocking query parameter (polling also supported).

## Quick Start

```python
# myapp.py
from motus.agent import ReActAgent
from motus.models import AnthropicChatClient, ChatMessage

client = AnthropicChatClient()

async def my_agent(message, state):
    agent = ReActAgent(
        client=client,
        model_name="claude-opus-4-6",
        system_prompt="You are a helpful assistant.",
    )
    for msg in state:
        await agent.add_message(msg)
    result = await agent(message.content)
    response = ChatMessage.assistant_message(content=result)
    return response, state + [message, response]
```

```bash
# Start the server
python -m motus.serve start myapp:my_agent --port 8000

# Chat interactively
python -m motus.serve chat http://localhost:8000
```

---

## Agent Function Contract

An agent function receives the new user message and the session state, and returns a response message along with the updated state:

```python
from motus.models import ChatMessage

def my_agent(message: ChatMessage, state: list[ChatMessage]) -> tuple[ChatMessage, list[ChatMessage]]:
    response = ChatMessage.assistant_message(content="hello")
    return response, state + [message, response]
```

| Parameter | Type | Description |
|---|---|---|
| `message` | `ChatMessage` | The new user message (constructed by the server from the HTTP request). |
| `state` | `list[ChatMessage]` | The session's state from the previous turn (empty list on first turn). |

**Return value**: `tuple[ChatMessage, list[ChatMessage]]` — the response message (surfaced to the HTTP client) and the updated state (stored in the session). The agent owns the state and can append messages, compact history, add system messages, or restructure freely.

Both sync and async functions are supported:

```python
# Sync
def my_agent(message, state):
    response = ChatMessage.assistant_message(content="hello")
    return response, state + [message, response]

# Async
async def my_agent(message, state):
    result = await some_api_call(message.content)
    response = ChatMessage.assistant_message(content=result)
    return response, state + [message, response]
```

Agent functions run in **worker subprocesses** and are resolved by import path. The function must be importable from the worker process (i.e., defined at module level).

---

## Python API

### `AgentServer`

```python
from motus.serve import AgentServer

server = AgentServer(my_agent, max_workers=None, ttl=0, timeout=0, max_sessions=0, allow_custom_ids=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent_fn` | `Callable \| str` | *required* | Agent function or import path string (e.g., `"myapp:my_agent"`). |
| `max_workers` | `int \| None` | `None` | Maximum concurrent worker processes. Defaults to `os.cpu_count()`, fallback `4`. Keyword-only. |
| `ttl` | `float` | `0` | TTL for idle/error sessions in seconds. `0` disables expiry. Keyword-only. |
| `timeout` | `float` | `0` | Max seconds per agent turn before the worker process is killed. `0` disables. Keyword-only. |
| `max_sessions` | `int` | `0` | Maximum number of concurrent sessions. `0` means unlimited. Keyword-only. |
| `shutdown_timeout` | `float` | `0` | Seconds to wait for in-flight tasks on shutdown before cancelling. `0` waits indefinitely. Keyword-only. |
| `allow_custom_ids` | `bool` | `False` | Enable `PUT /sessions/{session_id}` for client-specified session IDs. Keyword-only. |

#### `run(host, port, log_level) -> None`

Start the server (blocking).

| Parameter | Type | Default |
|---|---|---|
| `host` | `str` | `"0.0.0.0"` |
| `port` | `int` | `8000` |
| `log_level` | `str` | `"info"` |

#### `app -> FastAPI`

The underlying FastAPI application instance (created lazily on first access). Useful for testing or mounting in a larger application.

---

## REST API

### Health

#### `GET /health`

Server health check.

**Response** `200 OK` — `HealthResponse`:

```json
{ "status": "ok", "max_workers": 4, "running_workers": 2, "total_sessions": 2 }
```

---

### Sessions

#### `POST /sessions`

Create a new conversation session. Returns a `Location` header with the session URL.

**Request** (optional) — `CreateSessionRequest`:

```json
{ "state": [{ "role": "user", "content": "hello" }, { "role": "assistant", "content": "hi" }] }
```

When provided, `state` preloads the session with existing conversation history. Omit the body (or send `{}`) to start with an empty session.

**Response** `201 Created` — `SessionResponse`:

**Headers**: `Location: /sessions/{session_id}`

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "idle",
  "response": null,
  "error": null
}
```

**Errors**: `503` — maximum number of sessions reached (when `max_sessions` is configured).

#### `PUT /sessions/{session_id}`

Create a session with a client-specified ID. Requires `--allow-custom-ids` (CLI) or `allow_custom_ids=True` (Python API).

**Request** (optional) — `CreateSessionRequest`: same as `POST /sessions`.

**Response** `201 Created` — `SessionResponse`:

**Headers**: `Location: /sessions/{session_id}`

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "idle",
  "response": null,
  "error": null
}
```

**Errors**:

| Code | Condition |
|---|---|
| `400` | Session ID is not a valid UUID |
| `405` | Custom session IDs are not enabled |
| `409` | Session already exists |
| `503` | Maximum number of sessions reached |

#### `GET /sessions`

List all sessions.

**Response** `200 OK` — `list[SessionSummary]`:

```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "total_messages": 4,
    "status": "idle"
  }
]
```

#### `GET /sessions/{session_id}`

Get session details and the latest response. Supports optional long-polling via query parameters.

**Query parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `wait` | `bool` | `false` | When `true`, block until the session is no longer `"running"`. |
| `timeout` | `float` | `None` | Maximum seconds to wait (only meaningful when `wait=true`). If exceeded, returns the current state with `status: "running"`. Omit for unlimited wait. |

**Response** `200 OK` — `SessionResponse`:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "idle",
  "response": { "role": "assistant", "content": "hi there" },
  "error": null
}
```

While `status` is `"running"`, both `response` and `error` are `null`. They are only populated once the turn completes (as `"idle"` or `"error"`).

**Errors**: `404` — session not found (or deleted while waiting).

#### `GET /sessions/{session_id}/messages`

Get the full conversation history managed by the agent.

**Response** `200 OK` — `list[ChatMessage]`:

```json
[
  { "role": "user", "content": "hello" },
  { "role": "assistant", "content": "hi there" }
]
```

**Errors**: `404` — session not found.

#### `DELETE /sessions/{session_id}`

Delete a session. Safe to call while a turn is running — the running task is cancelled and the worker process is killed.

**Response** `204 No Content`.

**Errors**: `404` — session not found.

---

### Messages

#### `POST /sessions/{session_id}/messages`

Send a message. Returns immediately — the agent runs in the background.

**Request** — `MessageRequest`:

```json
{ "content": "hello" }
```

An optional `webhook` field can be included to receive the turn result via HTTP callback:

```json
{ "content": "hello", "webhook": { "url": "https://example.com/hook", "token": "secret", "include_messages": false } }
```

When `webhook` is provided, the server POSTs a `WebhookPayload` to the given URL after the turn completes. See [Webhooks](#webhooks) for details.

**Response** `202 Accepted` — `MessageResponse`:

**Headers**: `Location: /sessions/{session_id}`

```json
{ "session_id": "550e8400-e29b-41d4-a716-446655440000", "status": "running" }
```

**Errors**:

| Code | Condition |
|---|---|
| `404` | Session not found |
| `409` | Session is already processing a message |

### Long-Poll

Use `GET /sessions/{id}?wait=true` to block until the current turn completes:

```
POST /sessions/{id}/messages                    → 202  (status: "running")
GET  /sessions/{id}?wait=true                   → 200  SessionResponse
```

With a timeout:

```
GET  /sessions/{id}?wait=true&timeout=30        → 200  SessionResponse
```

| Scenario | Status code | `status` field |
|---|---|---|
| Agent finished successfully | `200` | `"idle"` |
| Agent raised an exception | `200` | `"error"` |
| Timeout exceeded | `200` | `"running"` |
| Session not found or deleted | `404` | — |

### Polling (fallback)

Clients can also poll `GET /sessions/{session_id}` (without `wait`) until `status` is no longer `"running"`:

```
POST /sessions/{id}/messages             → 202  (status: "running")
GET  /sessions/{id}                      → 200  (status: "running", response: null)
GET  /sessions/{id}                      → 200  (status: "idle", response: {...})
```

While `status` is `"running"`, `response` and `error` are `null`. On success, the `response` field contains the agent's reply. The agent's updated state is stored internally and can be retrieved via `GET /sessions/{id}/messages`. On failure, `status` becomes `"error"` and the `error` field contains the exception message.

### Webhooks

As an alternative to polling or long-poll, you can request a webhook callback when sending a message. Include a `webhook` field in the `MessageRequest`:

```json
{ "content": "hello", "webhook": { "url": "https://example.com/hook", "token": "secret", "include_messages": false } }
```

#### `WebhookSpec` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | *required* | The URL to POST the result to. |
| `token` | `str \| None` | `None` | Bearer token sent in the `Authorization` header. |
| `include_messages` | `bool` | `false` | When `true`, the full conversation history is included in the payload. |

#### `WebhookPayload`

After the turn completes (success, error, or cancellation), the server POSTs a JSON `WebhookPayload` to the webhook URL:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "idle",
  "response": { "role": "assistant", "content": "hi there" },
  "error": null,
  "messages": null
}
```

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The session ID. |
| `status` | `SessionStatus` | Final status of the turn (`"idle"` or `"error"`). |
| `response` | `ChatMessage \| None` | The agent's response (on success). |
| `error` | `str \| None` | Error message (on failure). |
| `messages` | `list[ChatMessage] \| None` | Full conversation history (only when `include_messages` was `true`). |

#### Delivery behavior

- Webhooks are delivered asynchronously after the turn completes. They do not block the turn itself.
- The HTTP request uses a **10-second timeout**.
- If delivery fails (network error, non-2xx response, timeout), the failure is logged but does **not** affect the turn result. The session state is unchanged.
- When a `token` is provided, the request includes an `Authorization: Bearer <token>` header.

---

## Session Lifecycle

### States

| Status | Description |
|---|---|
| `idle` | Waiting for input. Initial state after creation. |
| `running` | Processing a message. Rejects concurrent sends (409). |
| `error` | Agent raised an exception. `error` field contains the message. |

### Transitions

```
             POST /messages
    idle ──────────────────► running
     ▲                          │
     │          success         │
     └──────────────────────────┤
                                │
              failure           │
    error ◄─────────────────────┘
```

A session in `error` state can receive new messages (transitions back to `running`).

Sessions are held in memory and do not persist across server restarts. When `ttl` is set, a background task periodically sweeps idle and errored sessions whose last activity exceeds the TTL. When `timeout` is set, agent turns that exceed the limit are killed and the session transitions to `error` with an `"Agent timed out"` message.

---

## CLI

The CLI is invoked via `python -m motus.serve`.

### `start`

Start a server from a Python import path.

```
python -m motus.serve start <agent> [options]
```

| Flag | Default | Description |
|---|---|---|
| `agent` | *required* | Import path to agent function (e.g., `myapp:my_agent`) |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port |
| `--workers` | CPU count | Override max worker processes |
| `--ttl` | `0` | TTL for idle/error sessions in seconds (`0` = disabled) |
| `--timeout` | `0` | Max seconds per agent turn before it is killed (`0` = no limit) |
| `--max-sessions` | `0` | Maximum concurrent sessions (`0` = unlimited) |
| `--shutdown-timeout` | `0` | Seconds to wait for in-flight tasks on shutdown before cancelling (`0` = wait indefinitely) |
| `--allow-custom-ids` | `false` | Enable `PUT /sessions/{id}` for client-specified session IDs |
| `--log-level` | `info` | `debug`, `info`, `warning`, `error` |

```bash
python -m motus.serve start myapp:my_agent --port 8080 --workers 8
```

### `chat`

Send a message or enter interactive mode.

```
python -m motus.serve chat <url> [message] [options]
```

| Flag | Default | Description |
|---|---|---|
| `message` | — | Message to send (omit for interactive REPL) |
| `--session` | — | Resume an existing session instead of creating a new one |
| `--keep` | `false` | Don't delete the session on exit (prints session ID for later `--session` use) |

If `message` is omitted, enters a REPL. Without `--session`, a new session is created and deleted on exit. With `--keep`, the session is preserved for later resumption.

```bash
# Single message
python -m motus.serve chat http://localhost:8000 "hello"

# Interactive REPL
python -m motus.serve chat http://localhost:8000
> hello
hi there
> how are you?
I'm doing well!
^C
Bye!

# Keep session for later
python -m motus.serve chat http://localhost:8000 --keep
Session: 550e8400-... (use --session to resume)
> hello
hi there
^C
Bye!

# Resume a previous session
python -m motus.serve chat http://localhost:8000 --session 550e8400-...
> where were we?
```

### `health`

Check server health.

```
python -m motus.serve health <url>
```

```
Status: ok
Workers: 2/4
Total sessions: 2
```

### `create`

Create a new session.

```
python -m motus.serve create <url>
```

```bash
python -m motus.serve create http://localhost:8000
```

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "idle"
}
```

### `sessions`

List all sessions.

```
python -m motus.serve sessions <url>
```

```bash
python -m motus.serve sessions http://localhost:8000
```

```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "total_messages": 4,
    "status": "idle"
  }
]
```

### `get`

Get session details. Supports long-polling with `--wait`.

```
python -m motus.serve get <url> <id> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--wait` | `false` | Block until the session is no longer `"running"` |
| `--timeout` | — | Maximum seconds to wait |

```bash
python -m motus.serve get http://localhost:8000 550e8400-... --wait --timeout 30
```

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "idle",
  "response": { "role": "assistant", "content": "hi there" }
}
```

### `delete`

Delete a session.

```
python -m motus.serve delete <url> <id>
```

```bash
python -m motus.serve delete http://localhost:8000 550e8400-...
Deleted session 550e8400-...
```

### `messages`

Get the full conversation history for a session.

```
python -m motus.serve messages <url> <id>
```

```bash
python -m motus.serve messages http://localhost:8000 550e8400-...
```

```json
[
  { "role": "user", "content": "hello" },
  { "role": "assistant", "content": "hi there" }
]
```

### `send`

Send a message to a session. Returns immediately unless `--wait` is used.

```
python -m motus.serve send <url> <id> <message> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--role` | `user` | Message role |
| `--wait` | `false` | Wait for the turn to complete and print the final session state |
| `--timeout` | — | Maximum seconds to wait (with `--wait`) |
| `--webhook-url` | — | Webhook URL for completion callback |
| `--webhook-token` | — | Bearer token for webhook |
| `--webhook-include-messages` | `false` | Include full message history in webhook payload |

```bash
# Fire and forget
python -m motus.serve send http://localhost:8000 550e8400-... "hello"

# Wait for completion
python -m motus.serve send http://localhost:8000 550e8400-... "hello" --wait
```

---

## Architecture

```
              HTTP Requests
                   │
                   ▼
┌──────────────────────────────────────────┐
│              AgentServer                 │
│            (FastAPI + Uvicorn)           │
│                                          │
│   /health  /sessions  /sessions/{id}     │
│   /sessions/{id}/messages                │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│            WorkerExecutor                │
│     (Semaphore + multiprocessing.Process │
│              + Pipe IPC)                 │
└──────────────────┬───────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │Process 1│ │Process 2│ │Process N│
  │(one-shot│ │(one-shot│ │(one-shot│
  │ worker) │ │ worker) │ │ worker) │
  └─────────┘ └─────────┘ └─────────┘
```

Each message spawns a fresh subprocess via `multiprocessing.Process` with pipe-based IPC. An `asyncio.Semaphore` limits concurrency to `max_workers`. Processes are not reused — each one starts, runs the agent function, sends the result over the pipe, and exits. On timeout or cancellation, the process is killed immediately.

### File Structure

```
serve/
├── __init__.py       # Public export: AgentServer
├── __main__.py       # python -m motus.serve entry point
├── server.py         # AgentServer class (FastAPI routes, background tasks)
├── worker.py         # WorkerExecutor, subprocess execution
├── schemas.py        # Pydantic models (SessionStatus, request/response types)
├── session.py        # Session dataclass and in-memory SessionStore
└── cli.py            # CLI (start, chat, health, create, sessions, get, delete, messages, send)
```

---

## Data Models

All models are defined in `schemas.py`.

### `SessionStatus` (enum)

| Value | Description |
|---|---|
| `"idle"` | Waiting for input |
| `"running"` | Processing a message |
| `"error"` | Agent raised an exception |

### Request / Response Models

| Model | Used by | Fields |
|---|---|---|
| `HealthResponse` | `GET /health` | `status: str`, `max_workers: int`, `running_workers: int`, `total_sessions: int` |
| `CreateSessionRequest` | `POST /sessions`, `PUT /sessions/{id}` (optional body) | `state: list[ChatMessage] = []` |
| `SessionResponse` | `POST /sessions`, `GET /sessions/{id}` | `session_id: str`, `status: SessionStatus`, `response: ChatMessage \| None`, `error: str \| None` |
| `SessionSummary` | `GET /sessions` | `session_id: str`, `total_messages: int`, `status: SessionStatus` |
| `MessageRequest` | `POST .../messages` | Extends `ChatMessage` with `role: Literal["system", "user", "assistant", "tool"] = "user"`, `webhook: WebhookSpec \| None` (inherits `content`, `tool_calls`, `tool_call_id`, `name`, `base64_image`) |
| `MessageResponse` | `POST .../messages` | `session_id: str`, `status: SessionStatus` |
| `WebhookSpec` | `MessageRequest.webhook` | `url: str`, `token: str \| None`, `include_messages: bool` |
| `WebhookPayload` | Webhook delivery | `session_id: str`, `status: SessionStatus`, `response: ChatMessage \| None`, `error: str \| None`, `messages: list[ChatMessage] \| None` |
