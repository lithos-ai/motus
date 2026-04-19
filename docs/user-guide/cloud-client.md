# Cloud Client

`motus.cloud` is a programmatic Python client for invoking Motus agents — both ones served locally via `motus serve` and ones deployed to [Motus Cloud](https://console.lithosai.com/). It wraps the session/message REST protocol with typed responses, typed exceptions, interrupt handling, and automatic credential resolution.

## Quick start

```python
from motus.cloud import Client

with Client(base_url="http://localhost:8000") as c:
    result = c.chat("hello")
    print(result.message.content)
```

Against a deployed agent:

```python
import os
from motus.cloud import Client

client = Client(
    base_url="https://my-agent-xxxx.agent.lithosai.cloud",
    api_key=os.environ["LITHOSAI_API_KEY"],
)
```

Async equivalent:

```python
from motus.cloud import AsyncClient

async with AsyncClient(base_url="http://localhost:8000") as c:
    result = await c.chat("hello")
```

## Authentication

The client resolves an API key in this order (first non-empty wins):

1. The explicit `api_key=` constructor argument.
2. The `LITHOSAI_API_KEY` environment variable.
3. `~/.motus/credentials.json` (populated by `motus login`).

Passing `api_key=""` disables auth. If no key is found, no `Authorization` header is sent — fine for a local `motus serve` that isn't enforcing auth.

## Chat modes

### One-shot (`Client.chat`)

`Client.chat(content)` creates an ephemeral server session, sends the message, polls to a terminal state, and deletes the session on a clean idle exit. Interrupted, errored, or timed-out turns intentionally leave the server session alive so you can resume or inspect it.

```python
result = c.chat("what's 2+2?")
assert result.status.value == "idle"
print(result.message.content)
```

### Multi-turn (`Client.session`)

Use `client.session()` for multi-turn conversations. It returns a pinned `Session` with an owned server-side session that is deleted on context exit — unless `keep=True` is passed.

```python
with c.session() as s:
    s.chat("who are you?")
    s.chat("tell me more")
```

- `session(keep=True)` — keep the server session alive after close (prints the session_id via `logger.info`). Re-attach later with `session(session_id=...)`.
- `session(session_id=<existing>)` — attach to a session the caller already owns. **`close()` never issues `DELETE`** regardless of `keep`; the caller is responsible for cleanup.
- `session(initial_state=[...])` — seed the conversation state on creation. Cannot be combined with an explicit `session_id`.

## Interrupts

When the agent raises an interrupt (e.g., `tool_approval`, `user_input`), `ChatResult.status` is `interrupted` and `ChatResult.interrupts` contains typed `Interrupt(id, type, payload)` objects. Resume with the value the interrupt expects:

```python
result = c.chat("please delete /tmp/test.txt")
if result.status.value == "interrupted":
    # result.interrupts holds exactly one Interrupt here; use the convenience:
    result = result.resume({"approved": True})
print(result.message.content)
```

Or explicitly with multiple pending interrupts:

```python
result = c.resume(result.session_id, result.interrupts[0].id, {"approved": True})
```

## Timeouts and retries

Two independent knobs:

| Knob | Default | Meaning |
|------|---------|---------|
| `turn_timeout` | `None` (wait indefinitely) | Overall deadline for a single `chat()` turn including polling. Exceeding raises `SessionTimeout`. |
| `http_timeout` | `httpx.Timeout(connect=5, read=120, write=10, pool=5)` | Per-request `httpx` timeout. |
| `server_wait_slice` | `90.0` s | Maximum `?timeout=` passed to each long-poll `GET`. |
| `read_retry_budget` | `3` | Consecutive `httpx.TimeoutException` retries on polling `GET` before giving up with `BackendUnavailable`. |

`POST /messages`, `POST /resume`, and `DELETE /sessions/{id}` are never auto-retried — these are not idempotent server-side. If you want retries there, wrap the client with your own retry policy.

## Exceptions

All public methods raise subclasses of `MotusClientError`:

| Class | Trigger |
|-------|---------|
| `AuthError` | HTTP 401 / 403 |
| `SessionNotFound` | HTTP 404 on a session path |
| `InterruptNotFound` | HTTP 404 from `POST /resume` with an unknown `interrupt_id` |
| `SessionConflict` | HTTP 409 (send-while-running, resume mismatch) |
| `ServerBusy` | HTTP 503 (max sessions reached) |
| `BackendUnavailable` | Other 5xx, connect errors, or poll retry budget exhausted |
| `SessionTimeout` | `turn_timeout` exceeded while the server session is still running |
| `AgentError` | HTTP 200 with `session.status == "error"` |
| `ProtocolError` | Malformed / non-JSON response body |
| `SessionClosed` | Calling `Session.chat` / `Session.resume` after `close()` |
| `AmbiguousInterrupt` | `ChatResult.resume(value)` with zero or multiple pending interrupts |
| `SessionUnsupported` | HTTP 405 on `PUT /sessions/{id}` — server wasn't started with `--allow-custom-ids` |

`SessionTimeout` carries `session_id`, `elapsed`, and the last-known `SessionResponse` snapshot so you can reconnect via `client.get_session(session_id, wait=True)`.

Raw `httpx` exceptions never leak out.

## Session state is in-memory

The `motus serve` session store is **in-memory only** (no database) and sessions may be swept by a TTL sweep when the server is started with `--ttl`. Treat server-side session history as best-effort; if you need durable conversation logs, persist them on your side (e.g., store each `result.message` as you go).

## Low-level methods

For power users, the full REST protocol is exposed as thin wrappers:

- `client.health()` → `{status, max_workers, running_workers, total_sessions}`
- `client.create_session(session_id=?, initial_state=?)` → `SessionResponse`
- `client.get_session(session_id, wait=?, timeout=?)` → `SessionResponse`
- `client.list_sessions()` → `list[dict]`
- `client.delete_session(session_id)` — 404 is silent
- `client.get_messages(session_id)` → `list[dict]`
- `client.send_message(session_id, content, role=?, user_params=?, webhook=?, **fields)` → raw MessageResponse
- `client.resume(session_id, interrupt_id, value)` → `ChatResult`

Every method accepts `extra_headers=` and (on the constructor) `transport=` or `http_client=` for custom `httpx` wiring (mocks, proxies, custom CA bundles, mTLS).

## Not yet supported

Token-level streaming is not in this release. The server does not emit per-token events. A coarse session-event iterator (`chat_events()`) is planned for a future version and is intentionally not shipped now to avoid implying token streaming.
