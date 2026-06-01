# Deploy Reference

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Not authenticated` | No credentials | Run `motus login` to provision API credentials |
| 401/403 | Invalid or expired API key | Run `motus login` to re-authenticate |
| Build `Failed` with no logs | Build service temporarily unavailable | Retry, or contact support |
| Build succeeds but agent unreachable | Container crashes on startup — missing dependencies | Ensure `requirements.txt` includes all needed packages |
| `KeyError: 'OPENROUTER_API_KEY'` or similar / `Backend unavailable` | Agent code uses `os.environ["..."]` instead of letting the SDK read env vars | Use SDK defaults (e.g., `OpenAIChatClient()` with no args) — the platform auto-wires API keys and base URLs via the model proxy |
| `TypeError: xxx() takes 1 positional argument but 2 were given` | Agent function doesn't match the `(message, state)` contract | Wrap the function — see Agent Function Contract in SKILL.md |
| `AttributeError: 'str' object has no attribute 'content'` | Agent expects `str` but receives `ChatMessage` | Change parameter type or access `message.content` |
| `Could not import module: motus.openai_agents requires the OpenAI Agents SDK` | `openai-agents` not installed in container | Add `openai-agents` to `requirements.txt` and redeploy |
| `502 Failed to provision sandbox` on session create | Platform sandbox service unavailable or tenant quota | Retry; if persistent, contact support |

## Cloud Agent REST API

The deployed agent exposes a session-based REST API:

```
POST   /sessions                          — create session
GET    /sessions                          — list sessions
GET    /sessions/{id}                     — get session (add ?wait=true for long-poll)
DELETE /sessions/{id}                     — delete session
POST   /sessions/{id}/messages            — send message (returns 202, async)
GET    /sessions/{id}/messages            — get message history
```

Authentication: `Authorization: Bearer <api-key>` (from `motus login` credentials or `LITHOSAI_API_KEY` env var)

## SDK Import Mapping Details

The motus wrapper (`motus.openai_agents`) is a transparent drop-in replacement:

- Re-export all symbols from the original SDK via `from <sdk> import *`
- Wrap key entry points to inject tracing
- No behavior change — identical API surface, just with observability added
