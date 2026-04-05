# Post-Deploy Guide

## Interacting with a Deployed Agent

Base URL: `https://{project-id}.agent.{your-subdomain}`

### CLI (recommended)

```bash
# Interactive chat (creates session, multi-turn, deletes on exit)
motus serve chat https://my-project.agent.example.lithosai.cloud

# Single message
motus serve chat https://my-project.agent.example.lithosai.cloud "hello"

# Keep session alive after exit (prints session ID)
motus serve chat https://my-project.agent.example.lithosai.cloud --keep

# Resume an existing session
motus serve chat https://my-project.agent.example.lithosai.cloud --session <id>
```

### curl

```bash
BASE=https://my-project.agent.example.lithosai.cloud
AUTH="Authorization: Bearer $LITHOSAI_API_KEY"

# Create session
curl -X POST $BASE/sessions -H "$AUTH"
# → {"session_id": "abc-123", "status": "idle"}

# Send message (async — returns 202 immediately)
curl -X POST $BASE/sessions/abc-123/messages \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"content": "hello"}'

# Wait for response (long-poll, blocks until turn completes)
curl "$BASE/sessions/abc-123?wait=true" -H "$AUTH"

# Get conversation history
curl $BASE/sessions/abc-123/messages -H "$AUTH"

# Delete session
curl -X DELETE $BASE/sessions/abc-123 -H "$AUTH"
```

### Webhook (async notification)

Send a message with a webhook spec to get notified when the turn completes:

```bash
curl -X POST $BASE/sessions/abc-123/messages \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "content": "hello",
    "webhook": {
      "url": "https://your-server.com/callback",
      "token": "your-bearer-token",
      "include_messages": true
    }
  }'
```

The platform will POST to your webhook URL with `{session_id, status, response, messages}`.

## Health & Status Checks

```bash
# Server health (workers, sessions count)
motus serve health https://my-project.agent.example.lithosai.cloud

# List all active sessions
motus serve sessions https://my-project.agent.example.lithosai.cloud

# Get session details
motus serve get https://my-project.agent.example.lithosai.cloud <session-id>

# Get session details and wait for completion
motus serve get https://my-project.agent.example.lithosai.cloud <session-id> --wait
```

## Debugging

### Agent returns errors

```bash
# Check session status — look for "error" status and error message
curl "$BASE/sessions/<id>" -H "$AUTH" | python -m json.tool
```

Common error causes:
- Agent function raised an exception → error field contains traceback
- Agent timed out → "Agent timed out"
- Import error in agent code → check import path and dependencies

### Build failed

- Build shows "Failed" with no logs → the build service may be temporarily unavailable, retry or contact support
- Check build status: `motus deploy status <build-id>` (if available)

### Agent unreachable (DNS / connection error)

1. Verify DNS resolves: `nslookup {project}.agent.{subdomain}`
2. Build may still be in progress — deployment typically takes a few minutes
3. If the issue persists after 5 minutes, contact support

### Session state issues

- Sessions may be lost if the underlying service restarts
- Long-running sessions may hit TTL expiry if configured

## Testing Patterns

### Smoke test after deploy

```bash
# Quick round-trip test
motus serve chat https://my-project.agent.{subdomain} "ping"
# Expect a response within a few seconds
```

### Load test with multiple sessions

```bash
# Create multiple sessions in parallel
for i in $(seq 1 5); do
  curl -s -X POST $BASE/sessions -H "$AUTH" &
done
wait
```

### Compare local vs cloud

```bash
# Local
motus serve start agent:my_agent --port 8001
motus serve chat http://localhost:8001 "test message"

# Cloud
motus serve chat https://my-project.agent.{subdomain} "test message"

# Compare responses — they should be identical
```

## Cloud Tracing

If agent code uses motus wrappers (`motus.claude_agent` / `motus.openai_agents`), traces are automatically uploaded to the cloud.

Traces can be viewed via the LITHOSAI console. Each agent turn produces a trace with model calls, tool invocations, and timing data.
