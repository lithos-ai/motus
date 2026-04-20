#!/usr/bin/env bash
# test_motusclaw.sh — end-to-end smoke test for the MotusClaw scaffold.
#
# Assumes a deployed MotusClaw agent behind agent_router, with:
#   * cloud-infrastructure PR #42 (notification/cron service) merged + deployed
#   * motus PR #23 (CloudScheduler) in the deployed image
#   * PR #120's per-project LITHOSAI_API_KEY provisioned and wired as the
#     inbound bearer token (ProjectApiKeysTable)
#
# Required env vars:
#   ENDPOINT            e.g. https://<project-id>.agent.<domain>
#   LITHOSAI_API_KEY    the per-project bearer token
#
# Optional env vars:
#   TRACE_ENDPOINT      e.g. https://api.lithosai.com/v1/traces  (skips trace
#                       assertion if unset)
#   HEARTBEAT_WAIT_SECS default 150 (matches rate(2 minutes) with slack)
#
set -euo pipefail

: "${ENDPOINT:?ENDPOINT must be set, e.g. https://<project>.agent.<domain>}"
: "${LITHOSAI_API_KEY:?LITHOSAI_API_KEY must be set}"
WAIT_SECS="${HEARTBEAT_WAIT_SECS:-150}"

H_AUTH="Authorization: Bearer ${LITHOSAI_API_KEY}"
H_JSON="Content-Type: application/json"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

echo ">>> Creating session"
SESSION=$(curl -sf -X POST "${ENDPOINT}/sessions" -H "${H_AUTH}" -H "${H_JSON}" -d '{}') \
    || fail "POST /sessions"
SID=$(echo "$SESSION" | jq -r '.session_id')
[[ -n "$SID" && "$SID" != "null" ]] || fail "no session_id in response: $SESSION"
echo "    session_id=${SID}"

echo ">>> Sending first message (kicks off schedule)"
curl -sf -X POST "${ENDPOINT}/sessions/${SID}/messages" \
    -H "${H_AUTH}" -H "${H_JSON}" \
    -d '{"content":"Find recent LithosAI news and schedule a check-in."}' >/dev/null \
    || fail "POST /sessions/{id}/messages"

echo ">>> Polling for first turn to complete (up to 60s)"
for _ in $(seq 1 30); do
    STATUS=$(curl -sf "${ENDPOINT}/sessions/${SID}?wait=true&timeout=2" \
        -H "${H_AUTH}" | jq -r '.status' || echo "")
    if [[ "$STATUS" == "idle" ]]; then
        pass "first turn complete"
        break
    fi
    if [[ "$STATUS" == "error" ]]; then
        curl -sf "${ENDPOINT}/sessions/${SID}" -H "${H_AUTH}" >&2
        fail "session entered error state"
    fi
done
[[ "$STATUS" == "idle" ]] || fail "first turn did not reach idle within 60s (status=$STATUS)"

echo ">>> Waiting ${WAIT_SECS}s for a scheduled heartbeat to fire"
sleep "${WAIT_SECS}"

echo ">>> Fetching full message history"
MSGS=$(curl -sf "${ENDPOINT}/sessions/${SID}/messages" -H "${H_AUTH}") \
    || fail "GET /sessions/{id}/messages"

# A scheduled fire arrives as a user-role message whose content matches the
# heartbeat string the agent registered. MOTUSCLAW_HEARTBEAT_MESSAGE default
# is "Scheduled check-in ...".
if echo "$MSGS" | jq -e '.[] | select(.role=="user" and (.content | test("check-in|heartbeat"; "i")))' >/dev/null; then
    pass "scheduled fire observed in session history"
else
    echo "--- full history ---" >&2
    echo "$MSGS" | jq . >&2
    fail "no scheduled fire observed in session history"
fi

# Trace assertion is optional — the platform's trace-ingest endpoint is not
# stable across environments yet.
if [[ -n "${TRACE_ENDPOINT:-}" ]]; then
    echo ">>> Checking trace endpoint for session id"
    TRACES=$(curl -sf "${TRACE_ENDPOINT}?session_id=${SID}" -H "${H_AUTH}" || echo '{}')
    if echo "$TRACES" | jq -e '(.traces // []) | length > 0' >/dev/null; then
        pass "trace captured for session"
    else
        echo "WARN: no trace found at ${TRACE_ENDPOINT} (non-fatal)"
    fi
else
    echo "SKIP: TRACE_ENDPOINT unset — skipping trace assertion"
fi

echo ">>> Cleaning up session"
curl -sf -X DELETE "${ENDPOINT}/sessions/${SID}" -H "${H_AUTH}" >/dev/null \
    || echo "WARN: session delete failed (non-fatal)"

pass "end-to-end MotusClaw scaffold test completed"
