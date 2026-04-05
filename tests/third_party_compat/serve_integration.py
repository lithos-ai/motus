#!/usr/bin/env python3
"""Integration tests: deploy OAI Agents SDK examples via serve.

For each example, starts a serve server, creates a session, uses Claude
as a conversational test driver, and verifies round-trip message handling.

Usage::

    set -a && source .env && set +a
    python tests/third_party_compat/serve_integration.py
    python tests/third_party_compat/serve_integration.py --filter routing
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import anthropic
import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADAPTER_PATH = "tests.third_party_compat.serve_adapter:agent"
BASE_PORT = 19100

# ---------------------------------------------------------------------------
# Example catalog
# ---------------------------------------------------------------------------
# module: dotted module name importable from the OAI repo root
# agent_var: module-level variable name of the entry-point Agent
# desc: what the agent does (given to Claude as context)
# skip: reason to skip, or None

ExampleDef = dict


def ex(
    module: str,
    agent_var: str = "agent",
    desc: str = "",
    skip: str | None = None,
) -> ExampleDef:
    return {
        "module": module,
        "agent_var": agent_var,
        "desc": desc,
        "skip": skip,
    }


EXAMPLES: list[ExampleDef] = [
    # -- tools --
    ex(
        "tools",
        desc="Agent with get_weather tool. Ask about weather.",
    ),
    ex(
        "tool_guardrails",
        desc="Agent with tools guarded by input/output guardrails.",
    ),
    # -- agent patterns --
    ex(
        "routing",
        agent_var="triage_agent",
        desc="Language-routing triage agent. Say something in French, Spanish, or English.",
    ),
    ex(
        "agents_as_tools",
        agent_var="orchestrator_agent",
        desc="Orchestrator with Spanish, French, and Italian translator sub-agents. Ask for a translation into one of those languages.",
    ),
    ex(
        "agents_as_tools_structured",
        agent_var="orchestrator",
        desc="Orchestrator with structured agent-tool inputs for translation.",
    ),
    ex(
        "llm_as_a_judge",
        agent_var="story_outline_generator",
        desc="Generates story outlines. Ask it for a short story outline.",
    ),
    ex(
        "output_guardrails",
        desc="Helpful assistant agent. Ask it a general knowledge question (not math homework).",
    ),
    ex(
        "streaming_guardrails",
        desc="Agent with streaming guardrails. Ask a general question.",
    ),
    ex(
        "human_in_the_loop",
        desc="Agent with get_weather and send_email tools that require human approval. Ask about the weather or to send an email.",
    ),
    # -- handoffs --
    ex(
        "message_filter",
        agent_var="first_agent",
        desc="Multi-agent chain with handoffs and message filters. Greet it.",
    ),
    ex(
        "message_filter_streaming",
        agent_var="first_agent",
        desc="Multi-agent chain with streaming handoffs. Greet it.",
    ),
]

# ---------------------------------------------------------------------------
# Claude test driver
# ---------------------------------------------------------------------------


def _claude_generate_message(
    client: anthropic.Anthropic, agent_desc: str, conversation: list[dict]
) -> str:
    """Use Claude to generate the next user message for testing an agent."""
    system = (
        f"You are a QA tester. You are testing an AI agent described as:\n"
        f"{agent_desc}\n\n"
        f"Generate a short, natural user message to test this agent. "
        f"If this is the first message, introduce yourself and ask something "
        f"relevant. If you're continuing a conversation, respond naturally to "
        f"advance it. Keep messages under 30 words. Output ONLY the message."
    )
    messages = conversation or [
        {"role": "user", "content": "Generate the first test message."}
    ]
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=system,
        messages=messages,
    )
    return resp.content[0].text.strip()


def _claude_evaluate(
    client: anthropic.Anthropic, agent_desc: str, conversation: list[dict]
) -> tuple[bool, str]:
    """Use Claude to evaluate whether the agent responded appropriately."""
    system = (
        f"You are a QA evaluator. The agent being tested is:\n"
        f"{agent_desc}\n\n"
        f"Given the conversation below, did the agent respond reasonably? "
        f"Respond with exactly one line: PASS or FAIL followed by a brief reason."
    )
    convo_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in conversation
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=system,
        messages=[{"role": "user", "content": convo_text}],
    )
    line = resp.content[0].text.strip()
    passed = line.upper().startswith("PASS")
    return passed, line


# ---------------------------------------------------------------------------
# serve lifecycle helpers
# ---------------------------------------------------------------------------


def _start_server(module: str, agent_var: str, port: int) -> subprocess.Popen:
    """Start a serve server as a subprocess."""
    env = {
        **os.environ,
        "SERVE_EXAMPLE_MODULE": module,
        "SERVE_AGENT_VAR": agent_var,
        "PYTHONDONTWRITEBYTECODE": "1",
        # Worker subprocesses (forkserver) need repo root on PYTHONPATH
        # so they can resolve the adapter module
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    # Use a small inline script that imports AgentServer directly
    # (serve has no __main__.py; the CLI is via `motus serve start`)
    script = (
        "from motus.serve.server import AgentServer; "
        f"AgentServer({ADAPTER_PATH!r}, max_workers=1, timeout=120)"
        f".run(host='127.0.0.1', port={port}, log_level='error')"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _wait_healthy(port: int, timeout: float = 30) -> bool:
    """Poll health endpoint until server is ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(0.5)
    return False


def _stop_server(proc: subprocess.Popen):
    """Gracefully stop the serve process."""
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        proc.kill()
        proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# Single example runner
# ---------------------------------------------------------------------------


def run_example(example: ExampleDef, port: int, claude: anthropic.Anthropic) -> dict:
    """Deploy an example via serve and run a Claude-driven conversation."""
    if example["skip"]:
        return {"status": "skip", "reason": example["skip"]}

    proc = None
    try:
        # 1. Start server
        proc = _start_server(example["module"], example["agent_var"], port)

        if not _wait_healthy(port):
            stderr = proc.stderr.read().decode()[-500:] if proc.stderr else ""
            return {"status": "fail", "reason": f"Server failed to start: {stderr}"}

        base = f"http://127.0.0.1:{port}"

        # 2. Create session
        r = httpx.post(f"{base}/sessions", timeout=5)
        if r.status_code != 201:
            return {
                "status": "fail",
                "reason": f"Create session: {r.status_code} {r.text[:200]}",
            }
        session_id = r.json()["session_id"]

        # 3. Claude-driven conversation (up to 3 turns)
        conversation: list[dict] = []
        for turn in range(3):
            # Generate user message
            if turn == 0:
                user_msg = _claude_generate_message(claude, example["desc"], [])
            else:
                # Build valid alternating messages for Claude
                # Conversation so far is [user, assistant, user, assistant, ...]
                # Append a user turn asking for a follow-up
                follow_up_msgs = conversation + [
                    {
                        "role": "user",
                        "content": "Generate a follow-up message to continue testing.",
                    },
                ]
                user_msg = _claude_generate_message(
                    claude, example["desc"], follow_up_msgs
                )

            conversation.append({"role": "user", "content": user_msg})

            # Send to serve
            r = httpx.post(
                f"{base}/sessions/{session_id}/messages",
                json={"content": user_msg},
                timeout=10,
            )
            if r.status_code != 202:
                return {
                    "status": "fail",
                    "reason": f"Send message: {r.status_code} {r.text[:200]}",
                }

            # Wait for response (long-poll)
            r = httpx.get(
                f"{base}/sessions/{session_id}",
                params={"wait": "true", "timeout": "90"},
                timeout=100,
            )
            data = r.json()

            if data.get("status") == "error":
                return {
                    "status": "fail",
                    "reason": f"Agent error: {data.get('error', 'unknown')[:800]}",
                }

            agent_response = data.get("response", {}).get("content", "")
            if not agent_response:
                return {"status": "fail", "reason": "Empty agent response"}

            conversation.append({"role": "assistant", "content": agent_response})

            # After first turn, check if we should continue
            if turn == 0 and len(agent_response) > 20:
                # Got a substantive response on first turn — that's enough for most agents
                break

        # 4. Evaluate with Claude
        passed, evaluation = _claude_evaluate(claude, example["desc"], conversation)

        return {
            "status": "pass" if passed else "fail",
            "turns": len(conversation) // 2,
            "evaluation": evaluation,
            "first_response": conversation[1]["content"][:80]
            if len(conversation) > 1
            else "",
        }

    except Exception as e:
        return {"status": "fail", "reason": f"Exception: {e}"}

    finally:
        if proc:
            _stop_server(proc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="serve integration tests")
    parser.add_argument("--filter", help="Only run examples matching this substring")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", help="Write results to JSON file")
    args = parser.parse_args()

    examples = EXAMPLES
    if args.filter:
        examples = [
            e
            for e in examples
            if args.filter in e["module"] or args.filter in e.get("desc", "")
        ]

    print("=" * 90)
    print("SERVE INTEGRATION TESTS")
    print("=" * 90)
    print(f"Examples: {len(examples)} total")
    print()

    if args.dry_run:
        for e in examples:
            status = "SKIP" if e["skip"] else "RUN"
            print(f"  [{status:4}] {e['module']}:{e['agent_var']}")
        return

    claude = anthropic.Anthropic()
    counts = {"pass": 0, "fail": 0, "skip": 0}
    results = []

    for i, example in enumerate(examples):
        port = BASE_PORT + i
        label = f"{example['module']}:{example['agent_var']}"

        if example["skip"]:
            print(f"  [SKIP] {label:<65} {example['skip']}")
            counts["skip"] += 1
            results.append(
                {"example": label, "status": "skip", "reason": example["skip"]}
            )
            continue

        print(f"  [....] {label:<65}", end="", flush=True)
        result = run_example(example, port, claude)

        if result["status"] == "pass":
            print(f"\r  [PASS] {label:<65} {result.get('first_response', '')[:30]}")
            counts["pass"] += 1
        else:
            reason = result.get("reason", result.get("evaluation", ""))[:60]
            print(f"\r  [FAIL] {label:<65} {reason}")
            counts["fail"] += 1

        results.append({"example": label, **result})

    print(f"\n{'=' * 90}")
    print(
        f"Results: {counts['pass']} pass, {counts['fail']} fail, {counts['skip']} skip"
    )
    print(f"{'=' * 90}")

    if counts["fail"] > 0:
        print("\nFAILURES:")
        for r in results:
            if r["status"] == "fail":
                print(f"  {r['example']}:")
                print(f"    {r.get('reason', r.get('evaluation', ''))}")
                print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nJSON results written to {args.json}")


if __name__ == "__main__":
    main()
