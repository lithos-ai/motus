#!/usr/bin/env python3
"""Deploy OAI examples to cloud and test multi-turn conversations.

Deploys each example via `motus deploy` (git-based), creates sessions,
runs multi-turn conversations using Claude as client, and verifies
session state persistence.

Usage::

    set -a && source .env && set +a
    python tests/third_party_compat/cloud_deploy_test.py
    python tests/third_party_compat/cloud_deploy_test.py --filter tools
    python tests/third_party_compat/cloud_deploy_test.py --skip-deploy  # reuse existing deployments
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anthropic
import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBDOMAIN = "lithosai.cloud"
API_URL = os.environ.get("LITHOSAI_API_URL", f"https://api.{SUBDOMAIN}")
API_KEY = os.environ.get("LITHOSAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Example catalog — module-level agents deployable via serve
# ---------------------------------------------------------------------------

ExampleDef = dict


def ex(name: str, import_path: str, desc: str, skip: str | None = None) -> ExampleDef:
    return {
        "name": name,
        "project_id": f"oai-{name}",
        "import_path": import_path,
        "desc": desc,
        "skip": skip,
    }


EXAMPLES: list[ExampleDef] = [
    ex(
        "tools",
        "tools:agent",
        "Agent with get_weather tool. Ask about weather in various cities.",
    ),
    ex(
        "tool-guardrails",
        "tool_guardrails:agent",
        "Agent with tools guarded by input/output guardrails.",
    ),
    ex(
        "routing",
        "routing:triage_agent",
        "Language-routing triage agent. Say something in French, Spanish, or English.",
    ),
    ex(
        "agents-as-tools",
        "agents_as_tools:orchestrator_agent",
        "Orchestrator with Spanish, French, and Italian translator sub-agents.",
    ),
    ex(
        "llm-judge",
        "llm_as_a_judge:story_outline_generator",
        "Generates story outlines.",
    ),
    ex(
        "output-guardrails",
        "output_guardrails:agent",
        "Agent with output guardrail. Ask general knowledge questions.",
    ),
    ex(
        "streaming-guardrails",
        "streaming_guardrails:agent",
        "Agent with streaming guardrails. Ask a general question.",
    ),
    ex(
        "human-in-loop",
        "human_in_the_loop:agent",
        "Agent with get_weather and send_email tools.",
    ),
    ex(
        "message-filter",
        "message_filter:first_agent",
        "Multi-agent chain with handoffs and message filters.",
    ),
    ex(
        "message-filter-stream",
        "message_filter_streaming:first_agent",
        "Multi-agent chain with streaming handoffs.",
    ),
]


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


def _headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------


def deploy_example(example: ExampleDef) -> bool:
    """Deploy an example via `motus deploy` (local tarball upload).

    Deploys from examples/openai_agents/ so the Dockerfile picks up
    requirements.txt (openai-agents SDK) instead of the repo-root
    pyproject.toml.
    """
    project_id = example["project_id"]
    # Import paths are relative to the examples dir (e.g., "tools:agent")
    import_path = example["import_path"]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "motus",
            "deploy",
            "--project-id",
            project_id,
            "--secret",
            "OPENAI_API_KEY",
            "--secret",
            "OPENAI_BASE_URL=https://api.openai.com/v1",
            import_path,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT / "examples" / "openai_agents"),
    )

    if result.returncode == 0:
        print("    Deployed!")
        return True
    else:
        last_lines = result.stdout.strip().split("\n")[-3:]
        err_lines = (
            result.stderr.strip().split("\n")[-3:] if result.stderr.strip() else []
        )
        for line in last_lines + err_lines:
            print(f"    {line}")
        return False


def wait_healthy(project_id: str, timeout: float = 60) -> bool:
    """Wait for a deployed agent to become healthy."""
    base = f"https://{project_id}.agent.{SUBDOMAIN}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/health", headers=_headers(), timeout=5)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Multi-turn test
# ---------------------------------------------------------------------------


def _run_multi_turn(example: ExampleDef, claude: anthropic.Anthropic) -> dict:
    """Run a 3-turn conversation and verify state persistence."""
    base = f"https://{example['project_id']}.agent.{SUBDOMAIN}"
    headers = _headers()

    # Create session
    r = httpx.post(f"{base}/sessions", headers=headers, timeout=10)
    if r.status_code != 201:
        return {
            "status": "fail",
            "reason": f"Create session: {r.status_code} {r.text[:200]}",
        }
    session_id = r.json()["session_id"]

    conversation: list[dict] = []
    turns_completed = 0

    for turn in range(3):
        # Generate user message
        if turn == 0:
            user_msg = (
                claude.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=60,
                    system=f"You are testing an AI agent: {example['desc']}. Generate a short test message. Output ONLY the message.",
                    messages=[
                        {"role": "user", "content": "Generate the first test message."}
                    ],
                )
                .content[0]
                .text.strip()
            )
        else:
            user_msg = (
                claude.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=60,
                    system=f"You are testing an AI agent: {example['desc']}. Generate a follow-up that references the prior conversation.",
                    messages=conversation
                    + [{"role": "user", "content": "Generate a follow-up."}],
                )
                .content[0]
                .text.strip()
            )

        conversation.append({"role": "user", "content": user_msg})

        # Send message
        r = httpx.post(
            f"{base}/sessions/{session_id}/messages",
            json={"content": user_msg},
            headers=headers,
            timeout=10,
        )
        if r.status_code != 202:
            return {
                "status": "fail",
                "reason": f"Turn {turn + 1} send: {r.status_code}",
            }

        # Long-poll for response
        r = httpx.get(
            f"{base}/sessions/{session_id}",
            params={"wait": "true", "timeout": "90"},
            headers=headers,
            timeout=100,
        )
        data = r.json()

        if data.get("status") == "error":
            return {
                "status": "fail",
                "reason": f"Turn {turn + 1} error: {data.get('error', '')[:300]}",
            }

        agent_response = data.get("response", {}).get("content", "")
        if not agent_response:
            return {"status": "fail", "reason": f"Turn {turn + 1}: empty response"}

        conversation.append({"role": "assistant", "content": agent_response})
        turns_completed += 1

    # Verify message history persists
    r = httpx.get(f"{base}/sessions/{session_id}/messages", headers=headers, timeout=10)
    if r.status_code == 200:
        messages = r.json()
        stored_count = len(messages)
    else:
        stored_count = -1

    # Evaluate
    passed, evaluation = _evaluate(claude, example["desc"], conversation)

    # Cleanup
    httpx.delete(f"{base}/sessions/{session_id}", headers=headers, timeout=5)

    return {
        "status": "pass" if passed else "fail",
        "turns": turns_completed,
        "stored_messages": stored_count,
        "expected_messages": turns_completed * 2,  # user + assistant per turn
        "state_ok": stored_count == turns_completed * 2,
        "evaluation": evaluation,
        "first_response": conversation[1]["content"][:80]
        if len(conversation) > 1
        else "",
    }


def _evaluate(
    client: anthropic.Anthropic, desc: str, conversation: list[dict]
) -> tuple[bool, str]:
    convo_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content'][:200]}"
        for m in conversation
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=f"You are evaluating an agent: {desc}\nDid it respond coherently across all turns, maintaining context? Reply PASS or FAIL with a brief reason.",
        messages=[{"role": "user", "content": convo_text}],
    )
    line = resp.content[0].text.strip()
    return line.upper().startswith("PASS"), line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Cloud deploy + multi-turn test")
    parser.add_argument("--filter", help="Only run examples matching this substring")
    parser.add_argument(
        "--skip-deploy", action="store_true", help="Reuse existing deployments"
    )
    parser.add_argument("--json", help="Write results to JSON file")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: LITHOSAI_API_KEY not set")
        sys.exit(1)

    examples = EXAMPLES
    if args.filter:
        examples = [
            e
            for e in examples
            if args.filter in e["name"] or args.filter in e.get("desc", "")
        ]

    print("=" * 90)
    print("CLOUD DEPLOY + MULTI-TURN TEST")
    print("=" * 90)
    print(f"API: {API_URL}")
    print(f"Agent domain: *.agent.{SUBDOMAIN}")
    print("Deploy: local tarball")
    print(f"Examples: {len(examples)}")
    print()

    claude = anthropic.Anthropic()
    results = []
    counts = {"pass": 0, "fail": 0, "skip": 0}

    for example in examples:
        if example.get("skip"):
            print(f"  [SKIP] {example['name']:<30} {example['skip']}")
            counts["skip"] += 1
            results.append(
                {"name": example["name"], "status": "skip", "reason": example["skip"]}
            )
            continue

        print(f"  [{example['name']}]")

        # Deploy
        if not args.skip_deploy:
            print(f"    Deploying {example['import_path']}...")
            if not deploy_example(example):
                counts["fail"] += 1
                results.append(
                    {
                        "name": example["name"],
                        "status": "fail",
                        "reason": "deploy failed",
                    }
                )
                continue

        # Wait for healthy
        print("    Waiting for health...")
        if not wait_healthy(example["project_id"]):
            print("    Not healthy after 60s")
            counts["fail"] += 1
            results.append(
                {"name": example["name"], "status": "fail", "reason": "not healthy"}
            )
            continue

        # Multi-turn test
        print("    Running 3-turn conversation...")
        result = _run_multi_turn(example, claude)

        if result["status"] == "pass":
            state_str = f"state={'OK' if result.get('state_ok') else 'MISMATCH'} ({result.get('stored_messages')}/{result.get('expected_messages')})"
            print(f"    PASS — {result['turns']} turns, {state_str}")
            print(f"    First response: {result.get('first_response', '')[:60]}")
            counts["pass"] += 1
        else:
            reason = result.get("reason", result.get("evaluation", ""))[:80]
            print(f"    FAIL — {reason}")
            counts["fail"] += 1

        results.append({"name": example["name"], **result})

    print(f"\n{'=' * 90}")
    print(
        f"Results: {counts['pass']} pass, {counts['fail']} fail, {counts['skip']} skip"
    )
    print(f"{'=' * 90}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nJSON results written to {args.json}")


if __name__ == "__main__":
    main()
