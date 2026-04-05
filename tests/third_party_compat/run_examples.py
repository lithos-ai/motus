#!/usr/bin/env python3
"""Run OpenAI Agents SDK examples through motus and report results.

Each example in examples/openai_agents/ has its imports patched to use
motus.openai_agents, so they run with motus tracing with no hook needed.

Usage::

    set -a && source .env && set +a
    python tests/third_party_compat/run_examples.py
    python tests/third_party_compat/run_examples.py --filter routing
    python tests/third_party_compat/run_examples.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "openai_agents"

# ---------------------------------------------------------------------------
# Example catalog
# ---------------------------------------------------------------------------

ExampleDef = dict


def ex(
    path: str,
    timeout: int = 45,
    skip: str | None = None,
    env: dict | None = None,
    notes: str = "",
) -> ExampleDef:
    return {
        "path": path,
        "timeout": timeout,
        "skip_reason": skip,
        "env_extra": env or {},
        "notes": notes,
    }


EXAMPLES: list[ExampleDef] = [
    ex("tools.py"),
    ex("tool_guardrails.py"),
    ex("routing.py", timeout=60, notes="interactive, auto_mode, multi-turn"),
    ex("agents_as_tools.py", timeout=60, notes="interactive, auto_mode"),
    ex("agents_as_tools_structured.py", timeout=60),
    ex("llm_as_a_judge.py", timeout=90, notes="interactive, auto_mode"),
    ex("output_guardrails.py"),
    ex("streaming_guardrails.py"),
    ex("human_in_the_loop.py", notes="interactive, auto_mode"),
    ex("message_filter.py", timeout=60),
    ex("message_filter_streaming.py", timeout=60),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_example(example: ExampleDef) -> dict:
    """Run a single example, return result dict."""
    script = EXAMPLES_DIR / example["path"]
    if not script.exists():
        return {"status": "skip", "reason": f"file not found: {script}"}

    if example["skip_reason"]:
        return {"status": "skip", "reason": example["skip_reason"]}

    if not os.environ.get("OPENAI_API_KEY"):
        return {"status": "skip", "reason": "no OPENAI_API_KEY"}

    env = {
        **os.environ,
        **example["env_extra"],
        "EXAMPLES_INTERACTIVE_MODE": "auto",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=example["timeout"],
            env=env,
            cwd=str(EXAMPLES_DIR),
        )
        if result.returncode == 0:
            first_line = (
                result.stdout.strip().split("\n")[0][:80]
                if result.stdout.strip()
                else "(no output)"
            )
            return {"status": "pass", "output": first_line}
        else:
            last_lines = result.stderr.strip().split("\n")[-3:]
            return {"status": "fail", "error": "\n".join(last_lines)}
    except subprocess.TimeoutExpired:
        return {"status": "fail", "error": f"TIMEOUT after {example['timeout']}s"}


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenAI Agents SDK examples through motus"
    )
    parser.add_argument("--filter", help="Only run examples matching this substring")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--json", help="Write results to JSON file")
    args = parser.parse_args()

    examples = EXAMPLES
    if args.filter:
        examples = [ex for ex in examples if args.filter in ex["path"]]

    print("=" * 80)
    print("OPENAI AGENTS SDK EXAMPLE RUNNER")
    print("=" * 80)
    print(f"Examples dir: {EXAMPLES_DIR}")
    print(f"OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")
    print(f"Examples: {len(examples)} total")
    print()

    counts = {"pass": 0, "fail": 0, "skip": 0}
    results = []

    for example in examples:
        path = example["path"]

        if args.dry_run:
            status = "SKIP" if example["skip_reason"] else "RUN"
            reason = example["skip_reason"] or ""
            print(f"  [{status:4}] {path:<50} {reason}")
            continue

        if example["skip_reason"]:
            print(f"  [SKIP] {path:<45} {example['skip_reason']}")
            counts["skip"] += 1
            results.append(
                {"path": path, "status": "skip", "reason": example["skip_reason"]}
            )
            continue

        result = run_example(example)

        if result["status"] == "skip":
            print(f"  [SKIP] {path:<45} {result['reason']}")
            counts["skip"] += 1
        elif result["status"] == "pass":
            print(f"  [PASS] {path:<45} {result['output'][:30]}")
            counts["pass"] += 1
        else:
            err_last = result["error"].split("\n")[-1][:60]
            print(f"  [FAIL] {path:<45} {err_last}")
            counts["fail"] += 1

        results.append({"path": path, **result})

    if not args.dry_run:
        print(f"\n{'=' * 80}")
        print(
            f"Results: {counts['pass']} pass, {counts['fail']} fail, {counts['skip']} skip"
        )
        print(f"{'=' * 80}")

        if counts["fail"] > 0:
            print("\nFAILURES:")
            for r in results:
                if r["status"] == "fail":
                    print(f"  {r['path']}:")
                    for line in r.get("error", "").split("\n"):
                        print(f"    {line}")
                    print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nJSON results written to {args.json}")


if __name__ == "__main__":
    main()
