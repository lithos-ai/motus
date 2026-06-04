"""Run a Motus CodingAgent against a Mars OpenAI-compatible backend.

This example delegates to the maintained harness:

    python examples/mars/coding_agent_mars.py \
      --base-url http://10.96.0.86:30001/v1 \
      --model /mnt/shared/models/MiniMax-M2.7 \
      --trace-id coding-agent-smoke \
      --trace-path /tmp/coding-agent-smoke.json \
      --project-root /tmp/project \
      --disable-web \
      --disable-subagents \
      --max-steps 2 \
      --prompt "Use the bash tool to run pwd, then answer with the path."
"""

from __future__ import annotations

from motus.mars.coding_agent_harness import main

if __name__ == "__main__":
    raise SystemExit(main())
