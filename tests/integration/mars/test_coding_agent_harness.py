import argparse
import json
import os
from pathlib import Path

import pytest

from motus.mars.coding_agent_harness import run_recording
from motus.mars.tracing import load_trace


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for Mars CodingAgent integration test")
    return value


def _events_for_agent(events_dir: Path, agent_instance_id: str) -> list[dict]:
    if not events_dir.exists():
        pytest.skip(f"Mars events dir does not exist: {events_dir}")

    events = []
    for path in sorted(events_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                if event.get("agent_instance_id") == agent_instance_id:
                    events.append(event)
    return events


@pytest.mark.integration
@pytest.mark.asyncio
async def test_coding_agent_records_trace_and_mars_parses_tool(tmp_path):
    base_url = _required_env("MARS_BASE_URL")
    model = _required_env("MARS_MODEL")
    events_dir = Path(_required_env("MARS_EVENTS_DIR"))

    project_root = tmp_path / "project"
    project_root.mkdir()
    trace_path = tmp_path / "trace.json"
    agent_instance_id = "coding-agent-parser-smoke"

    result, _trace = await run_recording(
        argparse.Namespace(
            model=model,
            trace_path=trace_path,
            prompt="Use the bash tool to run pwd, then answer with the directory path.",
            prompt_file=None,
            base_url=base_url,
            api_key=os.environ.get("MARS_API_KEY") or "EMPTY",
            project_root=project_root,
            trace_id=agent_instance_id,
            agent_instance_id=agent_instance_id,
            agent_class_id="motus-coding-agent",
            max_steps=2,
            result_path=None,
            disable_web=True,
            disable_subagents=True,
        )
    )

    trace = load_trace(trace_path)
    assert result
    assert trace.trace_id == agent_instance_id
    assert trace.turns
    assert any(tool.name == "bash" for turn in trace.turns for tool in turn.tools)

    events = _events_for_agent(events_dir, agent_instance_id)
    assert any(
        event.get("event") == "tool_parsed"
        and "bash" in event.get("tool_names", [])
        for event in events
    )
