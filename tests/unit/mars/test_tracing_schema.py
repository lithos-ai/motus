import json
from pathlib import Path

import pytest

TRACE_PATH = Path(
    "/home/jjn/workspace/experiments/motustrace/phase4_fcfs_run2/traces/build-pov-ray__knWWE4h.json"
)


def test_loads_real_motustracing_file():
    from motus.mars.tracing import load_trace

    trace = load_trace(TRACE_PATH)

    assert trace.schema == "motustracing.agent_trace.v1"
    assert trace.trace_id == "build-pov-ray__knWWE4h"
    assert trace.agent.agent_instance_id == "build-pov-ray__knWWE4h"
    assert trace.agent.agent_class_id == "terminus_2"
    assert trace.system_prompt.text
    assert trace.turns[0].turn_index == 0
    assert trace.turns[0].input_delta.kind == "user"
    assert trace.turns[0].output_tokens == 465
    assert trace.turns[0].tools[0].name == "ls"
    assert trace.turns[0].tools[0].duration_ms == 196


def test_write_trace_round_trips_core_fields(tmp_path):
    from motus.mars.tracing import load_trace, write_trace

    trace = load_trace(TRACE_PATH)
    out = tmp_path / "trace.json"

    write_trace(trace, out)
    loaded = load_trace(out)

    assert loaded.trace_id == trace.trace_id
    assert loaded.agent == trace.agent
    assert loaded.system_prompt.token_count == trace.system_prompt.token_count
    assert len(loaded.turns) == len(trace.turns)
    assert loaded.turns[1].tools[0].args == trace.turns[1].tools[0].args


def test_invalid_turn_order_raises(tmp_path):
    from motus.mars.tracing import load_trace

    data = json.loads(TRACE_PATH.read_text())
    data["turns"][1]["turn_index"] = 9
    bad_trace = tmp_path / "bad.json"
    bad_trace.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="turn_index"):
        load_trace(bad_trace)


def test_tool_args_default_to_empty_dict(tmp_path):
    from motus.mars.tracing import load_trace

    data = json.loads(TRACE_PATH.read_text())
    data["turns"][0]["tools"][0].pop("args")
    path = tmp_path / "no_args.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    trace = load_trace(path)

    assert trace.turns[0].tools[0].args == {}

