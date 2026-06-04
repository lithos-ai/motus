import json


def _chat_message(role, content=None, **extra):
    data = {"role": role, "content": content}
    data.update(extra)
    return data


def test_trace_from_task_meta_builds_turns_from_model_and_tool_spans():
    from motus.mars.record import trace_from_task_meta

    task_meta = {
        1: {
            "task_type": "agent_call",
            "agent_id": "CodingAgent",
            "object_id": 123,
            "start_us": 0,
            "end_us": 3000,
        },
        2: {
            "task_type": "model_call",
            "parent": 1,
            "start_us": 100,
            "end_us": 200,
            "model_input_meta": [
                _chat_message("system", "system prompt"),
                _chat_message("user", "fix the bug"),
            ],
            "model_output_meta": {
                "id": "completion-1",
                "model": "model",
                "content": "thinking",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "pytest -q"}),
                        },
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"completion_tokens": 5},
            },
        },
        3: {
            "task_type": "tool_call",
            "parent": 1,
            "start_us": 250,
            "end_us": 1_250,
            "tool_input_meta": {
                "name": "bash",
                "arguments": {"command": "pytest -q"},
            },
            "tool_output_meta": "failed",
        },
        4: {
            "task_type": "model_call",
            "parent": 1,
            "start_us": 1_500,
            "end_us": 1_600,
            "model_input_meta": [
                _chat_message("system", "system prompt"),
                _chat_message("user", "fix the bug"),
                _chat_message(
                    "assistant",
                    "thinking",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "pytest -q"}),
                            },
                        }
                    ],
                ),
                _chat_message(
                    "tool",
                    "failed",
                    tool_call_id="call-1",
                    name="bash",
                ),
            ],
            "model_output_meta": {
                "id": "completion-2",
                "model": "model",
                "content": "done",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"completion_tokens": 2},
            },
        },
    }

    trace = trace_from_task_meta(
        task_meta,
        trace_id="trace-1",
        agent_instance_id="agent-1",
    )

    assert trace.trace_id == "trace-1"
    assert trace.agent.agent_instance_id == "agent-1"
    assert trace.agent.agent_class_id == "CodingAgent"
    assert trace.system_prompt.text == "system prompt"
    assert len(trace.turns) == 2
    assert trace.turns[0].input_delta.kind == "user"
    assert trace.turns[0].input_delta.text == "fix the bug"
    assert trace.turns[0].output_tokens == 5
    assert trace.turns[0].tools[0].name == "bash"
    assert trace.turns[0].tools[0].args == {"command": "pytest -q"}
    assert trace.turns[0].tools[0].duration_ms == 1
    assert trace.turns[0].is_terminal is False
    assert trace.turns[1].input_delta.kind == "tool_result"
    assert trace.turns[1].input_delta.text == "failed"
    assert trace.turns[1].output_tokens == 2
    assert trace.turns[1].tools == []
    assert trace.turns[1].is_terminal is True


def test_trace_from_task_meta_uses_tool_span_when_model_output_has_no_tool_call_details():
    from motus.mars.record import trace_from_task_meta

    task_meta = {
        10: {
            "task_type": "agent_call",
            "agent_id": "CustomAgent",
            "start_us": 0,
        },
        11: {
            "task_type": "model_call",
            "parent": 10,
            "start_us": 10,
            "end_us": 20,
            "model_input_meta": [
                _chat_message("system", "system"),
                _chat_message("user", "input"),
            ],
            "model_output_meta": {
                "usage": {"completion_tokens": 1},
            },
            "chosen_tools": ["read_file"],
        },
        12: {
            "task_type": "tool_call",
            "parent": 10,
            "start_us": 30,
            "end_us": 530,
            "tool_input_meta": {
                "name": "read_file",
                "arguments": {"path": "README.md"},
            },
        },
    }

    trace = trace_from_task_meta(task_meta, trace_id="trace-2")

    assert trace.agent.agent_class_id == "CustomAgent"
    assert trace.agent.agent_instance_id == "trace-2"
    assert trace.turns[0].tools[0].name == "read_file"
    assert trace.turns[0].tools[0].args == {"path": "README.md"}
    assert trace.turns[0].tools[0].duration_ms == 0.5
