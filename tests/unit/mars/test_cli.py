import json
from dataclasses import asdict


def _trace_json(trace_id="trace-1"):
    return {
        "schema": "motustracing.agent_trace.v1",
        "trace_id": trace_id,
        "agent": {
            "agent_instance_id": f"{trace_id}-agent",
            "agent_class_id": "class-1",
        },
        "system_prompt": {"text": "system"},
        "turns": [
            {
                "turn_index": 0,
                "input_delta": {"kind": "user", "text": "input"},
                "output_tokens": 1,
                "tools": [],
                "is_terminal": True,
            }
        ],
    }


def test_collect_trace_paths_accepts_files_and_directories(tmp_path):
    from motus.mars.cli import collect_trace_paths

    trace_a = tmp_path / "a.json"
    trace_b = tmp_path / "nested" / "b.json"
    ignored = tmp_path / "nested" / "notes.txt"
    trace_b.parent.mkdir()
    trace_a.write_text(json.dumps(_trace_json("a")), encoding="utf-8")
    trace_b.write_text(json.dumps(_trace_json("b")), encoding="utf-8")
    ignored.write_text("ignore", encoding="utf-8")

    assert collect_trace_paths([tmp_path]) == [trace_a, trace_b]
    assert collect_trace_paths([trace_b]) == [trace_b]
    assert collect_trace_paths([tmp_path], limit=1) == [trace_a]


def test_cli_replays_traces_and_writes_summary(tmp_path, monkeypatch):
    from motus.mars import cli
    from motus.mars.replay import ReplaySummary, TraceReplayResult

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "trace.json"
    trace_path.write_text(json.dumps(_trace_json()), encoding="utf-8")
    output_dir = tmp_path / "out"
    created_clients = []
    created_runners = []

    class FakeClient:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)

    class FakeRunner:
        def __init__(self, *, client, model, concurrency):
            created_runners.append(
                {"client": client, "model": model, "concurrency": concurrency}
            )

        async def run_many(self, traces):
            assert traces == [trace_path]
            return ReplaySummary(
                total_traces=1,
                completed_traces=1,
                failed_traces=0,
                elapsed_seconds=0.25,
                results=[
                    TraceReplayResult(
                        trace_id="trace-1",
                        agent_instance_id="trace-1-agent",
                        agent_class_id="class-1",
                        turns_completed=1,
                    )
                ],
            )

    monkeypatch.setattr(cli, "MarsOpenAIChatClient", FakeClient)
    monkeypatch.setattr(cli, "TraceReplayRunner", FakeRunner)

    exit_code = cli.main(
        [
            str(trace_dir),
            "--base-url",
            "http://mars.local/v1",
            "--api-key",
            "EMPTY",
            "--model",
            "mars-model",
            "--concurrency",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert created_clients == [
        {"api_key": "EMPTY", "base_url": "http://mars.local/v1"}
    ]
    assert created_runners[0]["model"] == "mars-model"
    assert created_runners[0]["concurrency"] == 2
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == asdict(
        ReplaySummary(
            total_traces=1,
            completed_traces=1,
            failed_traces=0,
            elapsed_seconds=0.25,
            results=[
                TraceReplayResult(
                    trace_id="trace-1",
                    agent_instance_id="trace-1-agent",
                    agent_class_id="class-1",
                    turns_completed=1,
                )
            ],
        )
    )
    assert (output_dir / "requests.jsonl").read_text(encoding="utf-8") == ""
    assert (output_dir / "errors.jsonl").read_text(encoding="utf-8") == ""


def test_write_summary_writes_request_and_error_jsonl(tmp_path):
    from motus.mars.cli import write_summary
    from motus.mars.replay import (
        ReplayError,
        ReplaySummary,
        ReplayTurnResult,
        TraceReplayResult,
    )

    summary = ReplaySummary(
        total_traces=2,
        completed_traces=1,
        failed_traces=1,
        elapsed_seconds=1.5,
        results=[
            TraceReplayResult(
                trace_id="trace-1",
                agent_instance_id="agent-1",
                agent_class_id="class-1",
                turns_completed=1,
                turn_results=[
                    ReplayTurnResult(
                        trace_id="trace-1",
                        agent_instance_id="agent-1",
                        agent_class_id="class-1",
                        turn_index=0,
                        output_tokens_requested=3,
                        output_tokens_observed=3,
                        planned_tools=[
                            {
                                "name": "bash",
                                "args": {"command": "pwd"},
                                "duration_ms": 10,
                            }
                        ],
                        duration_ms_requested=10.0,
                        started_at=1.0,
                        ended_at=2.0,
                        finish_reason="length",
                        usage={"completion_tokens": 3},
                    )
                ],
            )
        ],
        errors=[
            ReplayError(
                error="trace-2 failed",
                trace_id="trace-2",
                trace_path="/tmp/trace-2.json",
            )
        ],
    )

    write_summary(summary, tmp_path)

    requests = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    errors = [
        json.loads(line)
        for line in (tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert requests == [
        {
            "agent_class_id": "class-1",
            "agent_instance_id": "agent-1",
            "duration_ms_requested": 10.0,
            "ended_at": 2.0,
            "finish_reason": "length",
            "output_tokens_observed": 3,
            "output_tokens_requested": 3,
            "planned_tools": [
                {
                    "args": {"command": "pwd"},
                    "duration_ms": 10,
                    "name": "bash",
                }
            ],
            "started_at": 1.0,
            "trace_id": "trace-1",
            "turn_index": 0,
            "usage": {"completion_tokens": 3},
        }
    ]
    assert errors == [
        {
            "error": "trace-2 failed",
            "trace_id": "trace-2",
            "trace_path": "/tmp/trace-2.json",
        }
    ]
