from pathlib import Path


def test_load_prompt_from_text_or_file(tmp_path):
    from motus.mars.coding_agent_harness import load_prompt

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from file", encoding="utf-8")

    assert load_prompt(prompt="inline", prompt_file=None) == "inline"
    assert load_prompt(prompt=None, prompt_file=prompt_file) == "from file"


def test_coding_agent_record_cli_builds_mars_client_and_agent(tmp_path, monkeypatch):
    from motus.mars import coding_agent_harness

    prompt_file = tmp_path / "prompt.txt"
    trace_path = tmp_path / "trace.json"
    result_path = tmp_path / "result.txt"
    prompt_file.write_text("fix tests", encoding="utf-8")
    clients = []
    agents = []
    recorded = []
    sandboxes = []

    class FakeClient:
        def __init__(self, **kwargs):
            clients.append(kwargs)

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            agents.append(self)

    class FakeLocalShell:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            sandboxes.append(self)

    async def fake_record_agent_run(agent, prompt, path, **kwargs):
        recorded.append(
            {
                "agent": agent,
                "prompt": prompt,
                "path": Path(path),
                "kwargs": kwargs,
            }
        )
        return "done", object()

    monkeypatch.setattr(coding_agent_harness, "MarsOpenAIChatClient", FakeClient)
    monkeypatch.setattr(coding_agent_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(coding_agent_harness, "LocalShell", FakeLocalShell)
    monkeypatch.setattr(coding_agent_harness, "record_agent_run", fake_record_agent_run)

    exit_code = coding_agent_harness.main(
        [
            "--base-url",
            "http://mars.local/v1",
            "--api-key",
            "EMPTY",
            "--model",
            "mars-model",
            "--prompt-file",
            str(prompt_file),
            "--trace-path",
            str(trace_path),
            "--result-path",
            str(result_path),
            "--project-root",
            str(tmp_path),
            "--agent-instance-id",
            "instance-1",
            "--agent-class-id",
            "coding-agent",
            "--trace-id",
            "trace-1",
            "--max-steps",
            "3",
            "--disable-web",
            "--disable-subagents",
        ]
    )

    assert exit_code == 0
    assert clients == [
        {
            "api_key": "EMPTY",
            "base_url": "http://mars.local/v1",
            "agent_instance_id": "instance-1",
            "agent_class_id": "coding-agent",
        }
    ]
    assert agents[0].kwargs["client"].__class__ is FakeClient
    assert agents[0].kwargs["sandbox"] is sandboxes[0]
    assert sandboxes[0].kwargs == {"cwd": str(tmp_path)}
    assert agents[0].kwargs["model_name"] == "mars-model"
    assert agents[0].kwargs["project_root"] == tmp_path
    assert agents[0].kwargs["max_steps"] == 3
    assert agents[0].kwargs["enable_web"] is False
    assert agents[0].kwargs["enable_subagents"] is False
    assert recorded == [
        {
            "agent": agents[0],
            "prompt": "fix tests",
            "path": trace_path,
            "kwargs": {
                "trace_id": "trace-1",
                "agent_instance_id": "instance-1",
                "agent_class_id": "coding-agent",
                "source": {},
            },
        }
    ]
    assert result_path.read_text(encoding="utf-8") == "done\n"
