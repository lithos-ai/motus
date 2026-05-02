# ActusBot — Harbor benchmark agent

ActusBot is a multi-agent (main + executor / debugger / syntax_verifier / file_verifier sub-agents) implementation that runs against [Harbor](https://github.com/laude-institute/terminal-bench) tasks, built on the [Motus](https://github.com/lithos-ai/motus) agent framework.

## Layout

```
benchmark/harbor/
├── actus_bot/
│   ├── actus_bot.py          # ActusBot class (extends motus.agent.ReActAgent)
│   ├── tools.py              # Wraps harbor BaseEnvironment.exec into a sandbox_sh tool
│   ├── agent_prompts/        # System prompts per sub-agent
│   └── skills/               # SKILL.md files loaded on-demand by the executor
├── test_actus_bot.py         # ActusAgent — Harbor BaseAgent registration / runner
├── pyproject.toml
└── README.md
```

## Setup

From the repo root:

```bash
# motus itself, editable
pip install -e .

# this benchmark package
pip install -e benchmark/harbor

# harbor (terminal-bench) — install per its own instructions
pip install harbor-bench  # or follow upstream
```

Pick one backend and set the matching env vars in a `.env` file (loaded by `python-dotenv` at agent setup). Selection order in `setup()`: self-hosted > OpenRouter > Anthropic.

```bash
# Option A — local sglang / vllm (OpenAI-compatible endpoint)
SELF_HOSTED_BASE_URL=http://localhost:30000     # sglang default; vllm uses 8000
# optional:
SELF_HOSTED_API_KEY=...                         # only if server was started with --api-key
SELF_HOSTED_MODEL=/path/to/Llama-3.1-8B         # else: first model from /v1/models
SELF_HOSTED_ENGINE=sglang                       # sglang | vllm | auto (telemetry only)

# Option B — OpenRouter
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Option C — direct Anthropic
ANTHROPIC_API_KEY=...
```

Self-hosted launch examples:

```bash
# sglang
python -m sglang.launch_server --model-path /path/to/model --port 30000

# vllm
python -m vllm.entrypoints.openai.api_server --model /path/to/model --port 8000
```

## Run

The agent is registered as `ActusAgent` (name: `actusbot`) in `test_actus_bot.py`. Point Harbor at it via its run config and execute the task suite per Harbor's docs.

## Notes on the Motus port

This benchmark was originally built against the `lithos` library. The port to Motus replaces:

| lithos                                 | motus                                    |
| -------------------------------------- | ---------------------------------------- |
| `lithos.agent.ReActAgent`              | `motus.agent.ReActAgent`                 |
| `lithos.tools.FunctionTool`            | `motus.tools.FunctionTool`               |
| `lithos.models.AnthropicChatClient`    | `motus.models.AnthropicChatClient`       |
| `lithos.models.OpenAIChatClient`       | `motus.models.OpenAIChatClient`          |
| `ReActAgent(enable_memory=False)`      | `ReActAgent(memory=BasicMemory(enable_memory_tools=False))` |
| `ReActAgent(skill_dir=...)`            | adds `make_skill_tool(skills_dir)` into the tool dict |
