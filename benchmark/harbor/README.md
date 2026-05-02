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

Set credentials in a `.env` file (loaded by `python-dotenv` at agent setup):

```bash
# Either OpenRouter
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Or direct Anthropic
ANTHROPIC_API_KEY=...
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
