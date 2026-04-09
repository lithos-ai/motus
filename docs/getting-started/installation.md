# Installation

## Prerequisites

- **Python 3.12+**
- **uv** (recommended) or pip
- An API key from at least one LLM provider (OpenAI, Anthropic, Google, or OpenRouter)

## Install with uv

```bash
# Clone the repository
git clone https://github.com/gpuOS-ai/motus.git
cd motus

# Install core dependencies
uv sync
```

To include optional SDK integrations:

=== "OpenAI Agents SDK"

    ```bash
    uv sync --extra openai-agents
    ```

=== "Anthropic SDK"

    The Anthropic SDK is included as a core dependency. Upgrade if needed:

    ```bash
    uv pip install 'anthropic>=0.49.0'
    ```

=== "Google ADK"

    ```bash
    uv sync --extra google-adk
    ```

=== "All extras"

    ```bash
    uv sync --all-extras
    ```

## Install with pip

```bash
git clone https://github.com/gpuOS-ai/motus.git
cd motus
pip install -e .
```

## Verify the installation

```bash
python -c "from motus.agent import ReActAgent; print('Motus is ready')"
```

## Set up API keys

Export your provider API keys as environment variables. At least one is required:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=...
```

## Next steps

Head to the [Quickstart](quickstart.md) to build your first agent.
