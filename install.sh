#!/bin/sh
# Install the Motus CLI and deploy plugins for detected coding agents.
# Usage: curl -fsSL https://raw.githubusercontent.com/lithos-ai/motus/main/install.sh | sh
set -eu

echo "Installing motus..." >&2

# 1. Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..." >&2
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Install (or upgrade) the CLI — uv auto-downloads Python 3.12+ if needed
if uv tool list 2>/dev/null | grep -q lithosai-motus; then
    uv tool upgrade lithosai-motus
else
    uv tool install lithosai-motus
fi

# 3. Deploy plugins for detected coding agents
motus setup
