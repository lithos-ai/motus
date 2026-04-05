# Motus Full-Stack Agent Framework

## Introduction

<!-- TODO: What Motus is, who it's for, and why it exists -->

## Getting Started

### Installation

<!-- TODO: uv/pip install commands and prerequisites -->

### Configuration

<!-- TODO: Minimal config needed to run a first agent -->

## Key Concepts

<!-- TODO: How runtime, agent, models, tools, and memory relate to each other -->

## Running Example Agents

<!-- TODO: Concise guide to running each of our example agents -->

## Building an Agent

<!-- TODO: End-to-end example of defining and running a simple agent -->

## Contributing

### Development Setup

<!-- TODO: Clone, install, and verify the dev environment -->

### Pre-Commit Hooks

To maintain a unified code style across the repository and catch simple bugs before code review, Motus uses [pre-commit](https://pre-commit.com/).

Run the following commands in the project root to enable the Git hooks:

```bash
# Sync development dependencies (installs pre-commit via uv)
$ uv sync

# Install the git hooks
$ uv run pre-commit install
```

Once installed, you don't need to do anything special. Just commit your code as usual:

```bash
$ git commit -m "feat: add new feature"

trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
check for added large files..............................................Passed
detect private key.......................................................Passed
debug statements (python)................................................Passed
check for merge conflicts................................................Passed
check that scripts with shebangs are executable..........................Passed
dont commit to branch....................................................Passed
check that scripts with shebangs are executable..........................Passed
ruff format..............................................................Passed
ruff check...............................................................Passed
codespell................................................................Passed
```

### Running Tests

<!-- TODO: How to run the test suite and what to expect -->

### Code Style

<!-- TODO: Ruff config, formatting conventions, and naming rules -->

### Pull Request Process

<!-- TODO: Branch naming, review expectations, and merge policy -->

### Repository Structure

<!-- TODO: Annotated tree of src/motus/ and key directories -->

## Plugin for Claude Code, Codex, and Cursor

The `/motus` skill lets you build, configure, and deploy agents from inside your coding agent.

### Install the plugin

```sh
curl -fsSL https://www.lithosai.com/install.sh | sh
```

This installs the `/motus` skill into Claude Code, Codex, and Cursor.

### Usage

```
/motus                                              # guided agent creation
/motus I need a customer support agent with PII redaction
/motus deploy                                       # auto-detect and guided deploy
/motus deploy agent:my_agent                        # specify entry point
/motus deploy my-project agent:my_agent             # direct cloud deploy
```

### Alternative installation methods

#### Claude Code plugin marketplace

```sh
claude plugin marketplace add lithos-ai/motus && claude plugin install motus
```

Or from within Claude Code:

```
/plugin marketplace add lithos-ai/motus
/plugin install motus
```

#### Enable auto-updates

1. Open Claude Code and run `/plugin`
2. Go to **Marketplaces** tab
3. Select **motus-marketplace**
4. Choose **Enable auto-update**

#### npx skills (installs for all coding agents)

```sh
npx skills add lithos-ai/motus
```

### Team setup for Claude Code

Add to your project's `.claude/settings.json` for automatic availability:

```json
{
  "extraKnownMarketplaces": {
    "motus-marketplace": {
      "source": { "source": "github", "repo": "lithos-ai/motus" }
    }
  },
  "enabledPlugins": {
    "motus@motus-marketplace": true
  }
}
```
