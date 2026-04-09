# Plugin Packaging Design

## Requirements

1. **One-command install** on macOS and Linux: `curl -fsSL .../install.sh | sh`
2. **Four coding agents**: Claude Code, Codex, Cursor, Gemini CLI
3. **Automatic updates** where the agent supports it (Claude Code)
4. **Self-updating skills**: for agents without auto-update, the skill checks its version against the latest GitHub release and prompts the agent to rerun the installer
5. **CLI versioned independently** from the plugin

## Design

### Source of truth

Plugin files live at `plugins/motus/` in the repo. They are downloaded from GitHub by `install.sh` and copied to each agent's skill directory. The CLI (`lithosai-motus`) is a separate Python package installed via `uv tool install`.

### Repo layout

```
plugins/motus/                        <-- the actual plugin
    .claude-plugin/plugin.json
    .codex-plugin/plugin.json
    skills/motus/
        SKILL.md
        REFERENCE.md, PATTERNS.md, EXAMPLES.md, ...
        gemini-extension.json         <-- Gemini manifest listing all context files

.claude-plugin/marketplace.json       <-- Claude Code marketplace (source: "./plugins/motus")
.cursor-plugin/plugin.json            <-- Cursor plugin manifest (skills: "./plugins/motus/skills/")
.agents/plugins/marketplace.json      <-- Codex marketplace (repo-level discovery)
```

All marketplace files use `"name": "LithosAI"`.

### Per-agent deployment (`install.sh`)

| Agent | Deployment method | Installed location |
|-------|-------------------|--------------------|
| Claude Code | `claude plugin marketplace add` from GitHub | `~/.claude/plugins/` (managed by Claude Code, auto-updates) |
| Codex | Copy skill directory from GitHub | `~/.codex/skills/motus` |
| Cursor | Copy skill directory from GitHub | `~/.cursor/skills/motus` |
| Gemini | Copy skill directory from GitHub | `~/.gemini/extensions/motus` |

Claude Code gets its own managed copy via the GitHub marketplace with auto-update enabled. The other three agents get skill files copied from a GitHub tarball. Full plugin installs are not used for local deployment because a plugin containing a skill with the same name causes duplicate registration.

Gemini CLI uses `gemini-extension.json` to discover context files (it does not read `SKILL.md` by default). The manifest lists all markdown files in the skill directory.

### Update matrix

| Component | Update mechanism | Frequency |
|-----------|------------------|-----------|
| CLI | `uv tool upgrade lithosai-motus` | Manual (prompted by PyPI version check) |
| Claude Code plugin | Auto-update from GitHub | Automatic |
| Codex/Cursor/Gemini skill | Skill checks version against GitHub release, prompts agent to rerun installer | On next agent session |

### Version bumping

Plugin and CLI versions are independent.

**Plugin files:**

1. `plugins/motus/.claude-plugin/plugin.json` (`"version"`)
2. `plugins/motus/.codex-plugin/plugin.json` (`"version"`)
3. `plugins/motus/skills/motus/gemini-extension.json` (`"version"`)
4. `.cursor-plugin/plugin.json` (`"version"`)

**Marketplace metadata (independent):**

5. `.claude-plugin/marketplace.json` (`metadata.version`)

**CLI:**

6. `pyproject.toml` (`version`)
7. `src/motus/serve/server.py` (FastAPI `version`)
