# Plugin Packaging Design

## Requirements

1. **One-command install** on macOS and Linux: `curl -fsSL .../install.sh | sh`
2. **Four coding agents**: Claude Code, Codex, Cursor, Gemini CLI
3. **Automatic updates** where the agent supports it (Claude Code)
4. **Single-command updates** for everything else: `uv tool upgrade lithosai-motus`
5. **Plugin version = CLI version** (kept in sync manually)
6. **Agent-driven upgrades**: when the CLI prints an update-available message, the coding agent runs the upgrade command automatically

## Design

### Source of truth

Plugin files live at `src/motus/plugins/motus/` inside the Python package. They are bundled as `package-data` and installed to site-packages by `uv tool install`. The `plugins/motus` entry at the repo root is a **symlink** into the package source, so that GitHub-based marketplace discovery (Claude Code, Codex) can find the plugin using the relative path `./plugins/motus`.

### Repo layout

```
src/motus/plugins/motus/          <-- actual files (package data)
    .claude-plugin/plugin.json
    .codex-plugin/plugin.json
    .cursor-plugin/plugin.json
    skills/motus/SKILL.md ...

plugins/motus                     <-- symlink -> ../src/motus/plugins/motus

.claude-plugin/marketplace.json   <-- repo-root marketplace (source: "./plugins/motus")
.agents/plugins/marketplace.json  <-- repo-root marketplace (source: "./plugins/motus")
```

Both marketplace files use `"name": "LithosAI"` so that local and GitHub-based registrations overwrite rather than duplicate each other.

### Per-agent deployment (`motus setup`)

| Agent | Deployment method | Location |
|-------|-------------------|----------|
| Claude Code | `claude plugin marketplace add` from GitHub | `~/.claude/plugins/` (managed by Claude Code) |
| Codex | Personal marketplace pointing to package path | `~/.agents/plugins/marketplace.json` |
| Cursor | Symlink to package path | `~/.cursor/skills/motus` |
| Gemini | Symlink to package path | `~/.gemini/skills/motus` |

Claude Code gets its own managed copy via the GitHub marketplace with auto-update enabled. The other three agents point (via marketplace entry or symlink) to files inside the uv-managed package at `~/.local/share/uv/tools/lithosai-motus/.../site-packages/motus/plugins/motus/`.

### Update matrix

| Component | Update mechanism | Frequency |
|-----------|------------------|-----------|
| CLI | `uv tool upgrade lithosai-motus` | Manual (prompted by version check) |
| Claude Code plugin | Auto-update from GitHub | Automatic |
| Codex/Cursor/Gemini plugin | `uv tool upgrade` (same paths) | With CLI upgrade |

The CLI checks PyPI once every 24 hours and prints a message to stderr when a new version is available. SKILL.md instructs the coding agent to run the upgrade command when it sees this message, so updates happen organically during coding sessions.

### Version bumping

When releasing a new version, update these files:

1. `pyproject.toml` (`version = "X.Y.Z"`)
2. `src/motus/plugins/motus/.claude-plugin/plugin.json` (`"version": "X.Y.Z"`)
3. `src/motus/plugins/motus/.codex-plugin/plugin.json` (`"version": "X.Y.Z"`)
4. `src/motus/plugins/motus/.cursor-plugin/plugin.json` (`"version": "X.Y.Z"`)
