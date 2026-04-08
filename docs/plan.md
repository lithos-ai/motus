# Documentation Content Plan

> **Rule**: Each writing session handles at most 3 pages, then updates the checklist below.

---

## Writing Principles

### Voice & Tone
- Use **second person** ("you") in how-to and reference pages. Use "we" only in tutorials (quickstart).
- **Active voice, present tense**: "The function returns" not "A list will be returned."
- Sound like a knowledgeable colleague — clear, direct, not academic.

### Structure Per Page
- **Lead with what, not why**: First sentence tells the reader what this page covers.
- **Code before prose**: Show a minimal working example within the first screen, then explain.
- **One job per page**: A page either teaches (tutorial), instructs (how-to), describes (reference), or explains (concept). Never mix.
- **Scannable**: Use headings, tables, and short paragraphs. No wall-of-text. A reader skimming headings alone should grasp the page structure.
- **Conditions before instructions**: "To create an agent, call `Agent()`" not the reverse.

### Code Examples
- Every code block must be **copy-pasteable and runnable** — no pseudocode, no `...` placeholders.
- Include imports. Show the output where helpful.
- Keep examples minimal: demonstrate one concept per block.

### What NOT to Do
- Do not restate the same information in different words to fill space.
- Do not document internal implementation details in user-facing pages.
- Do not add commentary like "This is very powerful" or "As you can see." Let the code speak.
- Do not mix tutorial content into reference pages or vice versa.

---

## Dependency Map

Some pages share context and are best written together or sequentially:

```
getting-started/installation ──→ getting-started/quickstart ──→ getting-started/configuration
       (prereqs)                     (first agent)                  (env vars, API keys)

index.md ← draws its hero example from quickstart

user-guide/overview ──→ branches into all other user-guide/* pages
       (architecture diagram, module relationships)

user-guide/agents ←──→ user-guide/tools       (agents use tools)
user-guide/agents ←──→ user-guide/models      (agents use model clients)
user-guide/agents ←──→ user-guide/memory      (agents use memory)

user-guide/mcp-integration ← depends on user-guide/tools (MCP is a tool type)
user-guide/guardrails ← depends on user-guide/agents + user-guide/tools
user-guide/tracing ← depends on user-guide/runtime (tracing hooks into the task graph)
user-guide/serving ──→ user-guide/deployment   (serve locally, then deploy)

contributing/development-setup ──→ contributing/code-style ──→ contributing/testing ──→ contributing/pull-requests
```

---

## Writing Order & Checklist

Ordered by dependency and user journey. Each batch is 1–3 pages that make sense together.

### Phase 1: Entry Points (what every new user reads first)

- [x] **Batch 1**: `index.md` — Landing page with one-sentence description + 10-line hero code example + links to quickstart
- [x] **Batch 2**: `getting-started/installation.md` + `getting-started/quickstart.md` — Prerequisites, install commands, first working agent in < 5 minutes
- [x] **Batch 3**: `getting-started/configuration.md` — Environment variables, API keys, model provider setup

### Phase 2: Architecture Foundation

- [x] **Batch 4**: `user-guide/overview.md` — Architecture diagram (Mermaid), how runtime/agent/models/tools/memory relate, reading guide for the rest of the docs

### Phase 3: Core Modules (the "what you build with" layer)

- [x] **Batch 5**: `user-guide/agents.md` + `user-guide/models.md` — ReActAgent lifecycle, model client abstraction, provider switching
- [x] **Batch 6**: `user-guide/tools.md` + `user-guide/mcp-integration.md` — FunctionTool, @tool decorator, Tool protocol, MCP sessions and transports
- [x] **Batch 7**: `user-guide/memory.md` — BasicMemory → CompactionMemory progression, when to use which

### Phase 4: Advanced Features

- [x] **Batch 8**: `user-guide/runtime.md` + `user-guide/tracing.md` — @agent_task, AgentFuture, task graph, TraceManager, collection levels
- [x] **Batch 9**: `user-guide/guardrails.md` — Input/output guardrails on agents and tools, structured output guardrails
- [x] **Batch 10**: `user-guide/serving.md` + `user-guide/deployment.md` — AgentServer, sessions, FastAPI endpoints, motus deploy

### Phase 5: CLI & Integrations

- [x] **Batch 11**: `user-guide/cli.md` — motus serve, motus deploy, pluggable command system
- [x] **Batch 12**: `integrations/openai-agents.md` + `integrations/claude-agent.md` — SDK bridges, when to use which

### Phase 6: Contributor Docs

- [x] **Batch 13**: `contributing/development-setup.md` + `contributing/code-style.md` — Clone, install, pre-commit, ruff rules, naming conventions
- [x] **Batch 14**: `contributing/testing.md` + `contributing/pull-requests.md` — Test markers, VCR system, PR template, CI pipeline

### Phase 7: Metadata

- [ ] **Batch 15**: `about/changelog.md` + `about/license.md` — Version history skeleton, license text

---

## Content Sizing Guide

| Page Type | Target Length | Sections |
|-----------|-------------|----------|
| Landing page (`index.md`) | 40–80 lines | Hero example, 3–4 feature bullets, quick links |
| Getting Started pages | 80–150 lines | Step-by-step with code blocks at each step |
| User Guide pages | 100–250 lines | Intro → minimal example → detailed usage → configuration table → see also |
| Integration pages | 80–150 lines | When to use → install → example → API differences |
| Contributing pages | 80–150 lines | Straightforward instructions, no theory |
| About pages | 20–50 lines | Factual, minimal |

---

## Progress Log

| Date | Batch | Pages Written | Notes |
|------|-------|---------------|-------|
| 2026-03-16 | 1 | `index.md` | Landing page: hero example, feature list, card links, learn-more table |
| 2026-03-16 | 2 | `installation.md`, `quickstart.md` | Install via uv/pip, tabbed provider examples, 5-step first agent tutorial |
| 2026-04-02 | 3 | `configuration.md` | Env vars, API keys, motus.toml, .env pattern |
| 2026-04-02 | 4 | `overview.md` | Architecture diagram (Mermaid), module table, reading guide |
| 2026-04-02 | 5 | `agents.md`, `models.md` | ReActAgent lifecycle + params, 4-provider clients, CachePolicy |
| 2026-04-02 | 6 | `tools.md` (moved from root), `mcp-integration.md` | Moved existing tools.md; MCP patterns, transports, filtering |
| 2026-04-02 | 7 | `memory.md` | Three strategies, CompactionMemory config, session restore |
| 2026-04-02 | 8 | `runtime.md`, `tracing.md` | @agent_task, AgentFuture, resolve(), TraceManager, hooks |
| 2026-04-02 | 9 | `guardrails.md` | Agent + tool guardrails, structured output, chaining |
| 2026-04-02 | 10 | `serving.md`, `deployment.md` | serve2 endpoints, motus deploy, secrets |
| 2026-04-02 | 11 | `cli.md` | Command reference tables for serve + deploy |
| 2026-04-02 | 12 | `openai-agents.md`, `claude-agent.md` | SDK bridges, tracing, deployment |
| 2026-04-02 | 13 | `development-setup.md`, `code-style.md` | Dev env, pre-commit, ruff, naming, docstrings |
| 2026-04-02 | 14 | `testing.md`, `pull-requests.md` | 3-tier testing, VCR cassettes, PR process, code owners |
