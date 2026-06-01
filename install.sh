#!/bin/sh
# Install the Motus CLI and deploy skills for detected coding agents.
# Usage: curl -fsSL https://raw.githubusercontent.com/lithos-ai/motus/main/install.sh | sh
set -eu

org=lithos-ai
product=motus
repo=$product

echo "Installing $product..." >&2

# -- Install CLI ---------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1
then
	echo "Installing uv..." >&2
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
fi

if uv tool list 2>/dev/null | grep -q lithosai-motus
then
	uv tool upgrade lithosai-motus
else
	uv tool install lithosai-motus
fi

# -- Download skill files from latest GitHub release ---------------------------

tmp=${TMPDIR:-/tmp}/$product.$$
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp"

tag=$(curl -fsSL "https://api.github.com/repos/$org/$repo/releases/latest" | \
	grep '"tag_name"' | sed 's/.*: *"//;s/".*//')

if [ -z "$tag" ]
then
	echo "Error: could not determine latest release" >&2
	exit 1
fi

curl -fsSL "https://github.com/$org/$repo/archive/refs/tags/$tag.tar.gz" | tar xz -C "$tmp"
skill_src="$tmp/$repo-${tag#v}/plugins/$product/skills/$product"

if [ ! -d "$skill_src" ]
then
	echo "Error: skill not found in release $tag" >&2
	exit 1
fi

# -- Deploy skills for detected agents ----------------------------------------

installed=""
skipped=""

add_installed() { installed="${installed:+$installed, }$1"; }
add_skipped()   { skipped="${skipped:+$skipped, }$1"; }

# Claude Code: Vercel's skills package only checks whether the user has a
# `.claude` directory. Honor $CLAUDE_CONFIG_DIR for users who relocated it.
claude_home=${CLAUDE_CONFIG_DIR-$HOME/.claude}
if [ -d "$claude_home" ]
then
	# Resolve a `claude` binary: prefer the one on PATH, else the binary
	# bundled inside the VS Code / VS Code Insiders / Cursor extension at
	# resources/native-binary/claude. Walk those editor dirs in priority
	# order and pick the first installed extension's binary — gating on
	# `-x` so an editor with no Claude Code extension installed doesn't
	# leak its unexpanded glob pattern into `claude_cmd`.
	if command -v claude >/dev/null 2>&1
	then
		claude_cmd=claude
	else
		claude_cmd=`for editor in vscode vscode-insiders cursor
		do
			for bin in "$HOME"/.$editor/extensions/anthropic.claude-code-*/resources/native-binary/claude
			do
				[ -x "$bin" ] && echo "$bin"
			done
		done | awk '{print; exit}'`
	fi
	"${claude_cmd?}" plugin marketplace add "$org/$repo"
	"$claude_cmd" plugin install "$product@LithosAI"
	python3 <<-EOF 2>/dev/null
		import json
		from pathlib import Path

		claude_home = Path("$claude_home")
		for p in [
		    claude_home / 'plugins/known_marketplaces.json',
		    claude_home / 'settings.json',
		]:
		    if not p.exists(): continue
		    d = json.loads(p.read_text())
		    # settings.json nests under extraKnownMarketplaces
		    m = d.get('extraKnownMarketplaces', d)
		    if 'LithosAI' in m:
		        m['LithosAI']['autoUpdate'] = True
		        p.write_text(json.dumps(d, indent=2) + '\n')
	EOF
	add_installed "Claude Code"
else
	add_skipped "Claude Code"
fi

# Codex
if command -v codex >/dev/null 2>&1 || [ -d "$HOME/.codex" ]
then
	mkdir -p "$HOME/.codex/skills"
	rm -rf "$HOME/.codex/skills/$product"
	cp -R "$skill_src" "$HOME/.codex/skills/$product"
	add_installed "Codex"
else
	add_skipped "Codex"
fi

# Cursor
if [ -d "$HOME/Library/Application Support/Cursor" ] || \
    [ -d "$HOME/.config/Cursor" ]
then
	mkdir -p "$HOME/.cursor/skills"
	rm -rf "$HOME/.cursor/skills/$product"
	cp -R "$skill_src" "$HOME/.cursor/skills/$product"
	add_installed "Cursor"
else
	add_skipped "Cursor"
fi

# Gemini
if command -v gemini >/dev/null 2>&1 || [ -d "$HOME/.gemini" ]
then
	mkdir -p "$HOME/.gemini/extensions"
	rm -rf "$HOME/.gemini/extensions/$product"
	cp -R "$skill_src" "$HOME/.gemini/extensions/$product"
	add_installed "Gemini"
else
	add_skipped "Gemini"
fi

# -- Report --------------------------------------------------------------------

if [ -n "$installed" ]
then
	echo "Installed $product skill for: $installed"
fi
if [ -n "$skipped" ]
then
	echo "Skipped (not detected): $skipped"
fi
if [ -n "$installed" ]
then
	echo "Done. Restart your coding agent to pick up the /$product skill."
fi
