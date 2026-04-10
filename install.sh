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

# Claude Code — marketplace from GitHub, auto-updates independently
if command -v claude >/dev/null 2>&1
then
	claude plugin marketplace add "$org/$repo" 2>/dev/null || true
	claude plugin install "$product@LithosAI" 2>/dev/null || true
	python3 <<-EOF 2>/dev/null
		import json
		from pathlib import Path

		for p in [
		    Path.home() / '.claude/plugins/known_marketplaces.json',
		    Path.home() / '.claude/settings.json',
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

[ -n "$installed" ] && echo "Installed $product skill for: $installed"
[ -n "$skipped" ]   && echo "Skipped (not detected): $skipped"
[ -n "$installed" ] && echo "Done. Restart your coding agent to pick up the /$product skill."
