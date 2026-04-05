#!/bin/sh
set -eu

# Motus skill installer for Claude Code, Codex, and Cursor.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/lithos-ai/motus/main/install.sh | sh

org=lithos-ai
product=motus
repo=$product

echo "Installing $product skill..." >&2

tmp=${TMPDIR:-/tmp}/$product.$$
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp"

curl -fsSL https://github.com/$org/$repo/archive/main.tar.gz | tar xz -C "$tmp"
source=$tmp/$repo-main/plugins/$product/skills/$product

if [ ! -d "$source" ]
then
	echo "Error: skill not found in archive" >&2
	exit 1
fi

for target in .claude .codex
do
	if [ -d "$HOME/$target" ]
	then
		mkdir -p "$HOME/$target/skills"
		rm -rf "$HOME/$target/skills/$product"
		cp -R "$source" "$HOME/$target/skills"
	fi
done

# Cursor stores app data outside ~/.cursor, so check for it separately
if [ -d "$HOME/Library/Application Support/Cursor" ] || [ -d "$HOME/.config/Cursor" ]
then
	mkdir -p "$HOME/.cursor/skills"
	rm -rf "$HOME/.cursor/skills/$product"
	cp -R "$source" "$HOME/.cursor/skills"
fi

echo "Installed for: Claude Code, Codex, Cursor" >&2
echo "Done. The /$product skill is now available — restart your agent to pick it up." >&2
