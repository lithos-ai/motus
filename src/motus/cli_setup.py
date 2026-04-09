"""motus setup — deploy plugins for installed coding agents."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent / "plugins" / "motus"
SKILL_DIR = PLUGIN_DIR / "skills" / "motus"

_MARKETPLACE_NAME = "LithosAI"
_PLUGIN_NAME = "motus"
_GITHUB_REPO = "lithos-ai/motus"


# -- Agent detection ----------------------------------------------------------


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _has_claude() -> bool:
    return _has_command("claude")


def _has_codex() -> bool:
    return _has_command("codex") or (Path.home() / ".codex").is_dir()


def _has_cursor() -> bool:
    return (Path.home() / "Library" / "Application Support" / "Cursor").is_dir() or (
        Path.home() / ".config" / "Cursor"
    ).is_dir()


def _has_gemini() -> bool:
    return _has_command("gemini") or (Path.home() / ".gemini").is_dir()


# -- Claude Code ---------------------------------------------------------------


def _setup_claude() -> bool:
    """Register the plugin with Claude Code via the GitHub marketplace."""
    try:
        subprocess.run(
            ["claude", "plugin", "marketplace", "add", _GITHUB_REPO],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        # Marketplace may already be registered — continue to install
        pass

    try:
        subprocess.run(
            ["claude", "plugin", "install", f"{_PLUGIN_NAME}@{_MARKETPLACE_NAME}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # "already installed" is fine
        if "already installed" not in (e.stderr or ""):
            print(
                f"  Warning: claude plugin install failed: {e.stderr}", file=sys.stderr
            )

    # Enable auto-update on the marketplace entry that `claude plugin marketplace add`
    # created.  Only modify existing entries — never create stub entries.
    _claude_enable_auto_update()

    # Clean up stale skill copies
    stale = Path.home() / ".claude" / "skills" / _PLUGIN_NAME
    if stale.is_dir() and not stale.is_symlink():
        shutil.rmtree(stale)

    return True


def _claude_enable_auto_update() -> None:
    """Enable autoUpdate on existing LithosAI marketplace entries.

    Only modifies entries that already exist (created by ``claude plugin
    marketplace add``).  Never creates stub entries — that would corrupt
    the file with incomplete data.
    """
    known_path = Path.home() / ".claude" / "plugins" / "known_marketplaces.json"
    try:
        if not known_path.exists():
            return
        known = json.loads(known_path.read_text())
        if _MARKETPLACE_NAME in known:
            known[_MARKETPLACE_NAME]["autoUpdate"] = True
            known_path.write_text(json.dumps(known, indent=2) + "\n")
    except Exception:
        pass


# -- Codex ---------------------------------------------------------------------


def _setup_codex() -> bool:
    """Register the plugin with Codex via a personal marketplace."""
    marketplace_path = Path.home() / ".agents" / "plugins" / "marketplace.json"

    try:
        existing = (
            json.loads(marketplace_path.read_text())
            if marketplace_path.exists()
            else {}
        )
    except Exception:
        existing = {}

    if "name" not in existing:
        existing["name"] = _MARKETPLACE_NAME
    if "interface" not in existing:
        existing["interface"] = {"displayName": _MARKETPLACE_NAME}

    plugins = existing.setdefault("plugins", [])

    # Remove existing entry for this plugin
    plugins[:] = [p for p in plugins if p.get("name") != _PLUGIN_NAME]

    plugins.append(
        {
            "name": _PLUGIN_NAME,
            "source": {
                "source": "local",
                "path": str(PLUGIN_DIR),
            },
            "policy": {
                "installation": "INSTALLED_BY_DEFAULT",
            },
            "category": "Development",
        }
    )

    marketplace_path.parent.mkdir(parents=True, exist_ok=True)
    marketplace_path.write_text(json.dumps(existing, indent=2) + "\n")
    return True


# -- Cursor --------------------------------------------------------------------


def _setup_cursor() -> bool:
    """Symlink the skill into Cursor's skill directory."""
    target = Path.home() / ".cursor" / "skills" / _PLUGIN_NAME
    return _create_skill_symlink(target)


# -- Gemini --------------------------------------------------------------------


def _setup_gemini() -> bool:
    """Symlink the skill into Gemini CLI's skill directory."""
    target = Path.home() / ".gemini" / "skills" / _PLUGIN_NAME
    return _create_skill_symlink(target)


# -- Helpers -------------------------------------------------------------------


def _create_skill_symlink(target: Path) -> bool:
    """Create a symlink from target to SKILL_DIR, replacing any existing one."""
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)

    os.symlink(SKILL_DIR, target)
    return True


# -- CLI entry point -----------------------------------------------------------


_AGENTS = [
    ("Claude Code", _has_claude, _setup_claude),
    ("Codex", _has_codex, _setup_codex),
    ("Cursor", _has_cursor, _setup_cursor),
    ("Gemini", _has_gemini, _setup_gemini),
]


def _setup_handler(args):
    if not PLUGIN_DIR.is_dir():
        print(
            "Error: plugin files not found. Reinstall motus: uv tool install lithosai-motus",
            file=sys.stderr,
        )
        sys.exit(1)

    installed = []
    skipped = []

    for name, detect, setup in _AGENTS:
        if detect():
            try:
                setup()
                installed.append(name)
            except Exception as e:
                print(f"  Warning: {name} setup failed: {e}", file=sys.stderr)
                skipped.append(name)
        else:
            skipped.append(name)

    if installed:
        print(f"Installed motus plugin for: {', '.join(installed)}")
    if skipped:
        print(f"Skipped (not detected): {', '.join(skipped)}")

    if installed:
        print("Done. Restart your coding agent to pick up the /motus skill.")


def register_cli(subparsers):
    """Register the 'setup' command with the top-level CLI."""
    setup_parser = subparsers.add_parser(
        "setup",
        help="deploy motus plugins for installed coding agents",
    )
    setup_parser.set_defaults(func=_setup_handler)
