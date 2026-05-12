"""End-to-end tests for install.sh, exercised against real apps.

Each test runs install.sh inside a Linux container that has the real
`claude` CLI (via `npm i -g @anthropic-ai/claude-code`) and real VS
Code and Cursor installed. The bundled-binary tests install the
Claude Code extension via each editor's actual CLI (`code
--install-extension` / `cursor --install-extension`), so the
extension's directory naming and the bundled binary's exact location
are verified end-to-end — not hard-coded fixtures.

Assertions hit the resulting filesystem state — `installed_plugins.json`
must contain `motus@LithosAI`, the marketplace must be cloned, etc. —
so anything wrong in either install.sh's own logic *or* the claude
commands it invokes will surface.

`uv` is shimmed (no PyPI installs from tests). `claude`, VS Code, and
Cursor are real; the image rewrites GitHub SSH URLs to HTTPS so the
marketplace clone works without SSH keys. Requires Docker. Skipped
otherwise.

A session-scoped fixture builds the test image once per pytest
session — first build is slow (~3-5 min, dominated by the Cursor .deb
download and its GUI deps), every run after that hits the Docker
layer cache. Each test takes roughly 30-60 s.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest


def _docker_tests_enabled() -> bool:
    # Opt-in: these tests pull a multi-hundred-MB image, run real network
    # downloads (npm, apt, Cursor's CDN, GitHub clone), and take ~30 s
    # each. They should never fire from a bare `pytest`.
    if not os.environ.get("RUN_DOCKER_TESTS"):
        return False
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=5
            ).returncode
            == 0
        )
    except (subprocess.TimeoutExpired, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_tests_enabled(),
    reason="install.sh tests require RUN_DOCKER_TESTS=1 and a running Docker daemon",
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILE = dedent("""\
    FROM ubuntu:24.04
    RUN apt-get update -qq \\
        && apt-get install -qq -y --no-install-recommends \\
            curl python3 ca-certificates nodejs npm git \\
            wget gpg apt-transport-https \\
        && rm -rf /var/lib/apt/lists/*

    # Real Claude Code CLI (provides /usr/local/bin/claude)
    RUN npm install -g @anthropic-ai/claude-code

    # Real VS Code via Microsoft's apt repo
    RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc \\
            | gpg --dearmor > /usr/share/keyrings/packages.microsoft.gpg \\
        && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \\
            > /etc/apt/sources.list.d/vscode.list \\
        && apt-get update -qq \\
        && apt-get install -qq -y --no-install-recommends code \\
        && rm -rf /var/lib/apt/lists/*

    # Real Cursor via direct .deb download (no apt repo). Architecture
    # is auto-detected so the image builds the same way on amd64 / arm64.
    RUN arch=$(dpkg --print-architecture) \\
        && case "$arch" in \\
               amd64) platform=linux-x64 ;; \\
               arm64) platform=linux-arm64 ;; \\
               *) echo "unsupported arch: $arch" >&2; exit 1 ;; \\
           esac \\
        && deb_url=$(curl -fsSL "https://www.cursor.com/api/download?platform=$platform&releaseTrack=stable" \\
                     | python3 -c 'import json,sys;print(json.load(sys.stdin)["debUrl"])') \\
        && curl -fsSL "$deb_url" -o /tmp/cursor.deb \\
        && apt-get update -qq \\
        && apt-get install -qq -y --no-install-recommends /tmp/cursor.deb \\
        && rm /tmp/cursor.deb \\
        && rm -rf /var/lib/apt/lists/*

    # claude's `plugin marketplace add` prefers SSH; rewrite to HTTPS so
    # the test container doesn't need GitHub SSH keys.
    RUN git config --system url."https://github.com/".insteadOf "git@github.com:"
""")


@pytest.fixture(scope="session")
def test_image() -> str:
    """Build the base test image once per pytest session."""
    image = "motus-install-test:latest"
    subprocess.run(
        ["docker", "build", "-t", image, "-"],
        input=DOCKERFILE,
        text=True,
        check=True,
        capture_output=True,
    )
    return image


# A fake uv on PATH so we don't actually install lithosai-motus from PyPI.
# Logs invocations to $HOME/uv.log. `UV_TOOL_LIST_OUTPUT` lets a test
# pretend lithosai-motus is already installed (triggers the upgrade branch).
FAKE_UV = dedent("""\
    cat > /usr/local/bin/uv <<'EOSH'
    #!/bin/sh
    printf '%s\\n' "$*" >> "$HOME/uv.log"
    [ "$1 $2" = "tool list" ] && printf '%s' "${UV_TOOL_LIST_OUTPUT-}"
    exit 0
    EOSH
    chmod +x /usr/local/bin/uv
""")


def _run(
    image: str, home: Path, *, setup: str = "", env: str = ""
) -> subprocess.CompletedProcess:
    """Run install.sh inside `image` with `home` bind-mounted as HOME."""
    script = dedent(f"""\
        set -eu
        export HOME=/tmp/h
        cd "$HOME"
        {FAKE_UV}
        {setup}
        {env} sh /repo/install.sh
    """)
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:/repo:ro",
            "-v",
            f"{home}:/tmp/h",
            image,
            "bash",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _assert_motus_plugin_installed(home: Path) -> None:
    """Verify real claude cloned the marketplace and registered the plugin."""
    installed = json.loads(
        (home / ".claude/plugins/installed_plugins.json").read_text()
    )
    assert "motus@LithosAI" in installed["plugins"], installed
    marketplace = home / ".claude/plugins/marketplaces/LithosAI"
    assert (
        marketplace / "plugins" / "motus" / ".claude-plugin" / "plugin.json"
    ).exists()


def test_no_agents_skipped(test_image, tmp_path):
    """No agent dirs + no `claude` on PATH → all skipped, motus install attempted."""
    home = tmp_path / "h"
    home.mkdir()
    r = _run(test_image, home, setup="rm /usr/local/bin/claude")
    assert r.returncode == 0, r.stderr
    assert "Skipped (not detected): Claude Code, Codex, Cursor, Gemini" in r.stdout
    assert "tool install lithosai-motus" in (home / "uv.log").read_text()


def test_uv_upgrade_branch(test_image, tmp_path):
    """uv reports lithosai-motus installed → upgrade, not install."""
    home = tmp_path / "h"
    home.mkdir()
    r = _run(
        test_image,
        home,
        setup="rm /usr/local/bin/claude",
        env="UV_TOOL_LIST_OUTPUT=lithosai-motus",
    )
    assert r.returncode == 0, r.stderr
    uv_log = (home / "uv.log").read_text()
    assert "tool upgrade lithosai-motus" in uv_log
    assert "tool install lithosai-motus" not in uv_log


def test_claude_code_via_path(test_image, tmp_path):
    """`~/.claude` exists + claude on PATH → marketplace cloned + plugin installed."""
    home = tmp_path / "h"
    home.mkdir()
    r = _run(test_image, home, setup="mkdir -p $HOME/.claude")
    assert r.returncode == 0, r.stderr
    assert "Installed motus skill for: Claude Code" in r.stdout
    _assert_motus_plugin_installed(home)


def test_claude_code_via_bundled_vscode(test_image, tmp_path):
    """Real VS Code + Claude Code extension installed → script uses the
    bundled binary from `~/.vscode/extensions/anthropic.claude-code-*`."""
    home = tmp_path / "h"
    home.mkdir()
    r = _run(
        test_image,
        home,
        setup=dedent("""\
        mkdir -p $HOME/.claude
        # Real VS Code refuses to run as root without these flags;
        # `--install-extension` itself is headless.
        code --no-sandbox --user-data-dir=$HOME/.code-data \\
             --install-extension anthropic.claude-code 2>&1 | tail -1
        rm /usr/local/bin/claude
    """),
    )
    assert r.returncode == 0, r.stderr
    _assert_motus_plugin_installed(home)
    # Verify the extension landed where install.sh's glob expects.
    assert list(
        home.glob(
            ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
        )
    )


def test_claude_code_via_bundled_cursor(test_image, tmp_path):
    """Real Cursor + Claude Code extension installed → script uses the
    bundled binary from `~/.cursor/extensions/anthropic.claude-code-*`.
    Also exercises the for-loop's existence-gating since the `vscode`
    and `vscode-insiders` globs match nothing in this scenario."""
    home = tmp_path / "h"
    home.mkdir()
    r = _run(
        test_image,
        home,
        setup=dedent("""\
        mkdir -p $HOME/.claude
        cursor --no-sandbox --user-data-dir=$HOME/.cursor-data \\
               --install-extension anthropic.claude-code 2>&1 | tail -1
        rm /usr/local/bin/claude
    """),
    )
    assert r.returncode == 0, r.stderr
    _assert_motus_plugin_installed(home)
    assert list(
        home.glob(
            ".cursor/extensions/anthropic.claude-code-*/resources/native-binary/claude"
        )
    )


@pytest.mark.parametrize(
    "agent,setup,skill_path",
    [
        ("Codex", "mkdir -p $HOME/.codex", ".codex/skills/motus/SKILL.md"),
        ("Cursor", "mkdir -p $HOME/.config/Cursor", ".cursor/skills/motus/SKILL.md"),
        ("Gemini", "mkdir -p $HOME/.gemini", ".gemini/extensions/motus/SKILL.md"),
    ],
)
def test_other_agent_detected(test_image, tmp_path, agent, setup, skill_path):
    """Codex / Cursor / Gemini are detected by their config dir → skill copied."""
    home = tmp_path / "h"
    home.mkdir()
    r = _run(test_image, home, setup=f"rm /usr/local/bin/claude\n{setup}")
    assert r.returncode == 0, r.stderr
    assert f"Installed motus skill for: {agent}" in r.stdout
    assert (home / skill_path).exists()


def test_claude_config_dir_honored(test_image, tmp_path):
    """CLAUDE_CONFIG_DIR overrides the default ~/.claude lookup."""
    home = tmp_path / "h"
    home.mkdir()
    r = _run(
        test_image,
        home,
        setup="mkdir -p $HOME/relocated",
        env="CLAUDE_CONFIG_DIR=$HOME/relocated",
    )
    assert r.returncode == 0, r.stderr
    assert "Installed motus skill for: Claude Code" in r.stdout
    installed = json.loads(
        (home / "relocated/plugins/installed_plugins.json").read_text()
    )
    assert "motus@LithosAI" in installed["plugins"]
