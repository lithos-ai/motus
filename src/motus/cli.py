import argparse
import importlib
import json
import sys
import time
from pathlib import Path

_COMMAND_MODULES = [
    "motus.auth.cli",
    "motus.serve.cli",
    "motus.deploy.cli",
]

_PACKAGE = "lithosai-motus"


class _Formatter(argparse.HelpFormatter):
    """Skip the metavar header line for subparser groups."""

    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            return self._join_parts(
                self._format_action(a) for a in action._get_subactions()
            )
        return super()._format_action(action)


# -- Version check -------------------------------------------------------------

_CHECK_INTERVAL = 86400  # 24 hours
_TIMEOUT = 1.0  # seconds
_CACHE = Path.home() / ".motus" / "version_check.json"


def _check_for_update() -> None:
    """Print a message to stderr if a newer version is available on PyPI.

    Checks at most once every 24 hours.  All errors are silently ignored
    so this never blocks or disrupts the CLI.
    """
    try:
        from importlib.metadata import version

        current = version(_PACKAGE)

        try:
            cache = json.loads(_CACHE.read_text())
        except Exception:
            cache = {}

        now = time.time()

        if now - cache.get("last_check", 0) < _CHECK_INTERVAL:
            latest = cache.get("latest")
            if latest and latest != current:
                _print_update_message(current, latest)
            return

        import urllib.request

        url = f"https://pypi.org/pypi/{_PACKAGE}/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())

        latest = data["info"]["version"]

        try:
            _CACHE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE.write_text(json.dumps({"last_check": now, "latest": latest}))
        except Exception:
            pass

        if latest != current:
            _print_update_message(current, latest)
    except Exception:
        pass


def _print_update_message(current: str, latest: str) -> None:
    print(
        f"motus update available: {latest} (current: {current}). "
        f"Run: uv tool upgrade {_PACKAGE}",
        file=sys.stderr,
    )


def _ensure_utf8_stdout():
    """Reconfigure stdout/stderr to UTF-8 on platforms that default to a narrower encoding."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# -- Entry point ---------------------------------------------------------------


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        prog="motus",
        description="Motus Agent Framework",
        formatter_class=_Formatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", title="commands", metavar="<command>"
    )

    for module_path in _COMMAND_MODULES:
        module = importlib.import_module(module_path)
        module.register_cli(subparsers)

    _check_for_update()

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.parse_args([args.command, "--help"])
