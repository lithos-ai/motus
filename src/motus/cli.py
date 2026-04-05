import argparse
import importlib
import sys

_COMMAND_MODULES = [
    "motus.auth.cli",
    "motus.serve.cli",
    "motus.deploy.cli",
]


class _Formatter(argparse.HelpFormatter):
    """Skip the metavar header line for subparser groups."""

    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            return self._join_parts(
                self._format_action(a) for a in action._get_subactions()
            )
        return super()._format_action(action)


def main():
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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.parse_args([args.command, "--help"])
