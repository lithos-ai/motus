"""CLI for motus deploy — pack and upload a project to the build service."""

import sys

from motus.config import CONFIG


def _deploy_handler(args):
    import os

    from motus.deploy.deploy import deploy

    # Resolve name vs. project_id (flags override config entirely)
    flag_project_id = getattr(args, "project_id", None)
    flag_name = getattr(args, "name", None)

    if flag_project_id:
        project_id = flag_project_id
        name = None
    elif flag_name:
        project_id = None
        name = flag_name
    elif CONFIG.get("project_id"):
        project_id = CONFIG["project_id"]
        name = None
    elif CONFIG.get("name"):
        project_id = None
        name = CONFIG["name"]
    else:
        print(
            "Error: No project specified. Pass --name to create a new project"
            " or --project-id for an existing one."
        )
        sys.exit(1)

    import_path = getattr(args, "import_path") or CONFIG.get("import_path")
    if not import_path:
        print(
            "Error: No import path provided. Pass import-path argument or set 'import_path' in motus.toml"
        )
        sys.exit(1)

    git_url = getattr(args, "git_url")
    git_ref = getattr(args, "git_ref")

    if git_ref and not git_url:
        print("Error: Git ref provided without Git URL")
        sys.exit(1)

    CONFIG.update(
        {
            k: v
            for k, v in {
                "import_path": import_path,
                "git_url": git_url,
                "git_ref": git_ref,
            }.items()
            if v is not None
        }
    )

    secrets = {}
    for item in args.secrets or []:
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key = item
            value = os.environ.get(key)
            if value is None:
                print(f"Error: Secret '{key}' not found in environment")
                sys.exit(1)
        secrets[key] = value

    deploy(
        name=name,
        project_id=project_id,
        import_path=import_path,
        git_url=git_url,
        git_ref=git_ref,
        secrets=secrets or None,
    )


def register_cli(subparsers):
    """Register the 'deploy' command group."""
    deploy_parser = subparsers.add_parser(
        "deploy", help="deploy a project to the cloud"
    )
    project_group = deploy_parser.add_mutually_exclusive_group()
    project_group.add_argument(
        "--name",
        default=None,
        help="project name (creates a new project if needed)",
    )
    project_group.add_argument(
        "--project-id",
        default=None,
        help="existing project ID (reads from motus.toml if omitted)",
    )
    deploy_parser.add_argument(
        "import_path",
        nargs="?",
        default=None,
        metavar="import-path",
        help="server import path (module:variable; reads from motus.toml if omitted)",
    )
    deploy_parser.add_argument(
        "--git-url",
        metavar="GIT_URL",
        help="Git repository URL to build from instead of uploading local files",
    )
    deploy_parser.add_argument(
        "--git-ref",
        metavar="GIT_REF",
        help="Git ref (branch, tag, or commit SHA) to check out (used with --git-url)",
    )
    deploy_parser.add_argument(
        "--secret",
        dest="secrets",
        metavar="KEY=VALUE",
        action="append",
        help="secret env var for the agent container; KEY=VALUE or just KEY to read from environment (repeatable)",
    )
    deploy_parser.set_defaults(func=_deploy_handler)
