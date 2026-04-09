"""CLI commands for authentication: login, logout, whoami."""

import logging
import sys

from motus.auth.credentials import (
    clear_credentials,
    ensure_authenticated,
    get_api_url,
    load_credentials,
)


def _revoke_existing_key():
    """Revoke the currently stored API key (best-effort)."""
    import httpx

    creds = load_credentials()
    if not creds:
        return
    try:
        httpx.delete(
            f"{creds.cloud_api_url}/api-keys/{creds.key_id}",
            headers={"Authorization": f"Bearer {creds.api_key}"},
            timeout=10,
        )
    except Exception:
        pass


def _login_handler(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    api_url = args.api_url
    if not api_url:
        logging.error("No API URL provided. Pass --api-url <URL>")
        sys.exit(1)

    # Explicit login always revokes + re-authenticates
    _revoke_existing_key()
    clear_credentials()

    try:
        ensure_authenticated()
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        sys.exit(1)
    except Exception as e:
        logging.error("Login failed: %s", e)
        sys.exit(1)


def _logout_handler(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    _revoke_existing_key()
    clear_credentials()
    print("Logged out.")


def _whoami_handler(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    creds = load_credentials()
    if not creds:
        print("Not logged in. Run: motus login --api-url <URL>")
        sys.exit(1)

    prefix = creds.api_key[:12]
    print(f"API URL: {creds.cloud_api_url}")
    print(f"API key: {prefix}...")


def register_cli(subparsers):
    """Register login, logout, whoami commands."""
    login_parser = subparsers.add_parser("login", help="log in to Motus cloud")
    login_parser.add_argument(
        "--api-url",
        default=get_api_url(),
        metavar="URL",
        help="Motus API URL",
    )
    login_parser.set_defaults(func=_login_handler)

    logout_parser = subparsers.add_parser("logout", help="log out and revoke API key")
    logout_parser.set_defaults(func=_logout_handler)

    whoami_parser = subparsers.add_parser("whoami", help="show current login status")
    whoami_parser.set_defaults(func=_whoami_handler)
