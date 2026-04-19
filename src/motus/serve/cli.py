"""CLI for motus serve — start servers and interact with them.

Usage (via unified CLI):
    motus serve start myapp:my_agent --port 8000
    motus serve chat http://localhost:8000 "hello world"
    motus serve chat http://localhost:8000 --session abc-123
    motus serve chat http://localhost:8000 --keep
    motus serve health http://localhost:8000
    motus serve create http://localhost:8000
    motus serve sessions http://localhost:8000
    motus serve get http://localhost:8000 <session-id> --wait
    motus serve delete http://localhost:8000 <session-id>
    motus serve messages http://localhost:8000 <session-id>
    motus serve send http://localhost:8000 <session-id> "hello" --wait
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def _parse_params(items: list[str] | None) -> dict:
    """Parse KEY=VALUE param strings, coercing numeric values."""
    params: dict = {}
    for item in items or []:
        if "=" not in item:
            print(f"Invalid --param format '{item}'. Expected KEY=VALUE.")
            sys.exit(1)
        key, value = item.split("=", 1)
        for typ in (int, float):
            try:
                value = typ(value)
                break
            except ValueError:
                pass
        params[key] = value
    return params


def _make_client(base_url: str):
    """Build a motus.cloud.Client bound to ``base_url``."""
    from motus.cloud import Client

    return Client(base_url=base_url.rstrip("/"))


def _print_json_model(obj) -> None:
    if hasattr(obj, "model_dump_json"):
        print(obj.model_dump_json(indent=2, exclude_none=True))
    else:
        import json

        print(json.dumps(obj, indent=2, default=str))


def _handle_client_error(prefix: str, exc: Exception) -> None:
    from motus.cloud import MotusClientError

    if isinstance(exc, MotusClientError):
        print(f"{prefix}: {exc}")
    else:
        print(f"{prefix}: {type(exc).__name__}: {exc}")
    sys.exit(1)


def start_server(args) -> None:
    """Start a serve server from an agent import path or directory."""
    import_path = args.agent
    host = args.host
    port = args.port
    workers = args.workers
    log_level = args.log_level

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    if ":" not in import_path:
        print(f"Error: Invalid import path '{import_path}'")
        print("Expected format: module.path:variable")
        sys.exit(1)

    module_path, attr_name = import_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
        obj = getattr(module, attr_name)
    except ImportError as e:
        print(f"Error: Could not import module '{module_path}': {e}")
        sys.exit(1)
    except AttributeError:
        print(f"Error: Module '{module_path}' has no attribute '{attr_name}'")
        sys.exit(1)

    from motus.serve.protocol import ServableAgent
    from motus.serve.worker import _is_openai_agent

    is_valid = isinstance(obj, ServableAgent) or _is_openai_agent(obj) or callable(obj)
    if not is_valid:
        print(
            f"Error: '{import_path}' is not a ServableAgent, OpenAI Agent, or callable"
        )
        sys.exit(1)

    from .server import AgentServer

    server = AgentServer(
        import_path,
        max_workers=workers,
        ttl=args.ttl,
        timeout=args.timeout,
        max_sessions=args.max_sessions,
        shutdown_timeout=args.shutdown_timeout,
        allow_custom_ids=args.allow_custom_ids,
    )
    server.run(host=host, port=port, log_level=log_level)


# ---------- Interrupt prompting (CLI-only: TTY I/O) ----------


def _prompt_interrupt(intr) -> Any:
    """Prompt the user for a single interrupt; returns the resume value."""
    intr_type = intr.type
    payload = intr.payload

    if intr_type == "tool_approval":
        tool_name = payload.get("tool_name", "<unknown tool>")
        tool_args = payload.get("tool_args", {})
        print(f"\n[approval] Agent wants to call: {tool_name}")
        print(f"           Args: {tool_args}")
        answer = input("           Approve? [y/N]: ").strip().lower()
        return {"approved": answer == "y"}

    if intr_type == "user_input":
        questions = payload.get("questions", [])
        answers: dict = {}
        for q in questions:
            qtext = q.get("question", "")
            options = q.get("options", [])
            print(f"\n[ask] {qtext}")
            for i, opt in enumerate(options, start=1):
                print(f"  {i}. {opt.get('label')} — {opt.get('description', '')}")
            print(f"  {len(options) + 1}. (Other — type your own answer)")
            choice = input("  Choice: ").strip()
            try:
                idx = int(choice)
                if 1 <= idx <= len(options):
                    answers[qtext] = options[idx - 1]["label"]
                else:
                    answers[qtext] = input("  Your answer: ").strip()
            except ValueError:
                answers[qtext] = choice
        return {"answers": answers}

    print(f"[warn] unknown interrupt type: {intr_type}, returning no-op")
    return None


def _run_turn(session, content: str, params: dict | None) -> None:
    """Run one turn on ``session`` with interrupt loop; print assistant content on idle."""
    result = session.chat(content, user_params=params or None)
    while result.status.value == "interrupted":
        for intr in result.interrupts:
            value = _prompt_interrupt(intr)
            if value is None:
                return
            result = session.resume(intr.id, value)
            if result.status.value != "interrupted":
                break
    if result.status.value == "idle" and result.message is not None:
        print(result.message.content or "")
    elif result.status.value == "error":
        snap = result.snapshot
        print(f"Error: {snap.error if snap else 'unknown'}")


def chat_command(args) -> None:
    """Chat with the agent via the sessions API (delegates to motus.cloud)."""
    from motus.cloud import MotusClientError, Session, SessionNotFound

    params = _parse_params(args.params)
    client = _make_client(args.url)
    try:
        try:
            if args.session:
                # --session means "resume an existing conversation". Verify the
                # session actually exists before attaching so a typo surfaces
                # as "session not found" instead of silently creating a new
                # session or raising a misleading custom-ID error.
                try:
                    client.get_session(args.session)
                except SessionNotFound:
                    print(f"Error: session {args.session} not found")
                    sys.exit(1)
                session = Session(
                    client,
                    args.session,
                    owned=False,
                    keep=bool(args.keep),
                )
            else:
                session = client.session(keep=bool(args.keep))
        except MotusClientError as e:
            _handle_client_error("Error creating session", e)
            return  # for type-checkers; _handle_client_error exits

        if session.owned and args.keep:
            print(f"Session: {session.session_id} (use --session to resume)")

        try:
            with session:
                if args.message:
                    _run_turn(session, args.message, params)
                else:
                    print("Chat session started (Ctrl+C to quit)")
                    print()
                    while True:
                        try:
                            user_input = input("> ")
                        except (EOFError, KeyboardInterrupt):
                            print("\nBye!")
                            break
                        if not user_input.strip():
                            continue
                        _run_turn(session, user_input, params)
        except MotusClientError as e:
            _handle_client_error("Error", e)
    finally:
        client.close()


def health_check(args) -> None:
    """Check server health."""
    from motus.cloud import MotusClientError

    client = _make_client(args.url)
    try:
        data = client.health()
        print(f"Status: {data['status']}")
        if "max_workers" in data:
            print(f"Workers: {data['running_workers']}/{data['max_workers']}")
        if "total_sessions" in data:
            print(f"Total sessions: {data['total_sessions']}")
    except MotusClientError as e:
        _handle_client_error("Error", e)
    finally:
        client.close()


def create_session(args) -> None:
    """Create a new session."""
    from motus.cloud import MotusClientError

    client = _make_client(args.url)
    try:
        _print_json_model(client.create_session())
    except MotusClientError as e:
        _handle_client_error("Error", e)
    finally:
        client.close()


def list_sessions(args) -> None:
    """List all sessions."""
    from motus.cloud import MotusClientError

    client = _make_client(args.url)
    try:
        _print_json_model(client.list_sessions())
    except MotusClientError as e:
        _handle_client_error("Error", e)
    finally:
        client.close()


def get_session(args) -> None:
    """Get session details."""
    from motus.cloud import MotusClientError

    client = _make_client(args.url)
    try:
        snap = client.get_session(args.id, wait=args.wait, timeout=args.timeout)
        _print_json_model(snap)
    except MotusClientError as e:
        _handle_client_error("Error", e)
    finally:
        client.close()


def delete_session(args) -> None:
    """Delete a session."""
    from motus.cloud import MotusClientError

    client = _make_client(args.url)
    try:
        client.delete_session(args.id)
        print(f"Deleted session {args.id}")
    except MotusClientError as e:
        _handle_client_error("Error", e)
    finally:
        client.close()


def get_messages(args) -> None:
    """Get messages for a session."""
    from motus.cloud import MotusClientError

    client = _make_client(args.url)
    try:
        _print_json_model(client.get_messages(args.id))
    except MotusClientError as e:
        _handle_client_error("Error", e)
    finally:
        client.close()


def send_message(args) -> None:
    """Send a message to a session."""
    from motus.cloud import MotusClientError

    client = _make_client(args.url)
    try:
        params = _parse_params(args.params)
        webhook: dict[str, Any] | None = None
        if args.webhook_url:
            webhook = {"url": args.webhook_url}
            if args.webhook_token:
                webhook["token"] = args.webhook_token
            if args.webhook_include_messages:
                webhook["include_messages"] = True
        if args.wait:
            client.send_message(
                args.id,
                content=args.message,
                role=args.role,
                user_params=params or None,
                webhook=webhook,
            )
            snap = client.get_session(args.id, wait=True, timeout=args.timeout)
            _print_json_model(snap)
        else:
            resp = client.send_message(
                args.id,
                content=args.message,
                role=args.role,
                user_params=params or None,
                webhook=webhook,
            )
            _print_json_model(resp)
    except MotusClientError as e:
        _handle_client_error("Error", e)
    finally:
        client.close()


def register_cli(subparsers) -> None:
    """Register the 'serve' command group with the top-level CLI."""
    from motus.cli import _Formatter

    serve_parser = subparsers.add_parser(
        "serve",
        help="run and manage agent servers",
        formatter_class=_Formatter,
    )
    sub = serve_parser.add_subparsers(
        dest="serve_command", title="commands", metavar="<command>"
    )

    # start command
    start_parser = sub.add_parser("start", help="start an agent server")
    start_parser.add_argument(
        "agent", help="agent import path (module:variable) or directory"
    )
    start_parser.add_argument(
        "--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)"
    )
    start_parser.add_argument(
        "--port", type=int, default=8000, help="bind port (default: 8000)"
    )
    start_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (default: CPU count)",
    )
    start_parser.add_argument(
        "--ttl", type=float, default=0, help="idle session TTL in seconds (0 disables)"
    )
    start_parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="max seconds per agent turn (0 disables)",
    )
    start_parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="max concurrent sessions (0 = unlimited)",
    )
    start_parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=0,
        help="graceful shutdown timeout in seconds (0 waits forever)",
    )
    start_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="log level (default: info)",
    )
    start_parser.add_argument(
        "--allow-custom-ids",
        action="store_true",
        default=False,
        help="enable PUT /sessions/{id} for client-specified session IDs",
    )
    start_parser.set_defaults(func=start_server)

    # chat command
    chat_parser = sub.add_parser(
        "chat",
        help="chat with an agent",
        description="Send a message or enter interactive mode.",
    )
    chat_parser.add_argument("url", help="server URL")
    chat_parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="message to send (omit for interactive mode)",
    )
    chat_parser.add_argument(
        "--session", default=None, help="existing session ID to resume"
    )
    chat_parser.add_argument(
        "--keep", action="store_true", help="keep session alive after exit"
    )
    chat_parser.add_argument(
        "--param",
        dest="params",
        metavar="KEY=VALUE",
        action="append",
        help="per-request parameter passed to the agent (repeatable)",
    )
    chat_parser.set_defaults(func=chat_command)

    # health command
    health_parser = sub.add_parser("health", help="check server health")
    health_parser.add_argument("url", help="server URL")
    health_parser.set_defaults(func=health_check)

    # create command
    create_parser = sub.add_parser("create", help="create a new session")
    create_parser.add_argument("url", help="server URL")
    create_parser.set_defaults(func=create_session)

    # sessions command
    sessions_parser = sub.add_parser("sessions", help="list all sessions")
    sessions_parser.add_argument("url", help="server URL")
    sessions_parser.set_defaults(func=list_sessions)

    # get command
    get_parser = sub.add_parser("get", help="get session details")
    get_parser.add_argument("url", help="server URL")
    get_parser.add_argument("id", help="session ID")
    get_parser.add_argument("--wait", action="store_true", help="block until complete")
    get_parser.add_argument(
        "--timeout", type=float, default=None, help="wait timeout in seconds"
    )
    get_parser.set_defaults(func=get_session)

    # delete command
    delete_parser = sub.add_parser("delete", help="delete a session")
    delete_parser.add_argument("url", help="server URL")
    delete_parser.add_argument("id", help="session ID")
    delete_parser.set_defaults(func=delete_session)

    # messages command
    messages_parser = sub.add_parser("messages", help="get session messages")
    messages_parser.add_argument("url", help="server URL")
    messages_parser.add_argument("id", help="session ID")
    messages_parser.set_defaults(func=get_messages)

    # send command
    send_parser = sub.add_parser("send", help="send a message to a session")
    send_parser.add_argument("url", help="server URL")
    send_parser.add_argument("id", help="session ID")
    send_parser.add_argument("message", help="message content")
    send_parser.add_argument(
        "--role", default="user", help="message role (default: user)"
    )
    send_parser.add_argument("--wait", action="store_true", help="block until complete")
    send_parser.add_argument(
        "--timeout", type=float, default=None, help="wait timeout in seconds"
    )
    send_parser.add_argument(
        "--webhook-url", default=None, help="completion webhook URL"
    )
    send_parser.add_argument(
        "--webhook-token", default=None, help="webhook bearer token"
    )
    send_parser.add_argument(
        "--webhook-include-messages",
        action="store_true",
        help="include message history in webhook payload",
    )
    send_parser.add_argument(
        "--param",
        dest="params",
        metavar="KEY=VALUE",
        action="append",
        help="per-request parameter passed to the agent (repeatable)",
    )
    send_parser.set_defaults(func=send_message)
