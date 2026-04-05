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

import importlib
import os
import sys


def _parse_params(items):
    """Parse KEY=VALUE param strings, coercing numeric values."""
    params = {}
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


def _auth_headers() -> dict:
    """Return Authorization headers from login credentials or env vars, if available."""
    from motus.auth.credentials import get_api_key

    api_key = get_api_key()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _api_call(method, url, *, json_body=None, params=None, timeout=30):
    """Make an API call, return parsed JSON (or None for 204)."""
    import httpx

    with httpx.Client(timeout=timeout, headers=_auth_headers()) as client:
        kwargs = {}
        if json_body is not None:
            kwargs["json"] = json_body
        if params:
            kwargs["params"] = params
        try:
            r = getattr(client, method)(url, **kwargs)
        except httpx.ConnectError:
            print(f"Error: Could not connect to {url}")
            sys.exit(1)
        if r.status_code >= 400:
            print(f"Error: {r.status_code} - {r.text}")
            sys.exit(1)
        if r.status_code == 204:
            return None
        return r.json()


def _print_json(model_or_adapter, data):
    """Validate data against a Pydantic model/adapter and print as JSON."""
    from pydantic import TypeAdapter

    if isinstance(model_or_adapter, TypeAdapter):
        obj = model_or_adapter.validate_python(data)
        print(model_or_adapter.dump_json(obj, indent=2, exclude_none=True).decode())
    else:
        obj = model_or_adapter.model_validate(data)
        print(obj.model_dump_json(indent=2, exclude_none=True))


def start_server(args):
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


def _send_and_wait(client, base_url, session_id, message, params=None):
    """Send a message and wait for the result."""
    import httpx

    body = {"content": message}
    if params:
        body["user_params"] = params
    try:
        r = client.post(
            f"{base_url}/sessions/{session_id}/messages",
            json=body,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"Error: {e.response.status_code} - {e.response.text}")
        return

    r = client.get(f"{base_url}/sessions/{session_id}", params={"wait": "true"})
    r.raise_for_status()
    data = r.json()

    if data["status"] == "error":
        print(f"Error: {data['error']}")
    elif data.get("response"):
        print(data["response"]["content"])


def chat_command(args):
    """Chat with the agent via the sessions API.

    If a message is provided, sends a single request.
    Otherwise enters interactive multi-turn mode.
    """
    import httpx

    base_url = args.url.rstrip("/")
    message = args.message
    owned_session = args.session is None

    params = _parse_params(args.params)

    try:
        with httpx.Client(timeout=600, headers=_auth_headers()) as client:
            if args.session:
                session_id = args.session
            else:
                try:
                    r = client.post(f"{base_url}/sessions")
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    print(
                        f"Error creating session: {e.response.status_code} - {e.response.text}"
                    )
                    sys.exit(1)
                session_id = r.json()["session_id"]
                if args.keep:
                    print(f"Session: {session_id} (use --session to resume)")

            try:
                if message:
                    _send_and_wait(
                        client, base_url, session_id, message, params=params or None
                    )
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
                        _send_and_wait(
                            client,
                            base_url,
                            session_id,
                            user_input,
                            params=params or None,
                        )
            finally:
                if owned_session and not args.keep:
                    client.delete(f"{base_url}/sessions/{session_id}")
    except httpx.ConnectError:
        print(f"Error: Could not connect to {base_url}")
        sys.exit(1)


def health_check(args):
    """Check server health."""
    import httpx

    base_url = args.url.rstrip("/")

    with httpx.Client(timeout=10) as client:
        try:
            r = client.get(f"{base_url}/health")
            r.raise_for_status()
        except httpx.ConnectError:
            print(f"Error: Could not connect to {base_url}")
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code} - {e.response.text}")
            sys.exit(1)

        data = r.json()
        print(f"Status: {data['status']}")
        print(f"Workers: {data['running_workers']}/{data['max_workers']}")
        print(f"Total sessions: {data['total_sessions']}")


def create_session(args):
    """Create a new session."""
    from .schemas import SessionResponse

    base_url = args.url.rstrip("/")
    data = _api_call("post", f"{base_url}/sessions")
    _print_json(SessionResponse, data)


def list_sessions(args):
    """List all sessions."""
    from pydantic import TypeAdapter

    from .schemas import SessionSummary

    base_url = args.url.rstrip("/")
    data = _api_call("get", f"{base_url}/sessions")
    _print_json(TypeAdapter(list[SessionSummary]), data)


def get_session(args):
    """Get session details."""
    from .schemas import SessionResponse

    base_url = args.url.rstrip("/")
    params = {}
    if args.wait:
        params["wait"] = "true"
    if args.timeout is not None:
        params["timeout"] = str(args.timeout)
    timeout = max(30, (args.timeout or 0) + 10)
    data = _api_call(
        "get", f"{base_url}/sessions/{args.id}", params=params, timeout=timeout
    )
    _print_json(SessionResponse, data)


def delete_session(args):
    """Delete a session."""
    base_url = args.url.rstrip("/")
    _api_call("delete", f"{base_url}/sessions/{args.id}")
    print(f"Deleted session {args.id}")


def get_messages(args):
    """Get messages for a session."""
    from pydantic import TypeAdapter

    from ..models.base import ChatMessage

    base_url = args.url.rstrip("/")
    data = _api_call("get", f"{base_url}/sessions/{args.id}/messages")
    _print_json(TypeAdapter(list[ChatMessage]), data)


def send_message(args):
    """Send a message to a session."""
    from .schemas import MessageResponse, SessionResponse

    base_url = args.url.rstrip("/")
    body: dict = {"role": args.role, "content": args.message}
    params = _parse_params(args.params)
    if params:
        body["user_params"] = params
    if args.webhook_url:
        webhook = {"url": args.webhook_url}
        if args.webhook_token:
            webhook["token"] = args.webhook_token
        if args.webhook_include_messages:
            webhook["include_messages"] = True
        body["webhook"] = webhook
    data = _api_call("post", f"{base_url}/sessions/{args.id}/messages", json_body=body)
    if args.wait:
        params = {"wait": "true"}
        if args.timeout is not None:
            params["timeout"] = str(args.timeout)
        timeout = max(30, (args.timeout or 0) + 10)
        data = _api_call(
            "get", f"{base_url}/sessions/{args.id}", params=params, timeout=timeout
        )
        _print_json(SessionResponse, data)
    else:
        _print_json(MessageResponse, data)


def register_cli(subparsers):
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
        "--ttl",
        type=float,
        default=0,
        help="idle session TTL in seconds (0 disables)",
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
        "--session",
        default=None,
        help="existing session ID to resume",
    )
    chat_parser.add_argument(
        "--keep",
        action="store_true",
        help="keep session alive after exit",
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
        "--timeout",
        type=float,
        default=None,
        help="wait timeout in seconds",
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
        "--timeout",
        type=float,
        default=None,
        help="wait timeout in seconds",
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
