import importlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

if sys.version_info >= (3, 14):
    import tarfile
else:
    from backports.zstd import tarfile

import httpx
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

from motus.auth.credentials import ensure_authenticated
from motus.config import CONFIG
from motus.deploy.walk import walk

DEPLOY_MESSAGES = [
    "Deploying...",
    "Igniting boosters...",
    "Clearing the launchpad...",
    "Plotting trajectory...",
    "Entering orbit...",
    "Charging warp drive...",
    "Adjusting solar panels...",
    "Locking in coordinates...",
    "Passing through the atmosphere...",
    "Approaching cruising altitude...",
    "Coasting through the cosmos...",
    "Engaging thrusters...",
    "Running pre-flight checks...",
    "Aligning star charts...",
    "Reaching escape velocity...",
    "Scanning the horizon...",
    "Flying past the moon...",
    "Firing retro rockets...",
    "Stabilizing flight path...",
    "Navigating the stars...",
]


class DeployStatus:
    """Live status bar with spinner + elapsed time. No-op when stdout is not a TTY."""

    def __init__(self, phase: str = ""):
        self._live = None
        self._phase = phase
        self._start = time.monotonic()
        self._rotating_phase = None

    def __enter__(self):
        if sys.stdout.isatty():
            self._live = Live(
                get_renderable=self._render,
                refresh_per_second=8,
                redirect_stdout=True,
                transient=True,
            )
            self._live.start()
        return self

    def __exit__(self, exc_type, *_exc):
        if self._live:
            self._live.stop()
            if exc_type is None:
                print(f"Done in {self._elapsed()}.")

    def update(self, phase: str):
        """Set the status bar phase, or print the phase if not a TTY."""
        self._phase = phase
        self._rotating_phase = None
        if not self._live:
            print(phase)

    def rotate(self, messages: list[str], interval: float = 5.0):
        """Cycle through shuffled messages at the given interval."""
        first, *rest = messages
        shuffled = [first] + random.sample(rest, len(rest))
        start = time.monotonic()
        self._rotating_phase = lambda: shuffled[
            int((time.monotonic() - start) // interval) % len(shuffled)
        ]
        self._phase = first
        if not self._live:
            print(first)

    def _render(self):
        if self._rotating_phase:
            self._phase = self._rotating_phase()
        table = Table.grid(padding=(0, 1))
        table.add_row(Spinner("dots"), f"{self._phase} ({self._elapsed()})")
        return table

    def _elapsed(self) -> str:
        elapsed = int(time.monotonic() - self._start)
        mins, secs = divmod(elapsed, 60)
        return f"{mins}m{secs}s" if mins else f"{secs}s"


def stream_build_status(status_url: str, auth_headers: dict, status: DeployStatus):
    """Connect to the build status SSE stream and print log output in real time."""
    with httpx.Client(timeout=None, headers=auth_headers) as client:
        try:
            with client.stream("GET", status_url) as resp:
                if resp.status_code != 200:
                    try:
                        detail = json.loads(resp.read()).get("message", "")
                    except Exception:
                        detail = ""
                    if detail:
                        print(f"Error: {detail}")
                    else:
                        print(
                            f"Error: Failed to connect to build status stream: {resp.status_code}"
                        )
                    sys.exit(1)
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    event = data.get("status")
                    log_line = data.get("log_line")
                    if event == "deployed":
                        print("Deploy complete!")
                        return
                    if event == "failed":
                        print("Error: Build failed.")
                        sys.exit(1)
                    if not log_line:
                        if event == "queued":
                            status.update("Waiting for build...")
                        elif event == "building":
                            status.update("Building...")
                        elif event == "built":
                            status.update("Waiting for deploy...")
                        elif event == "deploying":
                            status.rotate(DEPLOY_MESSAGES)
                    else:
                        print(log_line)
        except httpx.ConnectError:
            print("Error: Could not connect to build status stream")
            sys.exit(1)
    # Stream ended without a terminal status
    print("Error: Build status stream ended unexpectedly")
    sys.exit(1)


def _exit_with_api_error(action: str, resp: httpx.Response) -> None:
    try:
        detail = resp.json().get("message", "")
    except Exception:
        detail = ""
    if detail:
        print(f"Error: {detail}")
    else:
        print(f"Error: {action}: {resp.status_code}")
    sys.exit(1)


def deploy(
    *,
    name: str | None = None,
    project_id: str | None = None,
    import_path: str,
    git_url: str | None = None,
    git_ref: str | None = None,
    secrets: dict[str, str] | None = None,
):
    """Validate, pack, and upload a project to the build service."""
    api_url, api_key = ensure_authenticated()
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    project_path = Path.cwd()

    with DeployStatus("Preparing...") as status:
        # Validate import path
        print("Validating import path...")
        if ":" not in import_path:
            print(
                f"Error: Invalid import path '{import_path}'. Expected format: module.path:server_variable"
            )
            sys.exit(1)

        module_path, attr_name = import_path.rsplit(":", 1)

        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))

        try:
            module = importlib.import_module(module_path)
            getattr(module, attr_name)
        except ImportError as e:
            print(f"Error: Could not import module '{module_path}': {e}")
            sys.exit(1)
        except AttributeError:
            print(f"Error: Module '{module_path}' has no attribute '{attr_name}'")
            sys.exit(1)
        print("Import path validated.")

        # Resolve project
        if project_id:
            try:
                resp = httpx.get(
                    f"{api_url}/projects/{project_id}",
                    headers=auth_headers,
                )
                if resp.status_code == 200:
                    print(f"Using project: {project_id}")
                else:
                    _exit_with_api_error("Failed to look up project", resp)
            except httpx.HTTPError as e:
                print(f"Error: Failed to look up project: {e}")
                sys.exit(1)
        elif name:
            print("Creating project...")
            try:
                resp = httpx.post(
                    f"{api_url}/projects",
                    json={"name": name},
                    headers=auth_headers,
                )
                if resp.status_code == 201:
                    project_id = resp.json()["project_id"]
                    print(f"Project created: {project_id}")
                else:
                    _exit_with_api_error("Failed to create project", resp)
            except httpx.HTTPError as e:
                print(f"Error: Failed to create project: {e}")
                sys.exit(1)

        # Create build
        print("Creating build...")
        build_payload: dict = {"project_id": project_id, "import_path": import_path}
        if git_url:
            build_payload["git_repo"] = {"url": git_url, "ref": git_ref}
        if secrets:
            build_payload["secrets"] = secrets

        try:
            resp = httpx.post(
                f"{api_url}/builds",
                json=build_payload,
                headers=auth_headers,
            )
            if resp.status_code != 201:
                _exit_with_api_error("Failed to create build", resp)
            body = resp.json()
        except httpx.HTTPError as e:
            print(f"Error: Failed to create build: {e}")
            sys.exit(1)
        build_id = body["build_id"]
        status_url = body.get("status_url", "")
        if not status_url:
            print("Error: Build response missing status_url")
            sys.exit(1)
        print(f"Build created: {build_id}")

        # Persist project/build to motus.toml for tracing
        CONFIG.update(project_id=project_id, build_id=build_id)

        # Git-based builds are queued immediately by the API and do not need an archive upload.
        if git_url:
            print(f"Queued git build from {git_url} (ref={git_ref})")
            stream_build_status(status_url, auth_headers, status)
            return

        source = body.get("source")
        if not isinstance(source, dict) or not source.get("upload_url"):
            print("Error: Unexpected build response (missing upload URL)")
            sys.exit(1)
        upload_url = str(source["upload_url"])

        with tempfile.TemporaryDirectory() as tmp_dir_path:
            tmp_dir_path = Path(tmp_dir_path)

            # Pack the project into an archive
            tar_path = tmp_dir_path / "project.tar.zst"
            print("Packing project archive...")
            with tarfile.open(tar_path, mode="w|zst") as tar:
                for file_path in walk(project_path):
                    tar.add(project_path / file_path, arcname=file_path)
            print("Project archive packed.")

            # Upload the archive to S3
            print("Uploading project archive...")
            try:
                with open(tar_path, "rb") as tar_file:
                    response = httpx.put(
                        upload_url,
                        content=tar_file,
                        headers={"Content-Type": "application/zstd"},
                    )
                    response.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(f"Error: Failed to upload archive: {e.response.status_code}")
                sys.exit(1)
            except httpx.HTTPError as e:
                print(f"Error: Failed to upload archive: {e}")
                sys.exit(1)
            print("Project archive uploaded.")

        stream_build_status(status_url, auth_headers, status)
