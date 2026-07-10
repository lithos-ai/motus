import asyncio
import os
import shutil
from typing import Awaitable, Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from openai import AsyncOpenAI
from uvicorn import Config, Server

from .runtime import init, is_initialized, shutdown
from .tools import (
    DEFAULT_TOOL_PROVIDER,
    FunctionTool,
    InputSchema,
    Tools,
    get_sandbox,
    normalize_tools,
    tool,
    tools,
    tools_from,
)
from .tracing import setup_tracing

# Configure OpenTelemetry at import time so every motus entry point — the
# runtime, framework integrations, CLIs, tests — emits spans through the
# same TracerProvider without each caller needing to remember setup. Config
# is frozen to env-var state here; programmatic overrides after import
# require ``shutdown_tracing(); setup_tracing(new_config)``.
setup_tracing()

__all__ = [
    "DEFAULT_TOOL_PROVIDER",
    "FunctionTool",
    "InputSchema",
    "Motus",
    "ModelClient",
    "Tools",
    "get_sandbox",
    "init",
    "is_initialized",
    "normalize_tools",
    "shutdown",
    "tool",
    "tools",
    "tools_from",
]


class ModelClient(AsyncOpenAI):
    """OpenAI chat completions."""

    def __init__(
        self,
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        if not api_key and not base_url:
            base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)


class Motus:
    def __init__(
        self,
        brave_api_key: str = os.getenv("BRAVE_API_KEY", ""),
        openai_api_key: str = os.getenv("OPENAI_API_KEY", ""),
        openai_base_url: str | None = None,
    ):
        self.models = ModelClient(openai_api_key, openai_base_url)
        # Give them another nice alias. (Let's move ToolClient.)
        self.tools = DEFAULT_TOOL_PROVIDER()
        self.app = FastAPI()

    def sandbox(self, *args, **kwargs):
        # We probably shouldn't be unwrapping an attribute like this.
        return self.tools.get_sandbox(*args, **kwargs)

    async def serve(
        self,
        agent: Callable[..., Awaitable],
        host: str = os.getenv("MOTUS_HOST", ""),
        port: int = int(os.getenv("MOTUS_PORT", "0")),
    ):
        """Serve a callable as an agent."""
        self.app.get("/agent")(agent)

        if host:
            # Defer default port assignment to Uvicorn.
            kwargs = {}
            if port:
                kwargs["port"] = port

            await Server(Config(self.app, host, **kwargs)).serve()
        else:
            # NOTE: There might be a nice middleware way to do this.
            # Here, we need to introspect the route to determine how to
            # interact on the command line.
            routes = [route for route in self.app.routes if isinstance(route, APIRoute)]
            assert len(routes) == 1, (
                f"Multiple routes not supported in CLI mode: {routes}"
            )
            # For each of the route's handler's parameters, prompt the user.
            (route,) = routes
            assert isinstance(route, APIRoute)
            arguments = {
                param.name: input(f'Query parameter "{param.name}": ')
                for param in route.dependant.query_params
            }
            # Run the handler directly with the inputs. I think this only
            # supports string inputs for now.
            value = await route.endpoint(**arguments)
            # If the output is a StreamingResponse, ask the user where to save
            # it.
            if isinstance(value, FileResponse):
                path = input("Response path: ")
                shutil.copyfile(value.path, path)
            else:
                print(value)

    def run(self, *args, **kwargs):
        """Start the server."""
        asyncio.run(*args, **kwargs)
