import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any, Self

from motus.runtime.agent_task import agent_task
from motus.runtime.types import TOOL_CALL


class Tool(ABC):
    """Abstract base class for all tools.

    Architecture::

        __call__(args: str)          ← base class default (json.loads), FunctionTool overrides with coercion
          └→ _execute(**kwargs)      ← shared: @agent_task + guardrails + error handling + _serialize
               └→ _invoke(**kwargs)  ← abstract: just "call the function"
    """

    def __init__(
        self,
        name: str,
        description: str | None,
        json_schema: dict,
        input_guardrails: list | None = None,
        output_guardrails: list | None = None,
        requires_approval: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.json_schema = json_schema
        self._input_guardrails = input_guardrails or []
        self._output_guardrails = output_guardrails or []

        self.requires_approval = False
        if requires_approval:
            self.set_requires_approval(True)

    def _make_approval_guardrail(self):
        tool_self = self  # closure capture

        async def _builtin_approval_guardrail(**kwargs):
            from motus.guardrails import ToolRejected
            from motus.serve.interrupt import interrupt

            decision = await interrupt(
                {
                    "type": "tool_approval",
                    "tool_name": tool_self.name,
                    "tool_args": kwargs,
                }
            )
            if not decision.get("approved"):
                raise ToolRejected(f"User rejected {tool_self.name}")

        return _builtin_approval_guardrail

    def set_requires_approval(self, value: bool) -> None:
        """Enable or disable the approval gate on this tool (idempotent)."""
        if value == self.requires_approval:
            return
        self.requires_approval = value
        if value:
            self._input_guardrails.insert(0, self._make_approval_guardrail())
        else:
            self._input_guardrails = [
                g
                for g in self._input_guardrails
                if getattr(g, "__name__", "") != "_builtin_approval_guardrail"
            ]

    def __call__(self, args: str):
        """Parse JSON args, then delegate to _execute."""
        try:
            parsed = json.loads(args) if args and args.strip() else {}
        except json.JSONDecodeError as e:
            # Return error to the model so it can retry with valid JSON
            return self._execute(
                __json_error=f"Invalid JSON in tool arguments: {e}. "
                f"Raw args: {args[:200]}. Please retry with valid JSON."
            )
        return self._execute(**parsed)

    @agent_task(task_type=TOOL_CALL)
    async def _execute(self, **kwargs) -> str:
        """Shared execution boundary — tracing + guardrails + error handling.

        Subclasses should NOT override this method.
        """
        try:
            # Handle JSON parse errors from __call__
            if "__json_error" in kwargs:
                return self._serialize(kwargs["__json_error"])

            if self._input_guardrails:
                from motus.guardrails import run_tool_input_guardrails

                kwargs = await run_tool_input_guardrails(self._input_guardrails, kwargs)

            result = await self._invoke(**kwargs)

            if self._output_guardrails:
                from motus.guardrails import run_tool_output_guardrails

                result = await run_tool_output_guardrails(
                    self._output_guardrails, result
                )
        except Exception as e:
            return json.dumps({"error": str(e)})

        return self._serialize(result)

    @abstractmethod
    async def _invoke(self, **kwargs) -> Any:
        """Dispatch the actual tool call. Subclasses must implement."""
        ...

    def _serialize(self, result: Any) -> str:
        """Encode the result to a string. Override for custom serialization."""
        if isinstance(result, str):
            return result
        return json.dumps(result)


class Tools(Mapping[str, Tool]):
    @abstractmethod
    def __getitem__(self, key: str) -> Tool: ...

    @abstractmethod
    def __iter__(self) -> Iterator[str]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    # --- Lifecycle ---

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    # --- Context managers ---

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()


class DictTools(Tools):
    def __init__(
        self,
        tools: Mapping[str, Tool],
        owned_sessions: list | None = None,
    ) -> None:
        self._tools = dict(tools)  # mutable — lazy connect appends tools
        self._owned_sessions: list = owned_sessions or []

    def __getitem__(self, key: str) -> Tool:
        return self._tools[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def update(self, other: Mapping[str, "Tool"]) -> None:
        """Merge tools from another mapping."""
        self._tools.update(other)

    async def _connect_mcp_sessions(self) -> None:
        """Connect owned (lazy) MCP sessions and register their tools.

        Called by ``Agent._execute`` before the first ``_run()``.
        """
        if not self._owned_sessions:
            return

        from .normalize import _normalize_mcp_session

        for session in self._owned_sessions:
            await session._ensure_connected()
            # Another task may have already registered this session's tools.
            if not any(
                getattr(t, "_mcp_session", None) is session
                for t in self._tools.values()
            ):
                _normalize_mcp_session(session, self._tools)

    def close(self) -> None:
        for session in self._owned_sessions:
            session.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    async def aclose(self) -> None:
        for session in self._owned_sessions:
            await session.aclose()
