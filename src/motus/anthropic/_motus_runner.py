"""Motus-instrumented subclasses of Anthropic SDK Beta Tool Runners.

Each subclass overrides _handle_request() and _generate_tool_call_response()
to emit model_call and tool_call OTel spans via the motus tracer.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from anthropic.lib.tools._beta_runner import (
    BetaAsyncStreamingToolRunner,
    BetaAsyncToolRunner,
    BetaStreamingToolRunner,
    BetaToolRunner,
)
from opentelemetry import context as otel_context
from opentelemetry import trace
from typing_extensions import override

from ._motus_tracing import (
    _now_ns,
    emit_model_span,
    emit_tool_span,
)

if TYPE_CHECKING:
    from anthropic.lib.streaming._beta_messages import (
        BetaAsyncMessageStream,
        BetaMessageStream,
    )
    from anthropic.types.beta import BetaMessageParam
    from anthropic.types.beta.beta_tool_result_block_param import (
        BetaToolResultBlockParam,
    )
    from anthropic.types.beta.parsed_beta_message import (
        ParsedBetaMessage,
        ResponseFormatT,
    )

log = logging.getLogger(__name__)


# ── Shared tracing logic ──


class _MotusTracingMixin:
    """Mixin that provides tracing helpers for all 4 runner variants.

    Expects the host class to set self._parent_span and self._parent_ctx
    in __init__.
    """

    _parent_span: trace.Span | None
    _parent_ctx: otel_context.Context | None

    def _init_tracing(
        self, parent_span: trace.Span | None = None, **kwargs: Any
    ) -> None:
        self._parent_span = parent_span
        if parent_span is not None:
            self._parent_ctx = trace.set_span_in_context(parent_span)
        else:
            self._parent_ctx = None

    def _emit_model_span(self, message: Any, start_ns: int, end_ns: int) -> None:
        emit_model_span(
            message=message,
            model=self._params.get("model", "claude"),
            input_messages=list(self._params.get("messages", [])),
            start_ns=start_ns,
            end_ns=end_ns,
            parent_context=self._parent_ctx,
        )

    def _process_tool_calls_sync(self) -> list[Any] | None:
        """Execute tool calls and emit OTel spans. Returns tool result list or None."""
        content = self._get_last_assistant_message_content()
        if not content:
            return None

        tool_use_blocks = [block for block in content if block.type == "tool_use"]
        if not tool_use_blocks:
            return None

        results: list[BetaToolResultBlockParam] = []

        for tool_use in tool_use_blocks:
            tool = self._tools_by_name.get(tool_use.name)
            start_ns = _now_ns()
            error_msg: str | None = None
            tool_output: Any = None

            if tool is None:
                warnings.warn(
                    f"Tool '{tool_use.name}' not found in tool runner. "
                    f"Available tools: {list(self._tools_by_name.keys())}. "
                    f"If using a raw tool definition, handle the tool call manually "
                    f"and use `append_messages()` to add the result. Otherwise, pass "
                    f"the tool using `beta_tool(func)` or a `@beta_tool` decorated "
                    f"function.",
                    UserWarning,
                    stacklevel=3,
                )
                error_msg = f"Tool '{tool_use.name}' not found"
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Error: {error_msg}",
                        "is_error": True,
                    }
                )
            else:
                try:
                    tool_output = tool.call(tool_use.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_output,
                        }
                    )
                except Exception as exc:
                    log.exception(
                        f"Error occurred while calling tool: {tool.name}",
                        exc_info=exc,
                    )
                    error_msg = repr(exc)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": error_msg,
                            "is_error": True,
                        }
                    )

            end_ns = _now_ns()
            emit_tool_span(
                tool_name=tool_use.name,
                tool_input=tool_use.input,
                tool_output=tool_output,
                start_ns=start_ns,
                end_ns=end_ns,
                parent_context=self._parent_ctx,
                error=error_msg,
            )

        return results

    async def _process_tool_calls_async(self) -> list[Any] | None:
        """Async version: execute tool calls and emit OTel spans."""
        content = await self._get_last_assistant_message_content()
        if not content:
            return None

        tool_use_blocks = [block for block in content if block.type == "tool_use"]
        if not tool_use_blocks:
            return None

        results: list[BetaToolResultBlockParam] = []

        for tool_use in tool_use_blocks:
            tool = self._tools_by_name.get(tool_use.name)
            start_ns = _now_ns()
            error_msg: str | None = None
            tool_output: Any = None

            if tool is None:
                warnings.warn(
                    f"Tool '{tool_use.name}' not found in tool runner. "
                    f"Available tools: {list(self._tools_by_name.keys())}. "
                    f"If using a raw tool definition, handle the tool call manually "
                    f"and use `append_messages()` to add the result. Otherwise, pass "
                    f"the tool using `beta_async_tool(func)` or a "
                    f"`@beta_async_tool` decorated function.",
                    UserWarning,
                    stacklevel=3,
                )
                error_msg = f"Tool '{tool_use.name}' not found"
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Error: {error_msg}",
                        "is_error": True,
                    }
                )
            else:
                try:
                    tool_output = await tool.call(tool_use.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_output,
                        }
                    )
                except Exception as exc:
                    log.exception(
                        f"Error occurred while calling tool: {tool.name}",
                        exc_info=exc,
                    )
                    error_msg = repr(exc)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": error_msg,
                            "is_error": True,
                        }
                    )

            end_ns = _now_ns()
            emit_tool_span(
                tool_name=tool_use.name,
                tool_input=tool_use.input,
                tool_output=tool_output,
                start_ns=start_ns,
                end_ns=end_ns,
                parent_context=self._parent_ctx,
                error=error_msg,
            )

        return results


# ── Sync, non-streaming ──


class MotusBetaToolRunner(_MotusTracingMixin, BetaToolRunner):
    """BetaToolRunner with motus OTel tracing."""

    def __init__(
        self,
        *,
        parent_span: trace.Span | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._init_tracing(parent_span=parent_span)

    @override
    @contextmanager
    def _handle_request(self) -> Iterator[ParsedBetaMessage[ResponseFormatT]]:
        start_ns = _now_ns()
        with super()._handle_request() as item:
            yield item
        end_ns = _now_ns()
        self._emit_model_span(self._get_last_message(), start_ns, end_ns)

    @override
    def _generate_tool_call_response(self) -> BetaMessageParam | None:
        results = self._process_tool_calls_sync()
        if results is None:
            return None
        return {"role": "user", "content": results}


# ── Sync, streaming ──


class MotusBetaStreamingToolRunner(_MotusTracingMixin, BetaStreamingToolRunner):
    """BetaStreamingToolRunner with motus OTel tracing."""

    def __init__(
        self,
        *,
        parent_span: trace.Span | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._init_tracing(parent_span=parent_span)

    @override
    @contextmanager
    def _handle_request(self) -> Iterator[BetaMessageStream[ResponseFormatT]]:
        start_ns = _now_ns()
        with super()._handle_request() as stream:
            yield stream
        end_ns = _now_ns()
        self._emit_model_span(self._get_last_message(), start_ns, end_ns)

    @override
    def _generate_tool_call_response(self) -> BetaMessageParam | None:
        results = self._process_tool_calls_sync()
        if results is None:
            return None
        return {"role": "user", "content": results}


# ── Async, non-streaming ──


class MotusBetaAsyncToolRunner(_MotusTracingMixin, BetaAsyncToolRunner):
    """BetaAsyncToolRunner with motus OTel tracing."""

    def __init__(
        self,
        *,
        parent_span: trace.Span | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._init_tracing(parent_span=parent_span)

    @override
    @asynccontextmanager
    async def _handle_request(
        self,
    ) -> AsyncIterator[ParsedBetaMessage[ResponseFormatT]]:
        start_ns = _now_ns()
        async with super()._handle_request() as item:
            yield item
        end_ns = _now_ns()
        message = await self._get_last_message()
        self._emit_model_span(message, start_ns, end_ns)

    @override
    async def _generate_tool_call_response(self) -> BetaMessageParam | None:
        results = await self._process_tool_calls_async()
        if results is None:
            return None
        return {"role": "user", "content": results}


# ── Async, streaming ──


class MotusBetaAsyncStreamingToolRunner(
    _MotusTracingMixin, BetaAsyncStreamingToolRunner
):
    """BetaAsyncStreamingToolRunner with motus OTel tracing."""

    def __init__(
        self,
        *,
        parent_span: trace.Span | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._init_tracing(parent_span=parent_span)

    @override
    @asynccontextmanager
    async def _handle_request(
        self,
    ) -> AsyncIterator[BetaAsyncMessageStream[ResponseFormatT]]:
        start_ns = _now_ns()
        async with super()._handle_request() as stream:
            yield stream
        end_ns = _now_ns()
        message = await self._get_last_message()
        self._emit_model_span(message, start_ns, end_ns)

    @override
    async def _generate_tool_call_response(self) -> BetaMessageParam | None:
        results = await self._process_tool_calls_async()
        if results is None:
            return None
        return {"role": "user", "content": results}
