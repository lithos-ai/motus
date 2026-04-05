"""Motus-instrumented subclasses of Anthropic SDK Beta Tool Runners.

Each subclass overrides _handle_request() and _generate_tool_call_response()
to emit model_call and tool_call spans into motus TraceManager.
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
from typing_extensions import override

from ._motus_tracing import (
    _now_us,
    build_model_call_meta,
    build_tool_call_meta,
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

    from motus.runtime.tracing import TraceManager

log = logging.getLogger(__name__)


# ── Sync, non-streaming ──


class MotusBetaToolRunner(BetaToolRunner):
    """BetaToolRunner with motus tracing spans."""

    def __init__(
        self,
        *,
        trace_manager: TraceManager | None = None,
        parent_task_id: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tm = trace_manager
        self._motus_parent = parent_task_id

    @override
    @contextmanager
    def _handle_request(self) -> Iterator[ParsedBetaMessage[ResponseFormatT]]:
        start_us = _now_us()
        task_id = self._tm.allocate_external_task_id() if self._tm else None
        input_messages = list(self._params.get("messages", []))

        with super()._handle_request() as item:
            yield item

        end_us = _now_us()
        if self._tm and task_id is not None:
            message = self._get_last_message()
            meta = build_model_call_meta(
                message=message,
                model=self._params.get("model", "claude"),
                input_messages=input_messages,
                start_us=start_us,
                end_us=end_us,
                parent=self._motus_parent,
            )
            self._tm.ingest_external_span(meta, task_id=task_id)

    @override
    def _generate_tool_call_response(self) -> BetaMessageParam | None:
        content = self._get_last_assistant_message_content()
        if not content:
            return None

        tool_use_blocks = [block for block in content if block.type == "tool_use"]
        if not tool_use_blocks:
            return None

        results: list[BetaToolResultBlockParam] = []

        for tool_use in tool_use_blocks:
            tool = self._tools_by_name.get(tool_use.name)
            task_id = self._tm.allocate_external_task_id() if self._tm else None
            start_us = _now_us()
            error_msg: str | None = None
            tool_output: Any = None

            if tool is None:
                warnings.warn(
                    f"Tool '{tool_use.name}' not found in tool runner. "
                    f"Available tools: {list(self._tools_by_name.keys())}. "
                    f"If using a raw tool definition, handle the tool call manually and use `append_messages()` to add the result. "
                    f"Otherwise, pass the tool using `beta_tool(func)` or a `@beta_tool` decorated function.",
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
                        f"Error occurred while calling tool: {tool.name}", exc_info=exc
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

            end_us = _now_us()
            if self._tm and task_id is not None:
                meta = build_tool_call_meta(
                    tool_name=tool_use.name,
                    tool_input=tool_use.input,
                    tool_output=tool_output,
                    start_us=start_us,
                    end_us=end_us,
                    parent=self._motus_parent,
                    error=error_msg,
                )
                self._tm.ingest_external_span(meta, task_id=task_id)

        return {"role": "user", "content": results}


# ── Sync, streaming ──


class MotusBetaStreamingToolRunner(BetaStreamingToolRunner):
    """BetaStreamingToolRunner with motus tracing spans."""

    def __init__(
        self,
        *,
        trace_manager: TraceManager | None = None,
        parent_task_id: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tm = trace_manager
        self._motus_parent = parent_task_id

    @override
    @contextmanager
    def _handle_request(self) -> Iterator[BetaMessageStream[ResponseFormatT]]:
        start_us = _now_us()
        task_id = self._tm.allocate_external_task_id() if self._tm else None
        input_messages = list(self._params.get("messages", []))

        with super()._handle_request() as stream:
            yield stream

        end_us = _now_us()
        if self._tm and task_id is not None:
            message = self._get_last_message()
            meta = build_model_call_meta(
                message=message,
                model=self._params.get("model", "claude"),
                input_messages=input_messages,
                start_us=start_us,
                end_us=end_us,
                parent=self._motus_parent,
            )
            self._tm.ingest_external_span(meta, task_id=task_id)

    @override
    def _generate_tool_call_response(self) -> BetaMessageParam | None:
        # Identical to MotusBetaToolRunner — sync tool calls
        content = self._get_last_assistant_message_content()
        if not content:
            return None

        tool_use_blocks = [block for block in content if block.type == "tool_use"]
        if not tool_use_blocks:
            return None

        results: list[BetaToolResultBlockParam] = []

        for tool_use in tool_use_blocks:
            tool = self._tools_by_name.get(tool_use.name)
            task_id = self._tm.allocate_external_task_id() if self._tm else None
            start_us = _now_us()
            error_msg: str | None = None
            tool_output: Any = None

            if tool is None:
                warnings.warn(
                    f"Tool '{tool_use.name}' not found in tool runner. "
                    f"Available tools: {list(self._tools_by_name.keys())}. "
                    f"If using a raw tool definition, handle the tool call manually and use `append_messages()` to add the result. "
                    f"Otherwise, pass the tool using `beta_tool(func)` or a `@beta_tool` decorated function.",
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
                        f"Error occurred while calling tool: {tool.name}", exc_info=exc
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

            end_us = _now_us()
            if self._tm and task_id is not None:
                meta = build_tool_call_meta(
                    tool_name=tool_use.name,
                    tool_input=tool_use.input,
                    tool_output=tool_output,
                    start_us=start_us,
                    end_us=end_us,
                    parent=self._motus_parent,
                    error=error_msg,
                )
                self._tm.ingest_external_span(meta, task_id=task_id)

        return {"role": "user", "content": results}


# ── Async, non-streaming ──


class MotusBetaAsyncToolRunner(BetaAsyncToolRunner):
    """BetaAsyncToolRunner with motus tracing spans."""

    def __init__(
        self,
        *,
        trace_manager: TraceManager | None = None,
        parent_task_id: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tm = trace_manager
        self._motus_parent = parent_task_id

    @override
    @asynccontextmanager
    async def _handle_request(
        self,
    ) -> AsyncIterator[ParsedBetaMessage[ResponseFormatT]]:
        start_us = _now_us()
        task_id = self._tm.allocate_external_task_id() if self._tm else None
        input_messages = list(self._params.get("messages", []))

        async with super()._handle_request() as item:
            yield item

        end_us = _now_us()
        if self._tm and task_id is not None:
            message = await self._get_last_message()
            meta = build_model_call_meta(
                message=message,
                model=self._params.get("model", "claude"),
                input_messages=input_messages,
                start_us=start_us,
                end_us=end_us,
                parent=self._motus_parent,
            )
            self._tm.ingest_external_span(meta, task_id=task_id)

    @override
    async def _generate_tool_call_response(self) -> BetaMessageParam | None:
        content = await self._get_last_assistant_message_content()
        if not content:
            return None

        tool_use_blocks = [block for block in content if block.type == "tool_use"]
        if not tool_use_blocks:
            return None

        results: list[BetaToolResultBlockParam] = []

        for tool_use in tool_use_blocks:
            tool = self._tools_by_name.get(tool_use.name)
            task_id = self._tm.allocate_external_task_id() if self._tm else None
            start_us = _now_us()
            error_msg: str | None = None
            tool_output: Any = None

            if tool is None:
                warnings.warn(
                    f"Tool '{tool_use.name}' not found in tool runner. "
                    f"Available tools: {list(self._tools_by_name.keys())}. "
                    f"If using a raw tool definition, handle the tool call manually and use `append_messages()` to add the result. "
                    f"Otherwise, pass the tool using `beta_async_tool(func)` or a `@beta_async_tool` decorated function.",
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
                        f"Error occurred while calling tool: {tool.name}", exc_info=exc
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

            end_us = _now_us()
            if self._tm and task_id is not None:
                meta = build_tool_call_meta(
                    tool_name=tool_use.name,
                    tool_input=tool_use.input,
                    tool_output=tool_output,
                    start_us=start_us,
                    end_us=end_us,
                    parent=self._motus_parent,
                    error=error_msg,
                )
                self._tm.ingest_external_span(meta, task_id=task_id)

        return {"role": "user", "content": results}


# ── Async, streaming ──


class MotusBetaAsyncStreamingToolRunner(BetaAsyncStreamingToolRunner):
    """BetaAsyncStreamingToolRunner with motus tracing spans."""

    def __init__(
        self,
        *,
        trace_manager: TraceManager | None = None,
        parent_task_id: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tm = trace_manager
        self._motus_parent = parent_task_id

    @override
    @asynccontextmanager
    async def _handle_request(
        self,
    ) -> AsyncIterator[BetaAsyncMessageStream[ResponseFormatT]]:
        start_us = _now_us()
        task_id = self._tm.allocate_external_task_id() if self._tm else None
        input_messages = list(self._params.get("messages", []))

        async with super()._handle_request() as stream:
            yield stream

        end_us = _now_us()
        if self._tm and task_id is not None:
            message = await self._get_last_message()
            meta = build_model_call_meta(
                message=message,
                model=self._params.get("model", "claude"),
                input_messages=input_messages,
                start_us=start_us,
                end_us=end_us,
                parent=self._motus_parent,
            )
            self._tm.ingest_external_span(meta, task_id=task_id)

    @override
    async def _generate_tool_call_response(self) -> BetaMessageParam | None:
        # Identical to MotusBetaAsyncToolRunner — async tool calls
        content = await self._get_last_assistant_message_content()
        if not content:
            return None

        tool_use_blocks = [block for block in content if block.type == "tool_use"]
        if not tool_use_blocks:
            return None

        results: list[BetaToolResultBlockParam] = []

        for tool_use in tool_use_blocks:
            tool = self._tools_by_name.get(tool_use.name)
            task_id = self._tm.allocate_external_task_id() if self._tm else None
            start_us = _now_us()
            error_msg: str | None = None
            tool_output: Any = None

            if tool is None:
                warnings.warn(
                    f"Tool '{tool_use.name}' not found in tool runner. "
                    f"Available tools: {list(self._tools_by_name.keys())}. "
                    f"If using a raw tool definition, handle the tool call manually and use `append_messages()` to add the result. "
                    f"Otherwise, pass the tool using `beta_async_tool(func)` or a `@beta_async_tool` decorated function.",
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
                        f"Error occurred while calling tool: {tool.name}", exc_info=exc
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

            end_us = _now_us()
            if self._tm and task_id is not None:
                meta = build_tool_call_meta(
                    tool_name=tool_use.name,
                    tool_input=tool_use.input,
                    tool_output=tool_output,
                    start_us=start_us,
                    end_us=end_us,
                    parent=self._motus_parent,
                    error=error_msg,
                )
                self._tm.ingest_external_span(meta, task_id=task_id)

        return {"role": "user", "content": results}
