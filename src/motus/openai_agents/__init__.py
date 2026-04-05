"""OpenAI Agents SDK compatibility layer for motus.

Re-exports the real openai-agents package. motus intercepts at two levels:
  - Tracing: MotusTracingProcessor bridges OAI SDK spans → motus TraceManager
  - Execution: MotusModel + tool wrapping sit in the call path (transparent
    pass-through now, future hooks for caching/routing/optimization)

Usage::

    from motus.openai_agents import Agent, Runner, function_tool
"""

from __future__ import annotations

import atexit
import logging

# Re-export everything from real OAI SDK (Agent, Runner, function_tool, etc.)
try:
    from agents import *  # noqa: F401,F403
    from agents import RunConfig as _RunConfig

    # Override Runner with MotusRunner
    from agents import Runner as _OriginalRunner
except ImportError as exc:
    raise ImportError(
        "motus.openai_agents requires the OpenAI Agents SDK. "
        "Install it with: uv pip install motus[openai-agents]"
    ) from exc

from ._motus_model import (
    MotusChatCompletionsModel,
    MotusResponsesModel,
)
from ._motus_provider import (
    MotusLitellmProvider,
    MotusMultiProvider,
    MotusOpenAIProvider,
)
from ._motus_tools import _wrap_tools_for_motus
from ._motus_tracing import MotusTracingProcessor

# Override OAI SDK classes so that imports from motus.openai_agents
# transparently return our Motus wrappers.
OpenAIChatCompletionsModel = MotusChatCompletionsModel  # noqa: F811
OpenAIResponsesModel = MotusResponsesModel  # noqa: F811
OpenAIProvider = MotusOpenAIProvider  # noqa: F811
MultiProvider = MotusMultiProvider  # noqa: F811
if MotusLitellmProvider is not None:
    LitellmProvider = MotusLitellmProvider  # noqa: F811

logger = logging.getLogger("AgentTracer")

_tracer = None


def register_tracing() -> None:  # noqa: C901
    """Register MotusTracingProcessor with OAI SDK (idempotent).

    Creates a standalone TraceManager (no runtime needed) and replaces
    OAI SDK's default processors with MotusTracingProcessor.
    """
    global _tracer
    if _tracer is not None:
        return

    try:
        from agents import set_trace_processors

        from motus.runtime.tracing import TraceManager

        _tracer = TraceManager()
        processor = MotusTracingProcessor(_tracer)
        # Replace default processors (which include BackendSpanExporter
        # that tries to POST traces to api.openai.com — fails with 401
        # when using non-OpenAI API keys like OpenRouter/Anthropic)
        set_trace_processors([processor])
        atexit.register(_auto_export)
        logger.debug("Registered MotusTracingProcessor with OAI SDK")
    except Exception as e:
        logger.debug(f"Could not register MotusTracingProcessor: {e}")


def get_tracer():
    """Get the TraceManager instance (call register_tracing() first)."""
    return _tracer


def _auto_export():
    """Auto-export traces on process exit (registered via atexit)."""
    if _tracer and _tracer.task_meta and _tracer.config.export_enabled:
        _tracer.export_trace()


class Runner:
    """Runner that intercepts LLM + tool calls via motus."""

    @classmethod
    async def run(cls, starting_agent, input="", *, run_config=None, **kwargs):
        register_tracing()
        run_config = _ensure_motus_config(run_config)
        _wrap_tools_for_motus(starting_agent)
        try:
            return await _OriginalRunner.run(
                starting_agent, input, run_config=run_config, **kwargs
            )
        finally:
            if _tracer is not None:
                _tracer.close()

    @classmethod
    def run_sync(cls, starting_agent, input="", *, run_config=None, **kwargs):
        register_tracing()
        run_config = _ensure_motus_config(run_config)
        _wrap_tools_for_motus(starting_agent)
        try:
            return _OriginalRunner.run_sync(
                starting_agent, input, run_config=run_config, **kwargs
            )
        finally:
            if _tracer is not None:
                _tracer.close()

    @classmethod
    def run_streamed(cls, starting_agent, input="", *, run_config=None, **kwargs):
        register_tracing()
        run_config = _ensure_motus_config(run_config)
        _wrap_tools_for_motus(starting_agent)
        return _OriginalRunner.run_streamed(
            starting_agent, input, run_config=run_config, **kwargs
        )


def _ensure_motus_config(run_config):
    """Inject MotusOpenAIProvider as the default model_provider.

    Replaces the default MultiProvider/OpenAIProvider with MotusOpenAIProvider.
    If the user has already set a Motus provider or a custom provider, keep it.
    """
    if run_config is None:
        return _RunConfig(model_provider=MotusOpenAIProvider())
    if hasattr(run_config, "model_provider") and run_config.model_provider is not None:
        provider = run_config.model_provider
        # Already a Motus provider — keep it
        if isinstance(provider, (MotusOpenAIProvider, MotusMultiProvider)):
            return run_config
        if MotusLitellmProvider is not None and isinstance(
            provider, MotusLitellmProvider
        ):
            return run_config
        # Default MultiProvider — upgrade internal openai_provider in-place
        from agents.models.multi_provider import MultiProvider
        from agents.models.openai_provider import OpenAIProvider as _OrigOpenAIProvider

        if isinstance(provider, MultiProvider):
            provider.__class__ = MotusMultiProvider
            provider.openai_provider.__class__ = MotusOpenAIProvider
            return run_config
        # Default OpenAIProvider — upgrade in-place to preserve configuration
        if isinstance(provider, _OrigOpenAIProvider):
            provider.__class__ = MotusOpenAIProvider
            return run_config
        # Custom user provider — keep it
        return run_config
    run_config.model_provider = MotusOpenAIProvider()
    return run_config


# Auto-register tracing on import — ensures BackendSpanExporter is replaced
# before any `with trace(...)` or `Runner.run()` call.
register_tracing()
