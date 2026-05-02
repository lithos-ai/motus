"""
Self-hosted chat client for OpenAI-compatible inference servers.

Currently supports:

- **sglang** — launch with::

      python -m sglang.launch_server --model-path /path/to/model --port 30000 ...

  Server exposes ``http://host:30000/v1/chat/completions`` (OpenAI-compatible).

- **vllm** — launch with::

      python -m vllm.entrypoints.openai.api_server --model /path/to/model --port 8000 ...

  Server exposes ``http://host:8000/v1/chat/completions`` (OpenAI-compatible).

Both engines speak the same OpenAI Chat Completions wire protocol, so this
class is a thin shim around :class:`OpenAIChatClient` with sane defaults for
local serving (no API key required, longer default timeout) plus engine-
agnostic helpers (health check, list_models).

Engine-specific knobs (sglang's ``regex`` / ``json_schema`` constraints,
vllm's ``min_p`` / ``repetition_penalty`` etc.) can be passed through the
standard OpenAI ``extra_body`` kwarg::

    client = SelfHostedChatClient(base_url="http://localhost:30000")
    await client.create(
        model="/path/to/Llama-3.1-8B-Instruct",
        messages=[ChatMessage.user_message("hi")],
        extra_body={"regex": r"\\d{4}-\\d{2}-\\d{2}"},  # sglang-only
    )

Why a single class instead of separate ``SglangChatClient`` and
``VllmChatClient``: 95% of the surface is identical. The engine differences
that exist (constraint-decoding params, server-management endpoints) are
small enough to dispatch on the ``engine`` field when needed. Splitting into
subclasses now would add boilerplate without payoff. If divergence grows
(e.g. one engine adds vision, the other gets streaming-only features), the
class can be split later — callers using the OpenAI-compatible subset will
not need to change.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

import httpx

from .openai_client import OpenAIChatClient

EngineName = Literal["sglang", "vllm", "auto"]


class SelfHostedChatClient(OpenAIChatClient):
    """OpenAI-compatible client targeting a locally hosted server.

    Args:
        base_url: Server base URL. Either with or without the ``/v1`` suffix
            is accepted; ``/v1`` is appended if missing.
            Examples: ``http://localhost:30000``, ``http://10.0.0.5:8000/v1``.
        api_key: Optional auth token. Self-hosted servers usually run without
            auth; we default to ``"EMPTY"`` because the underlying ``openai``
            SDK rejects an empty string. If your server runs with
            ``--api-key foo``, pass ``api_key="foo"`` here.
        engine: Identifier for the backend engine. Used only for
            engine-specific extras (currently none) and for telemetry. ``"auto"``
            (default) means the client makes no assumptions; pass ``"sglang"``
            or ``"vllm"`` if you want to lock the value down for downstream
            metrics.
        http_client: Optional ``httpx.AsyncClient`` for custom transport
            (test recording / replay).
        timeout: Per-request timeout in seconds. Defaults to 600s because
            cold-start prefill on large local models can be slow.
        **kwargs: Forwarded to :class:`AsyncOpenAI` (e.g. ``max_retries``).
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        engine: EngineName = "auto",
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 600.0,
        **kwargs: Any,
    ) -> None:
        if not base_url:
            raise ValueError(
                "SelfHostedChatClient requires base_url "
                "(e.g. http://localhost:30000)."
            )

        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/v1"):
            self._server_root = cleaned[:-3]
            normalized = cleaned
        else:
            self._server_root = cleaned
            normalized = cleaned + "/v1"

        super().__init__(
            api_key=api_key or os.environ.get("SELF_HOSTED_API_KEY", "EMPTY"),
            base_url=normalized,
            http_client=http_client,
            timeout=timeout,
            **kwargs,
        )
        self.engine: EngineName = engine

    # ------------------------------------------------------------------
    # Server-management helpers (not part of BaseChatClient)
    # ------------------------------------------------------------------

    async def health_check(self, timeout: float = 10.0) -> bool:
        """Return True if the server's ``/health`` endpoint returns 2xx.

        Both sglang and vllm expose ``/health``. Returns False on any
        connection error or non-2xx status.
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                resp = await c.get(f"{self._server_root}/health")
                return resp.is_success
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return the model IDs the server reports via ``/v1/models``.

        Useful when the served model name isn't known up front (e.g. sglang
        registers the model under the path passed to ``--model-path``, so the
        canonical name might be ``/home/foo/models/Llama-3.1-8B-Instruct``).
        """
        models = await self._client.models.list()
        return [m.id for m in models.data]

    async def resolve_model(self, hint: Optional[str] = None) -> str:
        """Pick a model id from ``/v1/models``.

        If ``hint`` is given, return the first id containing that substring
        (case-insensitive). Otherwise return the first id (most servers only
        host one model anyway).

        Raises ``RuntimeError`` if no models are registered or no id matches
        the hint.
        """
        ids = await self.list_models()
        if not ids:
            raise RuntimeError(
                f"No models reported by {self._server_root}/v1/models — is the "
                "server up and finished loading?"
            )
        if hint is None:
            return ids[0]
        h = hint.lower()
        for mid in ids:
            if h in mid.lower():
                return mid
        raise RuntimeError(
            f"No model id from {ids!r} contains hint {hint!r}."
        )
