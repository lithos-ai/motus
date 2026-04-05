"""Motus Model Providers — transparent wrappers over OAI SDK providers.

Each provider returns the corresponding Motus-wrapped Model class.
"""

from __future__ import annotations

from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider as _MultiProvider
from agents.models.openai_provider import OpenAIProvider

from ._motus_model import (
    MotusChatCompletionsModel,
    MotusLitellmModel,
    MotusResponsesModel,
)


class MotusOpenAIProvider(OpenAIProvider):
    """Wraps OpenAIProvider, returns Motus-wrapped Model instances.

    Inherits the SDK's default use_responses setting (True since v0.0.16),
    so both /v1/chat/completions and /v1/responses are supported.  Users
    can still pass ``use_responses=False`` to force chat/completions only.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_model(self, model_name: str | None) -> Model:
        model = super().get_model(model_name)

        # Already a Motus model
        if isinstance(model, (MotusChatCompletionsModel, MotusResponsesModel)):
            return model

        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from agents.models.openai_responses import OpenAIResponsesModel

        if isinstance(model, OpenAIChatCompletionsModel):
            return MotusChatCompletionsModel(
                model=model.model, openai_client=model._client
            )
        elif isinstance(model, OpenAIResponsesModel):
            return MotusResponsesModel(
                model=model.model,
                openai_client=model._client,
                model_is_explicit=model._model_is_explicit,
            )

        # Unknown model type (e.g. websocket) — return as-is
        return model


try:
    from agents.extensions.models.litellm_provider import (
        LitellmProvider as _OriginalLitellmProvider,
    )

    class MotusLitellmProvider(_OriginalLitellmProvider):
        """Wraps LitellmProvider, returns MotusLitellmModel."""

        def get_model(self, model_name: str | None) -> Model:
            if MotusLitellmModel is None:
                raise ImportError("litellm is not installed")
            from agents.models.openai_provider import get_default_model

            return MotusLitellmModel(model=model_name or get_default_model())

except ImportError:
    MotusLitellmProvider = None  # type: ignore[assignment,misc]


class MotusMultiProvider(_MultiProvider):
    """Inherits MultiProvider, upgrades internal providers to Motus versions.

    All routing logic, aclose(), provider_map etc. are inherited as-is.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # super() created self.openai_provider with all params (incl. websocket_base_url).
        # Upgrade its class so get_model() returns Motus-wrapped models.
        self.openai_provider.__class__ = MotusOpenAIProvider

    def _create_fallback_provider(self, prefix: str) -> ModelProvider:
        if prefix == "litellm" and MotusLitellmProvider is not None:
            return MotusLitellmProvider()
        return super()._create_fallback_provider(prefix)
