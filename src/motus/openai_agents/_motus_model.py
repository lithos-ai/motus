"""Motus Model wrappers — transparent pass-through subclasses.

Each class inherits from the corresponding OAI SDK Model and does nothing
extra. They exist as stable extension points for future motus features:
prompt caching, dynamic routing, cost control, TTFT measurement, etc.

Usage:
    # These are used internally by MotusModelProvider.
    # Users don't need to instantiate them directly.
    model = MotusChatCompletionsModel(model="gpt-4o", openai_client=client)
    model = MotusResponsesModel(model="gpt-4o", openai_client=client)
"""

from __future__ import annotations

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel


class MotusChatCompletionsModel(OpenAIChatCompletionsModel):
    """Transparent wrapper over OpenAIChatCompletionsModel.

    Future hooks: prompt caching, dynamic routing, cost control, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MotusResponsesModel(OpenAIResponsesModel):
    """Transparent wrapper over OpenAIResponsesModel.

    Future hooks: prompt caching, dynamic routing, cost control, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Optional: LiteLLM support (only available if litellm is installed)
try:
    from agents.extensions.models.litellm_model import LitellmModel

    class MotusLitellmModel(LitellmModel):
        """Transparent wrapper over LitellmModel.

        Future hooks: prompt caching, dynamic routing, cost control, etc.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

except ImportError:
    MotusLitellmModel = None  # type: ignore[assignment,misc]


# Backwards compat alias
MotusModel = MotusChatCompletionsModel
