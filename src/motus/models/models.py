from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Iterable

from .. import ModelClient
from ..tools.core.tool import DictTools, Tools


class Role(StrEnum):
    DEVELOPER = auto()
    USER = auto()
    ASSISTANT = auto()


class Message:
    role: Role


class ChatCompletionsModel:
    def __init__(self, client: ModelClient, model: str, instructions: Any) -> None:
        self.client = client
        self.model = model

    def __call__(
        self, messages: Iterable[Message], tools: Tools = DictTools({})
    ) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=[
                {
                    "type": "function",
                    "function": {
                        "name": t,
                        "description": tools[t].description,
                        "parameters": tools[t].json_schema,
                    },
                }
                for t in tools
            ],
        )


class Image:
    pass


@dataclass
class ImagesGenerateModel:
    client: ModelClient
    model: str | None = None

    def __call__(self, prompt: Any) -> Image:
        self.client.images.generate(prompt=prompt, model=self.model)
