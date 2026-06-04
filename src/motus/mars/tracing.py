from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraceTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    subagent_trace_file: str | None = None


class InputDelta(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    text: str | None = None
    input_ids: list[int] | None = None
    token_count: int | None = None


class TraceTurn(BaseModel):
    model_config = ConfigDict(extra="allow")

    turn_index: int
    input_delta: InputDelta
    output_tokens: int
    tools: list[TraceTool] = Field(default_factory=list)
    is_terminal: bool = False

    @model_validator(mode="after")
    def _validate_output_tokens(self) -> "TraceTurn":
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        return self


class AgentIdentity(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent_instance_id: str
    agent_class_id: str


class SystemPrompt(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = None
    input_ids: list[int] | None = None
    token_count: int | None = None


class AgentTrace(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_name: str = Field(alias="schema")
    trace_id: str
    source: dict[str, Any] = Field(default_factory=dict)
    tokenizer: dict[str, Any] = Field(default_factory=dict)
    agent: AgentIdentity
    system_prompt: SystemPrompt
    turns: list[TraceTurn]
    conversion: dict[str, Any] = Field(default_factory=dict)

    @property
    def schema(self) -> str:
        return self.schema_name

    @model_validator(mode="after")
    def _validate_turn_order(self) -> "AgentTrace":
        for expected, turn in enumerate(self.turns):
            if turn.turn_index != expected:
                raise ValueError(
                    f"turn_index mismatch at position {expected}: "
                    f"got {turn.turn_index}"
                )
        return self


def load_trace(path: str | Path) -> AgentTrace:
    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8") as f:
        return AgentTrace.model_validate(json.load(f))


def write_trace(trace: AgentTrace, path: str | Path) -> None:
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(
            trace.model_dump(mode="json", by_alias=True, exclude_none=True),
            f,
            indent=2,
        )
        f.write("\n")
