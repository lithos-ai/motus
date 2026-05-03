"""Agent templates — preconfigured ReActAgent variants for common tasks."""

from ._project_context import build_system_prompt
from .coding_agent import CodingAgent

__all__ = [
    "CodingAgent",
    "build_system_prompt",
]
