"""
Agent prompts for ActusBot.

This module exports all agent prompts and the agent_prompt mapping.
"""

from .debugger.prompt import DEBUGGER_PROMPT
from .executor.prompt import EXECUTOR_PROMPT
from .file_verifier.prompt import FILE_VERIFIER_PROMPT
from .main_agent.prompt import MAIN_AGENT_PROMPT
from .syntax_verifier.prompt import VERIFIER_PROMPT

# Mapping from agent name to system prompt
# 4 specialized agents: executor, debugger, syntax_verifier, file_verifier
agent_prompt = {
    "executor": EXECUTOR_PROMPT,
    "debugger": DEBUGGER_PROMPT,
    "syntax_verifier": VERIFIER_PROMPT,
    "file_verifier": FILE_VERIFIER_PROMPT,
}

__all__ = [
    "MAIN_AGENT_PROMPT",
    "EXECUTOR_PROMPT",
    "DEBUGGER_PROMPT",
    "VERIFIER_PROMPT",
    "FILE_VERIFIER_PROMPT",
    "agent_prompt",
]
