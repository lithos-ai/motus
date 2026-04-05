"""
Summarization utilities for the memory system.

Provides functions to summarize chat message history into SummarizationResult objects
that can be stored in the memory system.

Usage:
    from motus.memory.summarization import get_summarization_fn

    # Create a summarization function from a model name
    summarize = get_summarization_fn("gpt-4o-mini")

    # Summarize messages into a SummarizationResult
    result = summarize(messages)

    # Access key_name, description, and entry
    print(result.key_name)       # e.g., "auth-setup"
    print(result.description)    # e.g., "Discussion about OAuth2 setup"
    print(result.entry.content)  # Full summary

    # Store in memory
    memory.remember_short_term_promotable(result.key_name, result.entry)
"""

from .summarizer import (
    SummarizationOutput,
    SummarizationResult,
    get_summarization_fn,
    get_summarization_fn_async,
    summarize_messages,
    summarize_messages_async,
)

__all__ = [
    "get_summarization_fn",
    "get_summarization_fn_async",
    "summarize_messages",
    "summarize_messages_async",
    "SummarizationOutput",
    "SummarizationResult",
]
