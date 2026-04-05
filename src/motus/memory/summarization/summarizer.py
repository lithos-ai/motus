"""
Summarization agent for the memory system.

Provides functions to summarize chat message history into MemoryEntry objects
that can be stored in the memory system. Uses structured outputs for reliable
classification of memory type, priority, and tags.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from motus.models import ChatMessage

from ..models import MemoryEntry, MemoryPriority, MemoryType

logger = logging.getLogger("MemorySummarization")


class SummarizationOutput(BaseModel):
    """Structured output schema for conversation summarization."""

    key_name: str = Field(
        description="A short identifier for this memory (1-2 words, lowercase, hyphenated, e.g., 'auth-setup', 'user-prefs')"
    )
    description: str = Field(
        description="A brief one-sentence description of what this memory contains"
    )
    summary: str = Field(
        description="A concise but comprehensive summary of the conversation"
    )
    memory_type: MemoryType = Field(
        description="The type of memory this summary represents"
    )
    priority: MemoryPriority = Field(description="The priority level of this memory")
    tags: List[str] = Field(
        description="Relevant tags for categorizing this memory (3-7 tags recommended)"
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="Key people, places, or things mentioned in the conversation",
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Any action items or tasks mentioned that should be tracked",
    )


class SummarizationResult(BaseModel):
    """Result of a summarization operation containing the key, description, and memory entry."""

    key_name: str = Field(
        description="A short identifier for this memory (1-2 words, lowercase, hyphenated)"
    )
    description: str = Field(
        description="A brief one-sentence description of what this memory contains"
    )
    entry: MemoryEntry = Field(
        description="The memory entry containing the full summary"
    )

    model_config = {"arbitrary_types_allowed": True}


# Default system prompt for summarization with classification guidance
DEFAULT_SUMMARIZATION_PROMPT = """You are a conversation summarizer and memory classifier. Your task is to:
1. Generate a short key name and description for easy identification
2. Create a concise but comprehensive summary of the conversation
3. Classify the memory type, priority, and relevant tags

## Key Name and Description:
- **key_name**: A short 1-2 word identifier in lowercase hyphenated format. Should capture the main topic or action (e.g., "auth-setup", "db-migration", "user-prefs", "api-design", "bug-fix").
- **description**: A single brief sentence describing what this memory contains (e.g., "Discussion about setting up OAuth2 authentication flow").

## Memory Type Classification:
- **fact**: Factual information about specific entities (people, places, things). Use when the conversation establishes concrete facts like "John works at Google" or "The project deadline is March 15th".
- **episode**: Records of events or interactions. Use for conversation summaries that capture what happened during a discussion, meeting, or exchange.
- **semantic**: Conceptual or general knowledge. Use when the conversation covers abstract concepts, explanations, or general understanding.
- **procedural**: How-to knowledge and workflows. Use when the conversation covers processes, instructions, or step-by-step procedures.
- **preference**: User preferences and settings. Use when the user expresses likes, dislikes, preferred ways of working, or personal choices.

## Priority Classification:
- **low**: Casual conversation, minor details, or information unlikely to be needed again.
- **normal**: Standard information that may be useful for context in future conversations.
- **high**: Important information that should be readily accessible - key decisions, significant facts, or critical preferences.
- **critical**: Must-remember information - security-related, urgent deadlines, critical user requirements, or information the user explicitly emphasized as important.

## Guidelines for Summary:
- Capture the key topics discussed
- Note any important decisions or conclusions reached
- Include any action items or next steps mentioned
- Preserve important facts, preferences, or information about the user
- Keep the summary clear and well-structured
- Be concise but don't omit important details

## Guidelines for Tags:
- Include 3-7 relevant tags
- Use lowercase, hyphenated format (e.g., "project-planning", "user-preference")
- Include topic tags, entity tags, and context tags as appropriate
- Always include "conversation-summary" as one of the tags"""


def _format_messages_for_summary(messages: List[ChatMessage]) -> str:
    """
    Format messages into a string for summarization.

    Args:
        messages: List of ChatMessage objects to format.

    Returns:
        Formatted string representation of the conversation.
    """
    formatted_parts = []

    for msg in messages:
        role = msg.role.upper() if isinstance(msg.role, str) else msg.role.value.upper()

        if msg.content:
            formatted_parts.append(f"{role}: {msg.content}")

        if msg.tool_calls:
            for tc in msg.tool_calls:
                formatted_parts.append(
                    f"{role} [tool call]: {tc.function.name}({tc.function.arguments})"
                )

    return "\n\n".join(formatted_parts)


def _output_to_summarization_result(
    output: SummarizationOutput,
    message_count: int,
    model_name: str,
) -> SummarizationResult:
    """
    Convert a SummarizationOutput to a SummarizationResult.

    Args:
        output: The structured output from the LLM.
        message_count: Number of messages that were summarized.
        model_name: The model used for summarization.

    Returns:
        A SummarizationResult containing key_name, description, and the MemoryEntry.
    """
    entry = MemoryEntry(
        content=output.summary,
        memory_type=output.memory_type,
        priority=output.priority,
        tags=output.tags,
        metadata={
            "key_name": output.key_name,
            "description": output.description,
            "message_count": message_count,
            "summarized_at": datetime.now(timezone.utc).isoformat(),
            "model_used": model_name,
            "key_entities": output.key_entities,
            "action_items": output.action_items,
            "auto_classified": True,
        },
    )
    return SummarizationResult(
        key_name=output.key_name,
        description=output.description,
        entry=entry,
    )


def get_summarization_fn(
    model_name: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Optional[callable]:
    """
    Create a synchronous summarization function for the given model.

    This returns a function with signature `(List[ChatMessage]) -> SummarizationResult`
    that automatically classifies memory type, priority, and tags using structured outputs.

    Args:
        model_name: The name of the LLM model to use for summarization
                   (e.g., "gpt-4o-mini", "gpt-4o").
        api_key: Optional API key. If not provided, uses environment variables.
        system_prompt: Optional custom system prompt for summarization.

    Returns:
        A function that takes a list of messages and returns a SummarizationResult
        with key_name, description, and entry (with auto-classified memory type,
        priority, and tags), or None if the API key is not provided.

    Example:
        summarize = get_summarization_fn("gpt-4o-mini")
        if summarize:
            result = summarize(messages)
            memory.remember_short_term_promotable(result.key_name, result.entry)
    """
    import os

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        return None

    client = OpenAI(api_key=resolved_api_key)
    prompt = system_prompt or DEFAULT_SUMMARIZATION_PROMPT

    def summarize(messages: List[ChatMessage]) -> SummarizationResult:
        """Summarize messages and return a SummarizationResult."""
        if not messages:
            return SummarizationResult(
                key_name="empty",
                description="No messages to summarize.",
                entry=MemoryEntry(
                    content="No messages to summarize.",
                    memory_type=MemoryType.EPISODE,
                    priority=MemoryPriority.LOW,
                    tags=["empty-conversation"],
                ),
            )

        conversation_text = _format_messages_for_summary(messages)

        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Please summarize and classify the following conversation:\n\n{conversation_text}",
                },
            ],
            response_format=SummarizationOutput,
            temperature=0.3,
        )

        output = response.choices[0].message.parsed
        if output is None:
            # Fallback if parsing fails
            logger.warning("Structured output parsing failed, using fallback")
            return SummarizationResult(
                key_name="summary-fallback",
                description="Summary created with fallback due to parsing failure.",
                entry=MemoryEntry(
                    content=response.choices[0].message.content
                    or "Summary unavailable",
                    memory_type=MemoryType.EPISODE,
                    priority=MemoryPriority.NORMAL,
                    tags=["conversation-summary", "parse-fallback"],
                    metadata={
                        "message_count": len(messages),
                        "summarized_at": datetime.now(timezone.utc).isoformat(),
                        "model_used": model_name,
                        "auto_classified": False,
                    },
                ),
            )

        return _output_to_summarization_result(output, len(messages), model_name)

    return summarize


async def get_summarization_fn_async(
    model_name: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Optional[callable]:
    """
    Create an async summarization function for the given model.

    Args:
        model_name: The name of the LLM model to use for summarization.
        api_key: Optional API key.
        system_prompt: Optional custom system prompt.

    Returns:
        An async function that takes a list of messages and returns a
        SummarizationResult with key_name, description, and entry,
        or None if the API key is not provided.

    Example:
        summarize = await get_summarization_fn_async("gpt-4o-mini")
        if summarize:
            result = await summarize(messages)
    """
    import os

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        return None

    client = AsyncOpenAI(api_key=resolved_api_key)
    prompt = system_prompt or DEFAULT_SUMMARIZATION_PROMPT

    async def summarize(messages: List[ChatMessage]) -> SummarizationResult:
        """Summarize messages asynchronously and return a SummarizationResult."""
        if not messages:
            return SummarizationResult(
                key_name="empty",
                description="No messages to summarize.",
                entry=MemoryEntry(
                    content="No messages to summarize.",
                    memory_type=MemoryType.EPISODE,
                    priority=MemoryPriority.LOW,
                    tags=["empty-conversation"],
                ),
            )

        conversation_text = _format_messages_for_summary(messages)

        response = await client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Please summarize and classify the following conversation:\n\n{conversation_text}",
                },
            ],
            response_format=SummarizationOutput,
            temperature=0.3,
        )

        output = response.choices[0].message.parsed
        if output is None:
            logger.warning("Structured output parsing failed, using fallback")
            return SummarizationResult(
                key_name="summary-fallback",
                description="Summary created with fallback due to parsing failure.",
                entry=MemoryEntry(
                    content=response.choices[0].message.content
                    or "Summary unavailable",
                    memory_type=MemoryType.EPISODE,
                    priority=MemoryPriority.NORMAL,
                    tags=["conversation-summary", "parse-fallback"],
                    metadata={
                        "message_count": len(messages),
                        "summarized_at": datetime.now(timezone.utc).isoformat(),
                        "model_used": model_name,
                        "auto_classified": False,
                    },
                ),
            )

        return _output_to_summarization_result(output, len(messages), model_name)

    return summarize


def summarize_messages(
    messages: List[ChatMessage],
    model_name: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra_tags: Optional[List[str]] = None,
) -> SummarizationResult:
    """
    Summarize a list of messages into a SummarizationResult.

    This is a convenience function for one-off summarization. Memory type,
    priority, and tags are automatically determined by the LLM.

    Args:
        messages: List of ChatMessage objects to summarize.
        model_name: The LLM model to use for summarization.
        api_key: Optional API key.
        system_prompt: Optional custom system prompt.
        extra_tags: Optional additional tags to merge with auto-generated tags.

    Returns:
        A SummarizationResult with key_name, description, and entry.

    Example:
        result = summarize_messages(
            messages=chat_history,
            model_name="gpt-4o-mini",
            extra_tags=["weekly-review"],
        )
        memory.remember_short_term_promotable(result.key_name, result.entry)
    """
    summarize_fn = get_summarization_fn(model_name, api_key, system_prompt)
    result = summarize_fn(messages)

    if extra_tags:
        result.entry.tags = list(set(result.entry.tags + extra_tags))

    return result


async def summarize_messages_async(
    messages: List[ChatMessage],
    model_name: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra_tags: Optional[List[str]] = None,
) -> SummarizationResult:
    """
    Asynchronously summarize a list of messages with auto-classification.

    Args:
        messages: List of ChatMessage objects to summarize.
        model_name: The LLM model to use for summarization.
        api_key: Optional API key.
        system_prompt: Optional custom system prompt.
        extra_tags: Optional additional tags to merge with auto-generated tags.

    Returns:
        A SummarizationResult with key_name, description, and entry.

    Example:
        result = await summarize_messages_async(
            messages=chat_history,
            model_name="gpt-4o-mini",
        )
    """
    summarize_fn = await get_summarization_fn_async(model_name, api_key, system_prompt)
    result = await summarize_fn(messages)

    if extra_tags:
        result.entry.tags = list(set(result.entry.tags + extra_tags))

    return result
