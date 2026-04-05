"""
Memory system prompt templates.

Contains all prompt templates used for exposing memory capabilities to agents.
Separating templates from logic makes them easier to customize and maintain.
"""

from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Short-Term Memory Prompt Templates
# -----------------------------------------------------------------------------

SHORT_TERM_MEMORY_HEADER = """### Short-Term Memory

Key-value storage for session-specific data. Use `memory_search_short_term` with the exact key to retrieve.
"""

SHORT_TERM_KEYS_HEADER = "**Currently available keys:**"
SHORT_TERM_NO_KEYS = "**Currently available keys:** None"
SHORT_TERM_REGULAR_KEYS_HEADER = "Regular keys:"
SHORT_TERM_PROMOTABLE_KEYS_HEADER = (
    "Promotable keys (may be promoted to long-term memory):"
)


def _format_entry(entry: Dict[str, Optional[str]]) -> str:
    """Format a single entry with key and optional description."""
    key = entry.get("key", "")
    description = entry.get("description")
    if description:
        return f"  - `{key}`: {description}"
    return f"  - `{key}`"


def format_short_term_memory_prompt(
    regular_entries: List[Dict[str, Optional[str]]],
    promotable_entries: List[Dict[str, Optional[str]]],
    max_regular_entries: int = 50,
    max_promotable_entries: int = 20,
) -> str:
    """
    Format the short-term memory prompt with current keys and descriptions.

    Args:
        regular_entries: List of dicts with 'key' and optional 'description'
        promotable_entries: List of dicts with 'key' and optional 'description'
        max_regular_entries: Maximum regular entries to display
        max_promotable_entries: Maximum promotable entries to display

    Returns:
        Formatted short-term memory prompt string
    """
    lines = [SHORT_TERM_MEMORY_HEADER]

    if regular_entries or promotable_entries:
        lines.append(SHORT_TERM_KEYS_HEADER)

        if regular_entries:
            lines.append("")
            lines.append(SHORT_TERM_REGULAR_KEYS_HEADER)
            for entry in regular_entries[:max_regular_entries]:
                lines.append(_format_entry(entry))
            if len(regular_entries) > max_regular_entries:
                lines.append(
                    f"  - ... and {len(regular_entries) - max_regular_entries} more"
                )

        if promotable_entries:
            lines.append("")
            lines.append(SHORT_TERM_PROMOTABLE_KEYS_HEADER)
            for entry in promotable_entries[:max_promotable_entries]:
                lines.append(_format_entry(entry))
            if len(promotable_entries) > max_promotable_entries:
                lines.append(
                    f"  - ... and {len(promotable_entries) - max_promotable_entries} more"
                )
    else:
        lines.append(SHORT_TERM_NO_KEYS)

    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Long-Term Memory Prompt Templates
# -----------------------------------------------------------------------------

LONG_TERM_MEMORY_PROMPT = """### Long-Term Memory

Persistent semantic search across sessions. Use `memory_search_long_term` with a descriptive query.

**Memory types:** `fact`, `episode`, `semantic`, `procedural`, `preference`
"""


# -----------------------------------------------------------------------------
# Memory Usage Guidelines
# -----------------------------------------------------------------------------

MEMORY_USAGE_GUIDELINES = """### Guidelines

- **Short-term**: Use exact keys listed above for session context
- **Long-term**: Use descriptive queries for cross-session knowledge
- Search memory when the user references past context, don't ask them to repeat

### Fallback Strategy

Memory is one source of information, not the only one. When retrieved entries are:
- **Empty or not found**: The information may not have been stored—proceed with other tools
- **Irrelevant to the task**: Don't force a fit; use web search, file reads, or ask the user
- **Outdated or incomplete**: Supplement with current data from other sources

**Important**: Do not get stuck on memory retrieval. If a memory search doesn't yield useful results after one attempt, move on and complete the task using your other available tools (file operations, web search, code execution, etc.).
"""


# -----------------------------------------------------------------------------
# Combined Memory Prompt
# -----------------------------------------------------------------------------

MEMORY_CAPABILITIES_HEADER = """## Memory

"""


def format_memory_prompt(
    short_term_prompt: str,
    long_term_prompt: str = LONG_TERM_MEMORY_PROMPT,
    guidelines: str = MEMORY_USAGE_GUIDELINES,
) -> str:
    """
    Format the complete memory capabilities prompt.

    Args:
        short_term_prompt: Formatted short-term memory prompt
        long_term_prompt: Long-term memory prompt (uses default if not provided)
        guidelines: Memory usage guidelines (uses default if not provided)

    Returns:
        Complete memory capabilities prompt string
    """
    return f"{MEMORY_CAPABILITIES_HEADER}{short_term_prompt}\n{long_term_prompt}\n{guidelines}"
