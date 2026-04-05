"""Memory system configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CompactionMemoryConfig:
    """Configuration for CompactionMemory."""

    safety_ratio: float = 0.75  # Compact when over 75% of context limit
    token_threshold: Optional[int] = None  # Explicit threshold (overrides safety_ratio)
    compact_model_name: Optional[str] = (
        None  # Model for compaction (defaults to agent model)
    )
    max_tool_result_tokens: int = 50000  # Max tokens per tool result (0 = no limit)
    tool_result_truncation_suffix: str = "\n\n... [content truncated due to length]"

    # Session identity
    session_id: Optional[str] = None  # Auto-UUID if not provided

    # Conversation log persistence
    # Custom path for JSONL log files. None = use LocalConversationLogStore default.
    log_base_path: Optional[str] = None


@dataclass
class DatabaseMemoryConfig:
    """Configuration for DatabaseMemory."""

    # Short-term memory settings
    short_term_base_path: str = "./.motus_memory/short_term"
    short_term_max_entries: int = 1000
    short_term_default_ttl: int = 3600 * 24  # 24 hours

    # Long-term memory settings
    long_term_max_results: int = 20

    # Auto-compaction settings
    auto_compact: bool = True
    compact_safety_ratio: float = 0.75  # Compact when over 75% of context limit
    compact_preserve_ratio: float = 0.1  # Keep ~10% of max context for recent messages

    # Promotion settings (short-term -> long-term)
    auto_promote: bool = True
    promotion_access_threshold: int = 3  # Promote after 3+ accesses
    promotion_time_threshold_seconds: int = 3600  # Promote after 1 hour
    promotion_check_interval_seconds: int = 60  # Check every 60 seconds

    # Tool result truncation settings
    max_tool_result_tokens: int = 50000  # Max tokens per tool result (0 = no limit)
    tool_result_truncation_suffix: str = "\n\n... [content truncated due to length]"

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"

    # Summarization settings
    summarization_model: str = "gpt-4o-mini"
