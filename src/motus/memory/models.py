"""
Memory data models and types.

Contains core data structures for the memory system:
- MemoryScope: Defines the context/partition for memory operations
- MemoryType: Enum for categorizing memory entries
- MemoryPriority: Enum for priority levels
- MemoryEntry: A single memory entry with metadata
- PromotableEntry: Wrapper for entries that can be promoted to long-term memory
"""

import hashlib
import time
from datetime import datetime, timezone
from enum import StrEnum, auto
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryScope(BaseModel):
    """
    Defines the scope/context for memory operations.
    Used to partition memories by user, session, organization, etc.
    """

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    org_id: Optional[str] = None
    agent_id: Optional[str] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_path_prefix(self) -> str:
        """Generate a path prefix for file-based storage."""
        parts = []
        if self.org_id:
            parts.append(f"org_{self.org_id}")
        if self.user_id:
            parts.append(f"user_{self.user_id}")
        if self.agent_id:
            parts.append(f"agent_{self.agent_id}")
        if self.session_id:
            parts.append(f"session_{self.session_id}")
        if self.namespace:
            parts.append(self.namespace)
        return "/".join(parts) if parts else "default"

    def to_filter_dict(self) -> Dict[str, str]:
        """Generate a filter dictionary for vector database queries."""
        filters = {}
        if self.user_id:
            filters["user_id"] = self.user_id
        if self.session_id:
            filters["session_id"] = self.session_id
        if self.org_id:
            filters["org_id"] = self.org_id
        if self.agent_id:
            filters["agent_id"] = self.agent_id
        if self.namespace:
            filters["namespace"] = self.namespace
        return filters


class MemoryType(StrEnum):
    """Types of memory entries."""

    FACT = auto()  # Factual information about entities
    EPISODE = auto()  # Event/interaction records
    SEMANTIC = auto()  # Conceptual/general knowledge
    PROCEDURAL = auto()  # How-to knowledge, workflows
    PREFERENCE = auto()  # User preferences and settings


class MemoryPriority(StrEnum):
    """Priority levels for memory entries."""

    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class MemoryEntry(BaseModel):
    """A single memory entry with metadata."""

    id: str = Field(
        default_factory=lambda: hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]
    )
    content: str
    memory_type: MemoryType = MemoryType.FACT
    priority: MemoryPriority = MemoryPriority.NORMAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ttl_seconds: Optional[int] = None  # Time-to-live, None means permanent
    embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None  # Where this memory came from
    related_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if this memory entry has expired."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.accessed_at = datetime.now(timezone.utc)
        self.access_count += 1


class PromotableEntry(BaseModel):
    """
    Wrapper for entries stored in short-term memory with promotion tracking.
    Used to track access patterns and determine when to promote to long-term memory.
    """

    entry: MemoryEntry
    access_count: int = 0
    stored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = None

    def touch(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)

    def age_seconds(self) -> float:
        """Return the age of this entry in seconds."""
        return (datetime.now(timezone.utc) - self.stored_at).total_seconds()
