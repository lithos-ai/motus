"""
MemoryChunk — the unit of work passed from BackgroundMemory to the
MemoryUpdateAgent.

One chunk is created per compaction, containing all pre-compaction messages
plus a stable identifier that threads through the raw log filename and every
fact's source reference.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from motus.models.base import ChatMessage


@dataclass
class MemoryChunk:
    """
    A batch of conversation messages to be filed into the memory tree.

    Created at compaction time from all pre-compaction messages. The memory
    agent reads the raw chunk, splits it by topic if needed, and files
    annotations into the appropriate tree nodes.

    chunk_id   — stable identifier; used as the filename in raw_logs/ and
                 referenced by leaf nodes (ref_<chunk_id>.md) across the tree.
    messages   — raw ChatMessage objects.
    turn_start — index of the first message (0-based).
    turn_end   — index of the last message (inclusive).
    timestamp  — when this chunk was created.
    """

    chunk_id: str
    messages: list[ChatMessage]
    turn_start: int
    turn_end: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        messages: list[ChatMessage],
        turn_start: int,
        turn_end: int,
    ) -> "MemoryChunk":
        """Create a new chunk with an auto-generated ID."""
        return cls(
            chunk_id=str(uuid.uuid4())[:8],
            messages=messages,
            turn_start=turn_start,
            turn_end=turn_end,
        )

    def __repr__(self) -> str:
        return (
            f"MemoryChunk(id={self.chunk_id!r}, "
            f"turns={self.turn_start}-{self.turn_end}, "
            f"messages={len(self.messages)})"
        )
