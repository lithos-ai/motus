"""
Abstract base classes for memory storage backends.

Defines the interfaces that storage implementations must follow:
- MemoryStore: Generic key-value storage interface
- VectorStore: Vector database operations interface
- ConversationLogStore: Append-only conversation log interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from .models import MemoryEntry, MemoryScope

T = TypeVar("T")


class MemoryStore(ABC, Generic[T]):
    """
    Abstract base class for memory storage backends.
    Implementations can use files, databases, vector stores, etc.
    """

    @abstractmethod
    def store(self, key: str, value: T, scope: MemoryScope) -> None:
        """Store a value with the given key."""
        ...

    @abstractmethod
    def retrieve(self, key: str, scope: MemoryScope) -> Optional[T]:
        """Retrieve a value by key."""
        ...

    @abstractmethod
    def delete(self, key: str, scope: MemoryScope) -> bool:
        """Delete a value by key. Returns True if deleted."""
        ...

    @abstractmethod
    def list_keys(self, scope: MemoryScope, prefix: str = "") -> List[str]:
        """List all keys matching the scope and optional prefix."""
        ...

    @abstractmethod
    def clear(self, scope: MemoryScope) -> int:
        """Clear all entries for the given scope. Returns count deleted."""
        ...


class VectorStore(ABC):
    """
    Abstract base class for vector database operations.
    Implementations can use Pinecone, Chroma, Qdrant, etc.
    """

    @abstractmethod
    def upsert(
        self,
        entries: List[MemoryEntry],
        scope: MemoryScope,
    ) -> List[str]:
        """Insert or update memory entries. Returns list of IDs."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        scope: MemoryScope,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryEntry]:
        """Search for similar entries by embedding."""
        ...

    @abstractmethod
    def search_by_text(
        self,
        query: str,
        scope: MemoryScope,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryEntry]:
        """Search for similar entries by text (auto-embeds the query)."""
        ...

    @abstractmethod
    def delete(self, ids: List[str], scope: MemoryScope) -> int:
        """Delete entries by IDs. Returns count deleted."""
        ...

    @abstractmethod
    def get_by_ids(self, ids: List[str], scope: MemoryScope) -> List[MemoryEntry]:
        """Retrieve specific entries by their IDs."""
        ...

    @abstractmethod
    def clear(self, scope: MemoryScope) -> int:
        """Clear all entries for the given scope. Returns count deleted."""
        ...


class ConversationLogStore(ABC):
    """
    Abstract base class for conversation log storage backends.

    Manages append-only logs for CompactionMemory sessions.
    Each session is identified by a unique session_id string.
    Implementations can use local files, S3, DynamoDB, etc.
    """

    @abstractmethod
    def append(self, session_id: str, entry: Dict[str, Any]) -> None:
        """Append a single log entry to the session log.

        Args:
            session_id: Unique session identifier.
            entry: JSON-serializable dict to append.
        """
        ...

    @abstractmethod
    def read_entries(self, session_id: str) -> List[Dict[str, Any]]:
        """Read all log entries for a session, in chronological order.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of log entry dicts. Empty list if session not found.
        """
        ...

    @abstractmethod
    def search_messages(
        self, session_id: str, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search message entries by case-insensitive substring match on content.

        Args:
            session_id: Unique session identifier.
            query: Substring to search for in message content.
            max_results: Maximum number of results to return.

        Returns:
            List of matching message dicts with keys: role, content, ts.
        """
        ...

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check whether a log exists for the given session_id."""
        ...

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all known session_ids."""
        ...
