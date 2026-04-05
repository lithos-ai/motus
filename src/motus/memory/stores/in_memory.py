"""
In-memory vector store implementation.

Provides a simple in-memory vector store for development and testing.
For production, implement with Pinecone, Chroma, Qdrant, etc.
"""

from typing import Any, Callable, Dict, List, Optional

from ..interfaces import VectorStore
from ..models import MemoryEntry, MemoryScope


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for development/testing.
    For production, implement with Pinecone, Chroma, Qdrant, etc.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        dimension: int = 1536,
    ):
        self._entries: Dict[str, MemoryEntry] = {}
        self._scope_index: Dict[str, List[str]] = {}  # scope_key -> list of entry ids
        self._embed_fn = embed_fn
        self._dimension = dimension

    def _scope_key(self, scope: MemoryScope) -> str:
        return scope.to_path_prefix()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def upsert(
        self,
        entries: List[MemoryEntry],
        scope: MemoryScope,
    ) -> List[str]:
        scope_key = self._scope_key(scope)
        if scope_key not in self._scope_index:
            self._scope_index[scope_key] = []

        ids = []
        for entry in entries:
            # Generate embedding if not provided and embed_fn is available
            if entry.embedding is None and self._embed_fn:
                entry.embedding = self._embed_fn(entry.content)

            self._entries[entry.id] = entry
            if entry.id not in self._scope_index[scope_key]:
                self._scope_index[scope_key].append(entry.id)
            ids.append(entry.id)

        return ids

    def search(
        self,
        query_embedding: List[float],
        scope: MemoryScope,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryEntry]:
        scope_key = self._scope_key(scope)
        entry_ids = self._scope_index.get(scope_key, [])

        scored_entries = []
        for entry_id in entry_ids:
            entry = self._entries.get(entry_id)
            if entry is None or entry.is_expired():
                continue

            # Apply filters
            if filters:
                skip = False
                for k, v in filters.items():
                    if entry.metadata.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue

            if entry.embedding:
                score = self._cosine_similarity(query_embedding, entry.embedding)
                scored_entries.append((score, entry))

        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored_entries[:top_k]]

        # Update access stats
        for entry in results:
            entry.touch()

        return results

    def search_by_text(
        self,
        query: str,
        scope: MemoryScope,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryEntry]:
        if not self._embed_fn:
            raise ValueError("No embedding function provided for text search")
        query_embedding = self._embed_fn(query)
        return self.search(query_embedding, scope, top_k, filters)

    def delete(self, ids: List[str], scope: MemoryScope) -> int:
        scope_key = self._scope_key(scope)
        count = 0
        for entry_id in ids:
            if entry_id in self._entries:
                del self._entries[entry_id]
                if scope_key in self._scope_index:
                    if entry_id in self._scope_index[scope_key]:
                        self._scope_index[scope_key].remove(entry_id)
                count += 1
        return count

    def get_by_ids(self, ids: List[str], _scope: MemoryScope) -> List[MemoryEntry]:
        # Note: scope is unused in in-memory implementation but required by interface
        results = []
        for entry_id in ids:
            entry = self._entries.get(entry_id)
            if entry and not entry.is_expired():
                entry.touch()
                results.append(entry)
        return results

    def clear(self, scope: MemoryScope) -> int:
        """Clear all entries for the given scope. Returns count deleted."""
        scope_key = self._scope_key(scope)
        entry_ids = self._scope_index.get(scope_key, [])
        count = len(entry_ids)

        # Remove all entries for this scope
        for entry_id in entry_ids:
            if entry_id in self._entries:
                del self._entries[entry_id]

        # Clear the scope index
        if scope_key in self._scope_index:
            del self._scope_index[scope_key]

        return count
