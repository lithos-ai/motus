"""
Embedding utilities for the memory system.

Provides a simple interface to create embedding functions from model names.

Usage:
    from motus.memory.embedding import get_embedding_fn

    # Create an embedding function from a model name
    embed_fn = get_embedding_fn("text-embedding-3-small")

    # Use with the memory system
    memory = Memory(
        scope=MemoryScope(user_id="user123"),
        embed_fn=embed_fn,
    )
"""

from .embedder import (
    EmbeddingProvider,
    get_embedding_dimension,
    get_embedding_fn,
    get_embedding_fn_async,
)

__all__ = [
    "get_embedding_fn",
    "get_embedding_fn_async",
    "get_embedding_dimension",
    "EmbeddingProvider",
]
