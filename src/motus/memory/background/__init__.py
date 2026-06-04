"""
Background memory system.

A MemoryAgent with scoped file tools manages a hierarchical memory tree.
Chunks of conversation messages are sent to the memory agent at compaction
time (one batch per compaction), and the agent files them into the tree.
The main agent can query the tree via search_memory().
"""

from .chunk import MemoryChunk

__all__ = [
    "MemoryChunk",
]
