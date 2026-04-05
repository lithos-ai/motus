"""
Unit tests for BaseMemory abstract interface.

Tests cover:
- BaseMemory cannot be instantiated directly
- Memory is a valid BaseMemory subclass
- CompactionMemory is a valid BaseMemory subclass
"""

import unittest

from motus.memory.base_memory import BaseMemory
from motus.memory.compaction_memory import CompactionMemory
from motus.memory.config import DatabaseMemoryConfig
from motus.memory.database_memory import DatabaseMemory
from motus.memory.models import MemoryScope


def _mock_embed(text: str):
    return [0.0] * 128


def _mock_summarize(messages):
    return None


class TestBaseMemoryAbstract(unittest.TestCase):
    def test_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            BaseMemory()

    def test_memory_is_base_memory(self):
        scope = MemoryScope(session_id="test")
        config = DatabaseMemoryConfig(auto_compact=False)
        mem = DatabaseMemory(
            scope=scope,
            model_name="gpt-4",
            config=config,
            embed_fn=_mock_embed,
            summarize_fn=_mock_summarize,
        )
        self.assertIsInstance(mem, BaseMemory)

    def test_compaction_memory_is_base_memory(self):
        mem = CompactionMemory(
            model_name="gpt-4",
            compact_fn=lambda msgs, sp: "summary",
        )
        self.assertIsInstance(mem, BaseMemory)

    def test_memory_has_no_set_model(self):
        """Memory doesn't need set_model — model_name is required at init."""
        scope = MemoryScope(session_id="test")
        config = DatabaseMemoryConfig(auto_compact=False)
        mem = DatabaseMemory(
            scope=scope,
            model_name="gpt-4",
            config=config,
            embed_fn=_mock_embed,
            summarize_fn=_mock_summarize,
        )
        self.assertFalse(hasattr(mem, "set_model"))


if __name__ == "__main__":
    unittest.main()
