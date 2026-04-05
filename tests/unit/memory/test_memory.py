"""
Unit tests for the Memory system.

Tests cover:
- Short-term memory: insertion, retrieval, deletion, listing, TTL expiration
- Long-term memory: insertion, search, batch operations, filtering
- Memory promotion: access count, time-based, priority-based promotion
- Vector store: cosine similarity search, scope isolation
"""

import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from typing import List

from motus.memory.config import DatabaseMemoryConfig
from motus.memory.database_memory import DatabaseMemory
from motus.memory.models import (
    MemoryEntry,
    MemoryPriority,
    MemoryScope,
    MemoryType,
    PromotableEntry,
)
from motus.memory.stores import (
    FileSystemStore,
    InMemoryVectorStore,
)
from motus.memory.summarization import SummarizationResult


def mock_embed_fn(text: str) -> List[float]:
    """
    Simple mock embedding function for testing.
    Creates a deterministic embedding based on text hash.
    """
    import hashlib

    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    embedding = []
    for i in range(128):
        embedding.append(((hash_val >> i) & 1) * 2 - 1 + (i / 1000))
    norm = sum(x * x for x in embedding) ** 0.5
    return [x / norm for x in embedding]


def mock_summarize_fn(messages):
    """
    Mock summarization function for testing.
    Returns a simple SummarizationResult without calling any API.
    """
    content = (
        " ".join(m.content for m in messages if m.content)[:100]
        or "Summary of conversation"
    )

    return SummarizationResult(
        key_name="test_summary",
        description="Test summary description",
        entry=MemoryEntry(
            content=content,
            memory_type=MemoryType.EPISODE,
            tags=["summary", "test"],
        ),
    )


class TestShortTermMemory(unittest.TestCase):
    """Tests for short-term memory operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scope = MemoryScope(user_id="test_user", session_id="test_session")
        self.config = DatabaseMemoryConfig(
            short_term_base_path=self.temp_dir,
            auto_compact=False,
        )
        self.store = FileSystemStore(self.temp_dir)
        self.memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",
            config=self.config,
            short_term_store=self.store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

    def test_store_and_retrieve_value(self):
        """Test basic store and retrieve operations."""
        self.memory.remember_short_term("user_name", "Alice")
        result = self.memory.recall_short_term("user_name")
        self.assertEqual(result, "Alice")

    def test_store_complex_value(self):
        """Test storing complex data structures."""
        data = {"name": "Alice", "preferences": ["dark_mode", "vim"], "count": 42}
        self.memory.remember_short_term("user_data", data)
        result = self.memory.recall_short_term("user_data")
        self.assertEqual(result, data)

    def test_retrieve_nonexistent_key(self):
        """Test retrieving a key that doesn't exist."""
        result = self.memory.recall_short_term("nonexistent_key")
        self.assertIsNone(result)

    def test_delete_value(self):
        """Test deleting a stored value."""
        self.memory.remember_short_term("to_delete", "value")
        self.assertEqual(self.memory.recall_short_term("to_delete"), "value")

        result = self.memory.forget_short_term("to_delete")
        self.assertTrue(result)
        self.assertIsNone(self.memory.recall_short_term("to_delete"))

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist."""
        result = self.memory.forget_short_term("nonexistent")
        self.assertFalse(result)

    def test_list_keys(self):
        """Test listing all keys in short-term memory."""
        self.memory.remember_short_term("key1", "value1")
        self.memory.remember_short_term("key2", "value2")
        self.memory.remember_short_term("prefix_key3", "value3")

        keys = self.memory.list_short_term_keys()
        self.assertEqual(len(keys), 3)
        self.assertIn("key1", keys)
        self.assertIn("key2", keys)
        self.assertIn("prefix_key3", keys)

    def test_list_keys_with_prefix(self):
        """Test listing keys with a prefix filter."""
        self.memory.remember_short_term("task_1", "value1")
        self.memory.remember_short_term("task_2", "value2")
        self.memory.remember_short_term("other_key", "value3")

        keys = self.memory.list_short_term_keys(prefix="task_")
        self.assertEqual(len(keys), 2)
        self.assertIn("task_1", keys)
        self.assertIn("task_2", keys)
        self.assertNotIn("other_key", keys)

    def test_list_entries_with_description(self):
        """Test listing entries with descriptions."""
        self.memory.remember_short_term("key1", "value1", description="First key")
        self.memory.remember_short_term("key2", "value2", description="Second key")

        entries = self.memory.list_short_term_entries()
        self.assertEqual(len(entries), 2)

        entry_map = {e["key"]: e for e in entries}
        self.assertEqual(entry_map["key1"]["description"], "First key")
        self.assertEqual(entry_map["key2"]["description"], "Second key")

    def test_ttl_expiration(self):
        """Test that values expire after TTL."""
        self.memory.remember_short_term("expiring_key", "value", ttl_seconds=1)

        result = self.memory.recall_short_term("expiring_key")
        self.assertEqual(result, "value")

        time.sleep(1.5)

        result = self.memory.recall_short_term("expiring_key")
        self.assertIsNone(result)

    def test_overwrite_existing_key(self):
        """Test that storing to an existing key overwrites the value."""
        self.memory.remember_short_term("key", "original")
        self.assertEqual(self.memory.recall_short_term("key"), "original")

        self.memory.remember_short_term("key", "updated")
        self.assertEqual(self.memory.recall_short_term("key"), "updated")


class TestLongTermMemory(unittest.TestCase):
    """Tests for long-term memory operations."""

    def setUp(self):
        self.scope = MemoryScope(user_id="test_user", session_id="test_session")
        self.config = DatabaseMemoryConfig(auto_compact=False)
        self.vector_store = InMemoryVectorStore(embed_fn=mock_embed_fn, dimension=128)
        self.memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",
            config=self.config,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

    def test_store_memory_entry(self):
        """Test storing a MemoryEntry in long-term memory."""
        entry = MemoryEntry(
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            tags=["ui", "settings"],
        )
        entry_id = self.memory.remember_long_term(entry)

        self.assertIsNotNone(entry_id)
        self.assertTrue(len(entry_id) > 0)

    def test_store_string_content(self):
        """Test storing a simple string in long-term memory."""
        entry_id = self.memory.remember_long_term(
            "User's favorite color is blue",
            memory_type=MemoryType.FACT,
            tags=["preference"],
        )

        self.assertIsNotNone(entry_id)

    def test_search_by_text(self):
        """Test searching long-term memory by text query."""
        self.memory.remember_long_term(
            "User prefers dark mode for the IDE",
            memory_type=MemoryType.PREFERENCE,
        )
        self.memory.remember_long_term(
            "User likes Python programming language",
            memory_type=MemoryType.PREFERENCE,
        )
        self.memory.remember_long_term(
            "Meeting scheduled for Friday at 3pm",
            memory_type=MemoryType.FACT,
        )

        results = self.memory.recall("dark mode settings")

        self.assertTrue(len(results) > 0)
        self.assertTrue(any("dark mode" in r.content for r in results))

    def test_search_with_top_k(self):
        """Test limiting search results with top_k."""
        for i in range(10):
            self.memory.remember_long_term(f"Test memory entry number {i}")

        results = self.memory.recall("test memory", top_k=3)
        self.assertEqual(len(results), 3)

    def test_search_with_memory_type_filter(self):
        """Test filtering search results by memory type."""
        self.memory.remember_long_term(
            "User prefers vim keybindings",
            memory_type=MemoryType.PREFERENCE,
        )
        self.memory.remember_long_term(
            "User fact about vim",
            memory_type=MemoryType.FACT,
        )

        results = self.memory.recall(
            "vim", memory_types=[MemoryType.PREFERENCE], top_k=10
        )

        for result in results:
            if "vim" in result.content.lower():
                self.assertEqual(result.memory_type, MemoryType.PREFERENCE)

    def test_batch_insertion(self):
        """Test inserting multiple entries at once."""
        entries = [
            MemoryEntry(content="Batch entry 1", memory_type=MemoryType.FACT),
            MemoryEntry(content="Batch entry 2", memory_type=MemoryType.FACT),
            MemoryEntry(content="Batch entry 3", memory_type=MemoryType.FACT),
        ]

        ids = self.memory.remember_long_term_batch(entries)

        self.assertEqual(len(ids), 3)
        for entry_id in ids:
            self.assertIsNotNone(entry_id)

    def test_get_by_ids(self):
        """Test retrieving memories by their IDs."""
        entry1 = MemoryEntry(content="First entry", memory_type=MemoryType.FACT)
        entry2 = MemoryEntry(content="Second entry", memory_type=MemoryType.FACT)

        id1 = self.memory.remember_long_term(entry1)
        id2 = self.memory.remember_long_term(entry2)

        results = self.memory.recall_by_ids([id1, id2])

        self.assertEqual(len(results), 2)
        contents = [r.content for r in results]
        self.assertIn("First entry", contents)
        self.assertIn("Second entry", contents)

    def test_delete_memories(self):
        """Test deleting memories from long-term storage."""
        entry = MemoryEntry(content="To be deleted", memory_type=MemoryType.FACT)
        entry_id = self.memory.remember_long_term(entry)

        deleted_count = self.memory.forget_long_term([entry_id])
        self.assertEqual(deleted_count, 1)

        results = self.memory.recall_by_ids([entry_id])
        self.assertEqual(len(results), 0)

    def test_update_memory(self):
        """Test updating an existing memory entry."""
        entry = MemoryEntry(
            content="Original content",
            memory_type=MemoryType.FACT,
            tags=["original"],
        )
        entry_id = self.memory.remember_long_term(entry)

        success = self.memory.update_memory(
            entry_id, content="Updated content", tags=["updated"]
        )
        self.assertTrue(success)

        results = self.memory.recall_by_ids([entry_id])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Updated content")
        self.assertEqual(results[0].tags, ["updated"])


class TestMemoryPromotion(unittest.TestCase):
    """Tests for memory promotion from short-term to long-term."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scope = MemoryScope(user_id="test_user", session_id="test_session")
        self.config = DatabaseMemoryConfig(
            short_term_base_path=self.temp_dir,
            auto_compact=False,
            promotion_access_threshold=3,
            promotion_time_threshold_seconds=3600,
        )
        self.store = FileSystemStore(self.temp_dir)
        self.vector_store = InMemoryVectorStore(embed_fn=mock_embed_fn, dimension=128)
        self.memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",
            config=self.config,
            short_term_store=self.store,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

    def test_store_promotable_entry(self):
        """Test storing a promotable entry in short-term memory."""
        entry = MemoryEntry(
            content="Promotable content",
            memory_type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
        )
        self.memory.remember_short_term_promotable("promotable_key", entry)

        entries = self.memory.list_short_term_entries()
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["promotable"])

    def test_high_priority_immediate_promotion(self):
        """Test that HIGH priority entries are immediately promoted."""
        entry = MemoryEntry(
            content="Critical information",
            memory_type=MemoryType.FACT,
            priority=MemoryPriority.HIGH,
        )
        self.memory.remember_short_term_promotable("critical_key", entry)

        short_term_entries = self.memory.list_short_term_entries()
        self.assertEqual(len(short_term_entries), 0)

        long_term_results = self.memory.recall("Critical information")
        self.assertTrue(len(long_term_results) > 0)

    def test_critical_priority_immediate_promotion(self):
        """Test that CRITICAL priority entries are immediately promoted."""
        entry = MemoryEntry(
            content="Super critical data",
            memory_type=MemoryType.FACT,
            priority=MemoryPriority.CRITICAL,
        )
        self.memory.remember_short_term_promotable("super_critical", entry)

        short_term_entries = self.memory.list_short_term_entries()
        self.assertEqual(len(short_term_entries), 0)

    def test_access_count_promotion(self):
        """Test that entries are promoted after reaching access threshold."""
        entry = MemoryEntry(
            content="Frequently accessed",
            memory_type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
        )
        self.memory.remember_short_term_promotable("frequent_key", entry)

        for _ in range(self.config.promotion_access_threshold - 1):
            self.memory.access_promotable("frequent_key")
            entries = self.memory.list_short_term_entries()
            self.assertEqual(len(entries), 1)

        self.memory.access_promotable("frequent_key")

        entries = self.memory.list_short_term_entries()
        self.assertEqual(len(entries), 0)

        results = self.memory.recall("Frequently accessed")
        self.assertTrue(len(results) > 0)

    def test_manual_promotion(self):
        """Test manually promoting an entry."""
        entry = MemoryEntry(
            content="Manually promoted",
            memory_type=MemoryType.FACT,
        )
        self.memory.remember_short_term_promotable("manual_key", entry)

        long_term_id = self.memory.promote_entry("manual_key")
        self.assertIsNotNone(long_term_id)

        entries = self.memory.list_short_term_entries()
        self.assertEqual(len(entries), 0)

    def test_promotable_entry_tracking(self):
        """Test that promotable entries track access patterns."""
        entry = MemoryEntry(content="Tracked entry", memory_type=MemoryType.FACT)
        self.memory.remember_short_term_promotable("tracked_key", entry)

        for _ in range(2):
            self.memory.access_promotable("tracked_key")

        data = self.store.retrieve("tracked_key", self.scope)
        promotable = PromotableEntry(**data)
        self.assertEqual(promotable.access_count, 2)


class TestInMemoryVectorStore(unittest.TestCase):
    """Tests for the in-memory vector store implementation."""

    def setUp(self):
        self.store = InMemoryVectorStore(embed_fn=mock_embed_fn, dimension=128)
        self.scope = MemoryScope(user_id="test_user")

    def test_upsert_and_search(self):
        """Test basic upsert and search operations."""
        entry = MemoryEntry(
            content="Test content for search", memory_type=MemoryType.FACT
        )
        self.store.upsert([entry], self.scope)

        results = self.store.search_by_text("test content", self.scope)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].content, "Test content for search")

    def test_cosine_similarity_ranking(self):
        """Test that results are ranked by similarity."""
        entries = [
            MemoryEntry(content="Apple fruit is red", memory_type=MemoryType.FACT),
            MemoryEntry(content="Banana fruit is yellow", memory_type=MemoryType.FACT),
            MemoryEntry(content="Apple computer company", memory_type=MemoryType.FACT),
        ]
        self.store.upsert(entries, self.scope)

        results = self.store.search_by_text("Apple fruit", self.scope, top_k=3)
        self.assertTrue(len(results) > 0)

    def test_scope_isolation(self):
        """Test that different scopes are isolated."""
        scope1 = MemoryScope(user_id="user1")
        scope2 = MemoryScope(user_id="user2")

        entry1 = MemoryEntry(content="User1 secret data", memory_type=MemoryType.FACT)
        entry2 = MemoryEntry(content="User2 secret data", memory_type=MemoryType.FACT)

        self.store.upsert([entry1], scope1)
        self.store.upsert([entry2], scope2)

        results1 = self.store.search_by_text("secret data", scope1)
        results2 = self.store.search_by_text("secret data", scope2)

        self.assertEqual(len(results1), 1)
        self.assertEqual(results1[0].content, "User1 secret data")

        self.assertEqual(len(results2), 1)
        self.assertEqual(results2[0].content, "User2 secret data")

    def test_delete_entries(self):
        """Test deleting entries from the store."""
        entry = MemoryEntry(content="To be deleted", memory_type=MemoryType.FACT)
        ids = self.store.upsert([entry], self.scope)

        count = self.store.delete(ids, self.scope)
        self.assertEqual(count, 1)

        results = self.store.get_by_ids(ids, self.scope)
        self.assertEqual(len(results), 0)

    def test_get_by_ids(self):
        """Test retrieving entries by their IDs."""
        entries = [
            MemoryEntry(content="Entry A", memory_type=MemoryType.FACT),
            MemoryEntry(content="Entry B", memory_type=MemoryType.FACT),
        ]
        ids = self.store.upsert(entries, self.scope)

        results = self.store.get_by_ids(ids, self.scope)
        self.assertEqual(len(results), 2)

    def test_clear(self):
        """Test clearing all entries for a scope."""
        # Add entries to two different scopes
        scope1 = MemoryScope(user_id="user1")
        scope2 = MemoryScope(user_id="user2")

        entries1 = [
            MemoryEntry(content="Entry 1 for user1", memory_type=MemoryType.FACT),
            MemoryEntry(content="Entry 2 for user1", memory_type=MemoryType.FACT),
            MemoryEntry(content="Entry 3 for user1", memory_type=MemoryType.FACT),
        ]
        entries2 = [
            MemoryEntry(content="Entry 1 for user2", memory_type=MemoryType.FACT),
            MemoryEntry(content="Entry 2 for user2", memory_type=MemoryType.FACT),
        ]

        self.store.upsert(entries1, scope1)
        self.store.upsert(entries2, scope2)

        # Clear scope1
        count = self.store.clear(scope1)
        self.assertEqual(count, 3)

        # Verify scope1 is empty
        results1 = self.store.search_by_text("Entry", scope1)
        self.assertEqual(len(results1), 0)

        # Verify scope2 is unaffected
        results2 = self.store.search_by_text("Entry", scope2)
        self.assertEqual(len(results2), 2)

    def test_clear_empty_scope(self):
        """Test clearing an empty scope returns 0."""
        empty_scope = MemoryScope(user_id="nonexistent")
        count = self.store.clear(empty_scope)
        self.assertEqual(count, 0)

    def test_expired_entries_filtered(self):
        """Test that expired entries are filtered out of results."""
        entry = MemoryEntry(
            content="Expiring entry",
            memory_type=MemoryType.FACT,
            ttl_seconds=1,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=2),
        )
        self.store.upsert([entry], self.scope)

        results = self.store.search_by_text("Expiring entry", self.scope)
        self.assertEqual(len(results), 0)


class TestFileSystemStore(unittest.TestCase):
    """Tests for the file system store implementation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = FileSystemStore(self.temp_dir)
        self.scope = MemoryScope(user_id="test_user", session_id="test_session")

    def test_store_and_retrieve(self):
        """Test basic store and retrieve operations."""
        self.store.store("key1", {"value": "data1"}, self.scope)
        result = self.store.retrieve("key1", self.scope)
        self.assertEqual(result, {"value": "data1"})

    def test_retrieve_nonexistent(self):
        """Test retrieving a nonexistent key."""
        result = self.store.retrieve("nonexistent", self.scope)
        self.assertIsNone(result)

    def test_delete(self):
        """Test deleting an entry."""
        self.store.store("to_delete", {"value": "data"}, self.scope)
        result = self.store.delete("to_delete", self.scope)
        self.assertTrue(result)

        result = self.store.retrieve("to_delete", self.scope)
        self.assertIsNone(result)

    def test_list_keys(self):
        """Test listing all keys."""
        self.store.store("key1", {"value": "data1"}, self.scope)
        self.store.store("key2", {"value": "data2"}, self.scope)
        self.store.store("prefix_key3", {"value": "data3"}, self.scope)

        keys = self.store.list_keys(self.scope)
        self.assertEqual(len(keys), 3)

    def test_list_keys_with_prefix(self):
        """Test listing keys with a prefix filter."""
        self.store.store("task_1", {"value": "data1"}, self.scope)
        self.store.store("task_2", {"value": "data2"}, self.scope)
        self.store.store("other", {"value": "data3"}, self.scope)

        keys = self.store.list_keys(self.scope, prefix="task_")
        self.assertEqual(len(keys), 2)

    def test_clear(self):
        """Test clearing all entries for a scope."""
        self.store.store("key1", {"value": "data1"}, self.scope)
        self.store.store("key2", {"value": "data2"}, self.scope)

        count = self.store.clear(self.scope)
        self.assertEqual(count, 2)

        keys = self.store.list_keys(self.scope)
        self.assertEqual(len(keys), 0)

    def test_scope_path_prefix(self):
        """Test that scopes generate correct path prefixes."""
        scope1 = MemoryScope(user_id="user1", session_id="sess1")
        scope2 = MemoryScope(user_id="user2", session_id="sess2")

        self.store.store("key", {"value": "data1"}, scope1)
        self.store.store("key", {"value": "data2"}, scope2)

        result1 = self.store.retrieve("key", scope1)
        result2 = self.store.retrieve("key", scope2)

        self.assertEqual(result1, {"value": "data1"})
        self.assertEqual(result2, {"value": "data2"})


class TestMemoryScope(unittest.TestCase):
    """Tests for MemoryScope model."""

    def test_path_prefix_generation(self):
        """Test path prefix generation from scope attributes."""
        scope = MemoryScope(
            user_id="user123",
            session_id="sess456",
            org_id="org789",
        )
        prefix = scope.to_path_prefix()

        self.assertIn("user_user123", prefix)
        self.assertIn("session_sess456", prefix)
        self.assertIn("org_org789", prefix)

    def test_empty_scope_default_prefix(self):
        """Test that empty scope returns default prefix."""
        scope = MemoryScope()
        prefix = scope.to_path_prefix()
        self.assertEqual(prefix, "default")

    def test_filter_dict_generation(self):
        """Test filter dict generation for vector queries."""
        scope = MemoryScope(user_id="user123", namespace="workspace")
        filters = scope.to_filter_dict()

        self.assertEqual(filters["user_id"], "user123")
        self.assertEqual(filters["namespace"], "workspace")
        self.assertNotIn("session_id", filters)


class TestMemoryEntry(unittest.TestCase):
    """Tests for MemoryEntry model."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        entry = MemoryEntry(content="Test content")

        self.assertIsNotNone(entry.id)
        self.assertEqual(entry.memory_type, MemoryType.FACT)
        self.assertEqual(entry.priority, MemoryPriority.NORMAL)
        self.assertEqual(entry.access_count, 0)
        self.assertIsNone(entry.ttl_seconds)

    def test_expiration_check(self):
        """Test the is_expired method."""
        entry_no_ttl = MemoryEntry(content="No TTL")
        self.assertFalse(entry_no_ttl.is_expired())

        entry_expired = MemoryEntry(
            content="Expired",
            ttl_seconds=1,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=2),
        )
        self.assertTrue(entry_expired.is_expired())

        entry_valid = MemoryEntry(
            content="Valid",
            ttl_seconds=3600,
            created_at=datetime.now(timezone.utc),
        )
        self.assertFalse(entry_valid.is_expired())

    def test_touch_updates_access(self):
        """Test that touch() updates access timestamp and count."""
        entry = MemoryEntry(content="Touch test")
        original_count = entry.access_count
        original_accessed = entry.accessed_at

        time.sleep(0.01)
        entry.touch()

        self.assertEqual(entry.access_count, original_count + 1)
        self.assertGreater(entry.accessed_at, original_accessed)


class TestPromotableEntry(unittest.TestCase):
    """Tests for PromotableEntry model."""

    def test_touch_updates_tracking(self):
        """Test that touch() updates access tracking."""
        entry = MemoryEntry(content="Test")
        promotable = PromotableEntry(entry=entry)

        self.assertEqual(promotable.access_count, 0)
        self.assertIsNone(promotable.last_accessed_at)

        promotable.touch()

        self.assertEqual(promotable.access_count, 1)
        self.assertIsNotNone(promotable.last_accessed_at)

    def test_age_seconds(self):
        """Test age calculation."""
        entry = MemoryEntry(content="Test")
        promotable = PromotableEntry(
            entry=entry,
            stored_at=datetime.now(timezone.utc) - timedelta(seconds=60),
        )

        age = promotable.age_seconds()
        self.assertGreaterEqual(age, 60)
        self.assertLess(age, 62)


class TestWorkingMemoryCompaction(unittest.IsolatedAsyncioTestCase):
    """Tests for working memory compaction logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scope = MemoryScope(user_id="test_user", session_id="test_session")
        self.config = DatabaseMemoryConfig(
            short_term_base_path=self.temp_dir,
            auto_compact=False,
            compact_preserve_ratio=0.1,
        )
        self.store = FileSystemStore(self.temp_dir)
        self.vector_store = InMemoryVectorStore(embed_fn=mock_embed_fn, dimension=128)
        self.memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",  # 8192 context window
            config=self.config,
            short_term_store=self.store,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

    def _create_message(self, role: str, content: str, tool_calls=None):
        """Helper to create ChatMessage objects."""
        from motus.models import ChatMessage

        if role == "user":
            return ChatMessage.user_message(content)
        elif role == "assistant":
            msg = ChatMessage.assistant_message(content)
            if tool_calls:
                msg.tool_calls = tool_calls
            return msg
        elif role == "tool":
            return ChatMessage.tool_message(
                content, tool_call_id="test_tool_id", name="test_tool"
            )
        elif role == "system":
            return ChatMessage.system_message(content)
        else:
            raise ValueError(f"Unknown role: {role}")

    def test_find_token_budget_truncation_point_empty_messages(self):
        """Test with empty message list."""
        result = self.memory._find_token_budget_truncation_point([], 1000)
        self.assertEqual(result, 0)

    def test_find_token_budget_truncation_point_all_fit(self):
        """Test when all messages fit within budget."""
        messages = [
            self._create_message("user", "Hello"),
            self._create_message("assistant", "Hi there"),
            self._create_message("user", "How are you?"),
        ]
        self.memory._messages = messages

        # Large budget that fits all messages
        result = self.memory._find_token_budget_truncation_point(messages, 100000)
        self.assertEqual(result, 0)  # All fit, no truncation needed

    def test_find_token_budget_truncation_point_exceeds_budget(self):
        """Test finding cut point when messages exceed budget."""
        # Create messages with predictable sizes
        messages = [
            self._create_message("user", "A" * 100),  # ~25 tokens
            self._create_message("assistant", "B" * 100),  # ~25 tokens
            self._create_message("user", "C" * 100),  # ~25 tokens
            self._create_message("assistant", "D" * 100),  # ~25 tokens
        ]
        self.memory._messages = messages

        # Budget that only fits last 2 messages (~50 tokens)
        result = self.memory._find_token_budget_truncation_point(messages, 60)
        # Should cut somewhere in the middle, keeping recent messages
        self.assertGreater(result, 0)
        self.assertLess(result, len(messages))

    def test_find_token_budget_truncation_point_very_small_budget(self):
        """Test with budget smaller than any single message."""
        messages = [
            self._create_message("user", "A" * 1000),  # Large message
            self._create_message("assistant", "B" * 1000),
        ]
        self.memory._messages = messages

        # Very small budget
        result = self.memory._find_token_budget_truncation_point(messages, 10)
        # Should return index of last message (can't fit even one)
        self.assertEqual(result, len(messages))

    def test_find_safe_truncation_point_no_tool_messages(self):
        """Test safe truncation with no tool messages."""
        messages = [
            self._create_message("user", "Hello"),
            self._create_message("assistant", "Hi"),
            self._create_message("user", "Question"),
            self._create_message("assistant", "Answer"),
        ]

        # Any index should be safe (no tool messages to protect)
        result = self.memory._find_safe_truncation_point(messages, 2)
        self.assertEqual(result, 2)

    def test_find_safe_truncation_point_respects_tool_pairing(self):
        """Test that truncation doesn't split assistant/tool message groups."""
        from motus.models import ChatMessage, FunctionCall, ToolCall

        # Create assistant message with tool call
        assistant_msg = ChatMessage.assistant_message("Let me check that")
        assistant_msg.tool_calls = [
            ToolCall(
                id="tool_1",
                type="function",
                function=FunctionCall(name="search", arguments="{}"),
            )
        ]

        messages = [
            self._create_message("user", "Search for X"),
            assistant_msg,
            self._create_message("tool", "Search results here"),
            self._create_message("user", "Thanks"),
        ]

        # Target index 2 would cut in the middle of tool group
        # Should move back to index 1 (before assistant with tool_calls)
        result = self.memory._find_safe_truncation_point(messages, 2)
        self.assertEqual(result, 1)

    def test_find_safe_truncation_point_multiple_tool_calls(self):
        """Test with multiple consecutive tool messages."""
        from motus.models import ChatMessage, FunctionCall, ToolCall

        assistant_msg = ChatMessage.assistant_message("Running multiple tools")
        assistant_msg.tool_calls = [
            ToolCall(
                id="tool_1",
                type="function",
                function=FunctionCall(name="tool1", arguments="{}"),
            ),
            ToolCall(
                id="tool_2",
                type="function",
                function=FunctionCall(name="tool2", arguments="{}"),
            ),
        ]

        messages = [
            self._create_message("user", "Do tasks"),
            assistant_msg,
            self._create_message("tool", "Result 1"),
            self._create_message("tool", "Result 2"),
            self._create_message("user", "Great"),
        ]

        # Target index 3 would cut in middle of tool group
        result = self.memory._find_safe_truncation_point(messages, 3)
        self.assertEqual(result, 1)

    def test_find_safe_truncation_point_at_boundary(self):
        """Test truncation exactly at a safe boundary."""
        messages = [
            self._create_message("user", "First"),
            self._create_message("assistant", "Response"),
            self._create_message("user", "Second"),
        ]

        # Index 2 is a user message, should be safe
        result = self.memory._find_safe_truncation_point(messages, 2)
        self.assertEqual(result, 2)

    def test_find_safe_truncation_point_all_tool_messages(self):
        """Test edge case where all messages are tool messages."""
        messages = [
            self._create_message("tool", "Result 1"),
            self._create_message("tool", "Result 2"),
            self._create_message("tool", "Result 3"),
        ]

        # No safe point found, should return 0
        result = self.memory._find_safe_truncation_point(messages, 1)
        self.assertEqual(result, 0)

    async def test_compact_empty_messages(self):
        """Test compact with no messages."""
        result = await self.memory.compact()
        self.assertIsNone(result)

    async def test_compact_all_fit_within_budget(self):
        """Test compact when all messages fit within preserve budget."""
        # Add a few small messages
        await self.memory.add_message(self._create_message("user", "Hi"))
        await self.memory.add_message(self._create_message("assistant", "Hello"))

        # With gpt-4 (8192 context) and 0.1 ratio = 819 token budget
        # These tiny messages should fit easily
        result = await self.memory.compact()
        self.assertIsNone(result)
        self.assertEqual(len(self.memory.messages), 2)

    async def test_compact_with_preserve_ratio(self):
        """Test compaction using token-based ratio."""
        # Add many messages to exceed the preserve budget
        for i in range(50):
            await self.memory.add_message(
                self._create_message("user", f"Message {i} " + "X" * 200)
            )
            await self.memory.add_message(
                self._create_message("assistant", f"Response {i} " + "Y" * 200)
            )

        original_count = len(self.memory.messages)
        self.assertEqual(original_count, 100)

        # Compact with 10% preserve ratio
        result = await self.memory.compact(preserve_ratio=0.1)

        # Should have compacted (summarization result returned)
        self.assertIsNotNone(result)
        # Should have fewer messages now
        self.assertLess(len(self.memory.messages), original_count)

    async def test_compact_respects_tool_call_pairing(self):
        """Test that compact doesn't break tool call groups."""
        from motus.models import ChatMessage, FunctionCall, ToolCall

        # Add messages including tool calls
        for i in range(20):
            await self.memory.add_message(
                self._create_message("user", f"Question {i} " + "X" * 100)
            )
            assistant_msg = ChatMessage.assistant_message(f"Using tool {i}")
            assistant_msg.tool_calls = [
                ToolCall(
                    id=f"tool_{i}",
                    type="function",
                    function=FunctionCall(name="search", arguments="{}"),
                )
            ]
            await self.memory.add_message(assistant_msg)
            await self.memory.add_message(
                self._create_message("tool", f"Tool result {i} " + "Z" * 100)
            )

        # Compact
        result = await self.memory.compact(preserve_ratio=0.05)

        if result is not None:
            # Verify remaining messages don't start with a tool message
            # (which would indicate a broken tool call group)
            remaining = self.memory.messages
            if remaining:
                self.assertNotEqual(
                    remaining[0].role,
                    "tool",
                    "Compaction broke tool call pairing",
                )

    async def test_compact_stores_summary_in_short_term(self):
        """Test that compaction stores summary in short-term memory."""
        # Add enough messages to trigger compaction
        for i in range(30):
            await self.memory.add_message(
                self._create_message("user", f"Message {i} " + "X" * 300)
            )
            await self.memory.add_message(
                self._create_message("assistant", f"Response {i} " + "Y" * 300)
            )

        result = await self.memory.compact(preserve_ratio=0.05)

        if result is not None:
            # Check that summary was stored in short-term memory
            entries = self.memory.list_short_term_entries()
            summary_entries = [e for e in entries if e.get("promotable")]
            self.assertGreater(
                len(summary_entries), 0, "Summary should be stored as promotable entry"
            )

    async def test_compact_with_custom_preserve_ratio(self):
        """Test compaction with different preserve ratios."""
        # Add messages
        for i in range(40):
            await self.memory.add_message(
                self._create_message("user", f"Message {i} " + "X" * 200)
            )
            await self.memory.add_message(
                self._create_message("assistant", f"Response {i} " + "Y" * 200)
            )

        # Test with larger preserve ratio (should keep more messages)
        result = await self.memory.compact(preserve_ratio=0.5)

        if result is not None:
            remaining_50_percent = len(self.memory.messages)

            # Reset and test with smaller ratio
            self.memory._messages = []
            for i in range(40):
                await self.memory.add_message(
                    self._create_message("user", f"Message {i} " + "X" * 200)
                )
                await self.memory.add_message(
                    self._create_message("assistant", f"Response {i} " + "Y" * 200)
                )

            await self.memory.compact(preserve_ratio=0.1)
            remaining_10_percent = len(self.memory.messages)

            # Larger ratio should preserve more messages
            self.assertGreaterEqual(remaining_50_percent, remaining_10_percent)

    async def test_config_preserve_ratio_used_by_default(self):
        """Test that config's compact_preserve_ratio is used when not specified."""
        # Create memory with specific preserve ratio in config
        config = DatabaseMemoryConfig(
            short_term_base_path=self.temp_dir,
            auto_compact=False,
            compact_preserve_ratio=0.2,  # 20%
        )
        memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",
            config=config,
            short_term_store=self.store,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

        # Add messages
        for i in range(30):
            await memory.add_message(
                self._create_message("user", f"Message {i} " + "X" * 300)
            )
            await memory.add_message(
                self._create_message("assistant", f"Response {i} " + "Y" * 300)
            )

        # Compact without specifying ratio (should use config's 0.2)
        await memory.compact()

        # Verify it used the config ratio by checking message count
        # With 20% of 8192 = ~1638 token budget
        self.assertGreater(len(memory.messages), 0)


class TestMemoryReset(unittest.IsolatedAsyncioTestCase):
    """Tests for the Memory.reset() method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scope = MemoryScope(user_id="test_user", session_id="test_session")
        self.config = DatabaseMemoryConfig(
            short_term_base_path=self.temp_dir,
            auto_compact=False,
        )
        self.store = FileSystemStore(self.temp_dir)
        self.vector_store = InMemoryVectorStore(embed_fn=mock_embed_fn, dimension=128)
        self.memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",
            config=self.config,
            short_term_store=self.store,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

    def _create_message(self, role: str, content: str):
        """Helper to create ChatMessage objects."""
        from motus.models import ChatMessage

        if role == "user":
            return ChatMessage.user_message(content)
        elif role == "assistant":
            return ChatMessage.assistant_message(content)
        else:
            raise ValueError(f"Unknown role: {role}")

    async def test_reset_clears_working_memory(self):
        """Test that reset clears all messages from working memory."""
        await self.memory.add_message(self._create_message("user", "Hello"))
        await self.memory.add_message(self._create_message("assistant", "Hi there"))
        await self.memory.add_message(self._create_message("user", "How are you?"))

        self.assertEqual(len(self.memory.messages), 3)

        result = self.memory.reset()

        self.assertEqual(len(self.memory.messages), 0)
        self.assertEqual(result["messages"], 3)

    def test_reset_clears_short_term_memory(self):
        """Test that reset clears all short-term memory entries."""
        self.memory.remember_short_term("key1", "value1")
        self.memory.remember_short_term("key2", "value2")
        self.memory.remember_short_term("key3", "value3")

        keys_before = self.memory.list_short_term_keys()
        self.assertEqual(len(keys_before), 3)

        result = self.memory.reset()

        keys_after = self.memory.list_short_term_keys()
        self.assertEqual(len(keys_after), 0)
        self.assertEqual(result["short_term"], 3)

    def test_reset_clears_long_term_memory(self):
        """Test that reset clears all long-term memory entries."""
        self.memory.remember_long_term("Fact 1", memory_type=MemoryType.FACT)
        self.memory.remember_long_term("Fact 2", memory_type=MemoryType.FACT)
        self.memory.remember_long_term(
            "Preference 1", memory_type=MemoryType.PREFERENCE
        )

        # Verify entries exist
        results_before = self.memory.recall("Fact")
        self.assertGreater(len(results_before), 0)

        result = self.memory.reset()

        # Verify entries are cleared
        results_after = self.memory.recall("Fact")
        self.assertEqual(len(results_after), 0)
        self.assertEqual(result["long_term"], 3)

    async def test_reset_clears_all_memory_types(self):
        """Test that reset clears working, short-term, and long-term memory together."""
        # Add working memory
        await self.memory.add_message(self._create_message("user", "Hello"))
        await self.memory.add_message(self._create_message("assistant", "Hi"))

        # Add short-term memory
        self.memory.remember_short_term("session_data", {"key": "value"})
        self.memory.remember_short_term("temp_value", 42)

        # Add long-term memory
        self.memory.remember_long_term(
            "User prefers dark mode", memory_type=MemoryType.PREFERENCE
        )
        self.memory.remember_long_term("Important fact", memory_type=MemoryType.FACT)

        # Reset all
        result = self.memory.reset()

        # Verify all are cleared
        self.assertEqual(len(self.memory.messages), 0)
        self.assertEqual(len(self.memory.list_short_term_keys()), 0)
        self.assertEqual(len(self.memory.recall("mode")), 0)

        # Verify return values
        self.assertEqual(result["messages"], 2)
        self.assertEqual(result["short_term"], 2)
        self.assertEqual(result["long_term"], 2)

    def test_reset_on_empty_memory(self):
        """Test that reset on empty memory returns zeros."""
        result = self.memory.reset()

        self.assertEqual(result["messages"], 0)
        self.assertEqual(result["short_term"], 0)
        self.assertEqual(result["long_term"], 0)

    def test_reset_does_not_affect_other_scopes(self):
        """Test that reset only clears memory for the current scope."""
        # Create another memory instance with a different scope
        other_scope = MemoryScope(user_id="other_user", session_id="other_session")
        other_memory = DatabaseMemory(
            scope=other_scope,
            model_name="gpt-4",
            config=self.config,
            short_term_store=self.store,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

        # Add data to both scopes
        self.memory.remember_short_term("key1", "value1")
        other_memory.remember_short_term("key2", "value2")

        self.memory.remember_long_term(
            "Fact for test_user", memory_type=MemoryType.FACT
        )
        other_memory.remember_long_term(
            "Fact for other_user", memory_type=MemoryType.FACT
        )

        # Reset only the first memory
        self.memory.reset()

        # Verify first scope is cleared
        self.assertEqual(len(self.memory.list_short_term_keys()), 0)
        self.assertEqual(len(self.memory.recall("Fact")), 0)

        # Verify other scope is unaffected
        self.assertEqual(len(other_memory.list_short_term_keys()), 1)
        self.assertEqual(len(other_memory.recall("Fact")), 1)

    def test_reset_clears_promotable_entries(self):
        """Test that reset clears promotable entries in short-term memory."""
        entry = MemoryEntry(
            content="Promotable content",
            memory_type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
        )
        self.memory.remember_short_term_promotable("promotable_key", entry)

        entries_before = self.memory.list_short_term_entries()
        self.assertEqual(len(entries_before), 1)
        self.assertTrue(entries_before[0]["promotable"])

        result = self.memory.reset()

        entries_after = self.memory.list_short_term_entries()
        self.assertEqual(len(entries_after), 0)
        self.assertEqual(result["short_term"], 1)


class TestDatabaseMemoryConfig(unittest.IsolatedAsyncioTestCase):
    """Tests for DatabaseMemoryConfig behavior."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scope = MemoryScope(user_id="test_user", session_id="test_session")
        self.store = FileSystemStore(self.temp_dir)
        self.vector_store = InMemoryVectorStore(embed_fn=mock_embed_fn, dimension=128)

    def _create_message(self, role: str, content: str):
        """Helper to create ChatMessage objects."""
        from motus.models import ChatMessage

        if role == "user":
            return ChatMessage.user_message(content)
        elif role == "assistant":
            return ChatMessage.assistant_message(content)
        elif role == "system":
            return ChatMessage.system_message(content)
        else:
            raise ValueError(f"Unknown role: {role}")

    async def test_get_context_includes_memory_prompt(self):
        """Test that get_context includes memory prompt for DatabaseMemory."""
        system_prompt = "You are a helpful assistant."
        memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",
            short_term_store=self.store,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
            system_prompt=system_prompt,
        )

        await memory.add_message(self._create_message("user", "Hello"))

        context = memory.get_context()

        # First message should be system message with system prompt AND memory prompt
        self.assertEqual(context[0].role, "system")
        self.assertIn(system_prompt, context[0].content)
        # Should contain memory-related content
        self.assertIn("memory", context[0].content.lower())

    async def test_auto_compact_skipped_when_auto_compact_disabled(self):
        """Test that auto_compact is skipped when auto_compact is False."""
        config = DatabaseMemoryConfig(
            short_term_base_path=self.temp_dir,
            auto_compact=False,
            compact_safety_ratio=0.0001,  # Very low threshold to trigger compaction
        )
        memory = DatabaseMemory(
            scope=self.scope,
            model_name="gpt-4",
            config=config,
            short_term_store=self.store,
            long_term_store=self.vector_store,
            embed_fn=mock_embed_fn,
            summarize_fn=mock_summarize_fn,
        )

        # Add many messages that would normally trigger compaction
        for i in range(50):
            await memory.add_message(
                self._create_message("user", f"Message {i} " + "X" * 500)
            )
            await memory.add_message(
                self._create_message("assistant", f"Response {i} " + "Y" * 500)
            )

        # All messages should still be present (no compaction)
        self.assertEqual(len(memory.messages), 100)


if __name__ == "__main__":
    unittest.main()
