"""
Unit tests for BackgroundMemory.

Tests cover:
- MemoryChunk dataclass (id, fields, repr)
- initialize_memory_dir / write_raw_chunk on-disk format
- memory_tools: scoped read/write/edit/delete, glob/grep/list, path-traversal safety,
  raw_logs immutability
- BackgroundMemory construction + config defaults/overrides
- Compaction triggers MemoryUpdateAgent (with mocked agent class)
- build_tools() returns search_memory; disabled path returns []
- search_memory tool delegates to MemorySearchAgent
- Lifecycle hooks (on_agent_start no-op, on_agent_complete awaits update)
- Reset clears messages
- get_session_state + BackgroundSessionState round-trip via SessionState.from_dict
- restore() rehydrates a BackgroundMemory from a state snapshot
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from motus.memory.background.chunk import MemoryChunk
from motus.memory.background.memory_agent import (
    initialize_memory_dir,
    write_raw_chunk,
)
from motus.memory.background.memory_tools import make_memory_tools
from motus.memory.background_memory import BackgroundMemory
from motus.memory.config import BackgroundMemoryConfig
from motus.memory.session_state import (
    BackgroundSessionState,
    CompactionSessionState,
    SessionState,
)
from motus.models import ChatMessage


def _make_memory(root: Path, **kwargs) -> BackgroundMemory:
    """Create a BackgroundMemory rooted at a temp path. Client is a MagicMock."""
    client = MagicMock()
    return BackgroundMemory(
        memory_client=client,
        memory_model_name="test-model",
        root=root,
        **kwargs,
    )


# =============================================================================
# MemoryChunk
# =============================================================================


class TestMemoryChunk(unittest.TestCase):
    """Tests for the MemoryChunk dataclass."""

    def test_create_generates_8_char_id(self):
        chunk = MemoryChunk.create(messages=[], turn_start=0, turn_end=0)
        self.assertIsInstance(chunk.chunk_id, str)
        self.assertEqual(len(chunk.chunk_id), 8)

    def test_create_generates_unique_ids(self):
        ids = {
            MemoryChunk.create(messages=[], turn_start=0, turn_end=0).chunk_id
            for _ in range(50)
        }
        # 8 hex chars → collisions possible but astronomically unlikely at N=50
        self.assertGreater(len(ids), 45)

    def test_create_sets_fields(self):
        msgs = [ChatMessage.user_message("hi")]
        chunk = MemoryChunk.create(messages=msgs, turn_start=0, turn_end=0)
        self.assertEqual(chunk.messages, msgs)
        self.assertEqual(chunk.turn_start, 0)
        self.assertEqual(chunk.turn_end, 0)

    def test_timestamp_is_set(self):
        chunk = MemoryChunk.create(messages=[], turn_start=0, turn_end=0)
        self.assertIsNotNone(chunk.timestamp)

    def test_repr(self):
        chunk = MemoryChunk.create(
            messages=[ChatMessage.user_message("a"), ChatMessage.user_message("b")],
            turn_start=0,
            turn_end=1,
        )
        r = repr(chunk)
        self.assertIn(chunk.chunk_id, r)
        self.assertIn("0-1", r)
        self.assertIn("messages=2", r)


# =============================================================================
# initialize_memory_dir
# =============================================================================


class TestInitializeMemoryDir(unittest.TestCase):
    """Tests for initialize_memory_dir."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.root = self.tmpdir / "memory"

    def test_creates_root_and_raw_logs(self):
        initialize_memory_dir(self.root)
        self.assertTrue(self.root.is_dir())
        self.assertTrue((self.root / "raw_logs").is_dir())

    def test_creates_memory_md_when_absent(self):
        initialize_memory_dir(self.root)
        entry = self.root / "memory.md"
        self.assertTrue(entry.is_file())
        self.assertIn("No facts stored yet", entry.read_text())

    def test_idempotent_preserves_existing_memory_md(self):
        initialize_memory_dir(self.root)
        (self.root / "memory.md").write_text("# Custom\n- fact [abc123:0-0]\n")

        # Second init should not clobber the existing file
        initialize_memory_dir(self.root)
        self.assertIn("Custom", (self.root / "memory.md").read_text())


# =============================================================================
# write_raw_chunk
# =============================================================================


class TestWriteRawChunk(unittest.TestCase):
    """Tests for write_raw_chunk."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        initialize_memory_dir(self.root)

    def test_writes_file_under_raw_logs(self):
        chunk = MemoryChunk.create(
            messages=[ChatMessage.user_message("hi")], turn_start=0, turn_end=0
        )
        write_raw_chunk(self.root, chunk)
        path = self.root / "raw_logs" / f"{chunk.chunk_id}.md"
        self.assertTrue(path.exists())

    def test_meta_block_has_chunk_id_and_count(self):
        msgs = [ChatMessage.user_message("a"), ChatMessage.assistant_message("b")]
        chunk = MemoryChunk.create(messages=msgs, turn_start=0, turn_end=1)
        write_raw_chunk(self.root, chunk)

        content = (self.root / "raw_logs" / f"{chunk.chunk_id}.md").read_text()
        self.assertIn("<meta>", content)
        self.assertIn(chunk.chunk_id, content)
        self.assertIn('"num_messages": 2', content)
        self.assertIn("<messages>", content)
        self.assertIn("[0] user:", content)
        self.assertIn("[1] assistant:", content)

    def test_immutable_skip_if_exists(self):
        """Second write_raw_chunk with same id must not overwrite."""
        chunk = MemoryChunk.create(
            messages=[ChatMessage.user_message("original")],
            turn_start=0,
            turn_end=0,
        )
        write_raw_chunk(self.root, chunk)
        path = self.root / "raw_logs" / f"{chunk.chunk_id}.md"
        first = path.read_text()

        # Mutate chunk and write again — file should be unchanged
        chunk.messages = [ChatMessage.user_message("tampered")]
        write_raw_chunk(self.root, chunk)
        self.assertEqual(path.read_text(), first)

    def test_escapes_newlines_in_content(self):
        chunk = MemoryChunk.create(
            messages=[ChatMessage.user_message("line1\nline2\nline3")],
            turn_start=0,
            turn_end=0,
        )
        write_raw_chunk(self.root, chunk)
        content = (self.root / "raw_logs" / f"{chunk.chunk_id}.md").read_text()
        # Raw log preserves indexing; newlines become \\n so each message is one line
        self.assertIn(r"line1\nline2\nline3", content)


# =============================================================================
# memory_tools — path safety + raw_logs immutability
# =============================================================================


class TestMemoryToolsPathSafety(unittest.IsolatedAsyncioTestCase):
    """Scoped file tools must reject paths that escape the memory root."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        initialize_memory_dir(self.root)
        self.tools = {t.__name__: t for t in make_memory_tools(self.root)}

    async def test_read_file_rejects_parent_escape(self):
        with self.assertRaises(ValueError):
            await self.tools["read_file"]("../../etc/passwd")

    async def test_write_file_rejects_absolute_path_outside_root(self):
        with self.assertRaises(ValueError):
            await self.tools["write_file"]("/etc/passwd", "hacked")

    async def test_delete_file_blocks_raw_logs(self):
        # Create a raw log file
        raw = self.root / "raw_logs" / "test.md"
        raw.write_text("content")

        result = await self.tools["delete_file"]("raw_logs/test.md")
        self.assertIn("cannot delete raw logs", result)
        self.assertTrue(raw.exists())

    async def test_delete_file_refuses_directories(self):
        result = await self.tools["delete_file"]("raw_logs")
        self.assertIn("not a file", result)


class TestMemoryToolsReadWrite(unittest.IsolatedAsyncioTestCase):
    """Happy-path read/write/edit/delete."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        initialize_memory_dir(self.root)
        self.tools = {t.__name__: t for t in make_memory_tools(self.root)}

    async def test_write_then_read(self):
        await self.tools["write_file"]("preferences.md", "# Preferences\n- foo\n")
        content = await self.tools["read_file"]("preferences.md")
        # read_file adds line-number prefix
        self.assertIn("# Preferences", content)
        self.assertIn("- foo", content)

    async def test_write_creates_parent_dirs(self):
        await self.tools["write_file"]("projects/phoenix.md", "deadline: June 15")
        self.assertTrue((self.root / "projects" / "phoenix.md").exists())

    async def test_read_file_not_found(self):
        content = await self.tools["read_file"]("does-not-exist.md")
        self.assertIn("Error", content)
        self.assertIn("not found", content)

    async def test_edit_file_unique_replacement(self):
        await self.tools["write_file"]("a.md", "hello world\nfoo bar\n")
        msg = await self.tools["edit_file"]("a.md", "hello world", "HELLO WORLD")
        self.assertIn("Edited", msg)
        self.assertIn("HELLO WORLD", (self.root / "a.md").read_text())

    async def test_edit_file_ambiguous_multi_match(self):
        await self.tools["write_file"]("a.md", "cat\ncat\ncat\n")
        result = await self.tools["edit_file"]("a.md", "cat", "dog")
        self.assertIn("appears", result)
        self.assertIn("3 times", result)
        # unchanged
        self.assertEqual((self.root / "a.md").read_text().count("cat"), 3)

    async def test_edit_file_replace_all(self):
        await self.tools["write_file"]("a.md", "cat cat cat")
        await self.tools["edit_file"]("a.md", "cat", "dog", replace_all=True)
        self.assertEqual((self.root / "a.md").read_text(), "dog dog dog")

    async def test_edit_file_text_not_found(self):
        await self.tools["write_file"]("a.md", "hello")
        result = await self.tools["edit_file"]("a.md", "nope", "yes")
        self.assertIn("not found", result)

    async def test_delete_file_happy_path(self):
        await self.tools["write_file"]("temp.md", "scratch")
        result = await self.tools["delete_file"]("temp.md")
        self.assertIn("Deleted", result)
        self.assertFalse((self.root / "temp.md").exists())


class TestMemoryToolsSearch(unittest.IsolatedAsyncioTestCase):
    """glob_search, grep_search, list_files."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        initialize_memory_dir(self.root)
        self.tools = {t.__name__: t for t in make_memory_tools(self.root)}

    async def test_glob_search_finds_markdown(self):
        await self.tools["write_file"]("projects/phoenix.md", "x")
        await self.tools["write_file"]("projects/atlas.md", "y")
        await self.tools["write_file"]("other.txt", "z")

        result = await self.tools["glob_search"]("projects/*.md")
        self.assertIn("phoenix.md", result)
        self.assertIn("atlas.md", result)
        self.assertNotIn("other.txt", result)

    async def test_glob_search_no_matches(self):
        result = await self.tools["glob_search"]("nope/*.md")
        self.assertIn("No files", result)

    async def test_grep_search_finds_substring(self):
        await self.tools["write_file"]("preferences.md", "- vim\n- dark mode\n")
        await self.tools["write_file"]("projects.md", "- phoenix\n")

        result = await self.tools["grep_search"]("vim")
        self.assertIn("preferences.md", result)
        self.assertNotIn("projects.md", result)

    async def test_list_files_returns_entries(self):
        await self.tools["write_file"]("a.md", "1")
        await self.tools["write_file"]("subdir/b.md", "2")

        result = await self.tools["list_files"](".")
        self.assertIn("a.md", result)
        self.assertIn("subdir", result)

    async def test_list_files_empty_dir(self):
        (self.root / "empty").mkdir()
        result = await self.tools["list_files"]("empty")
        self.assertIn("empty", result)


# =============================================================================
# BackgroundMemory — construction + config
# =============================================================================


class TestBackgroundMemoryInit(unittest.TestCase):
    """Construction, defaults, and config overrides."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_default_config_applied(self):
        """Without bg_config, defaults should flow in from BackgroundMemoryConfig."""
        mem = _make_memory(root=self.root)
        self.assertEqual(mem._memory_max_steps, 30)
        self.assertTrue(mem._enable_memory_tools)

    def test_direct_root_override_wins(self):
        """root= argument takes precedence over bg_config.root."""
        other_root = Path(tempfile.mkdtemp())
        mem = _make_memory(
            root=other_root,
            bg_config=BackgroundMemoryConfig(root=str(self.root)),
        )
        self.assertEqual(mem._memory_root, other_root)

    def test_expanduser_applied_to_root(self):
        """'~/.motus/memory' should expand."""
        mem = BackgroundMemory(
            memory_client=MagicMock(),
            memory_model_name="test",
            root=Path("~/.motus/test-memory"),
        )
        self.assertNotIn("~", str(mem._memory_root))

    def test_initializes_memory_dir_on_init(self):
        mem = _make_memory(root=self.root)
        self.assertTrue((mem._memory_root / "raw_logs").is_dir())
        self.assertTrue((mem._memory_root / "memory.md").is_file())

    def test_disable_memory_tools(self):
        mem = _make_memory(root=self.root, enable_memory_tools=False)
        self.assertEqual(mem.build_tools(), [])

    def test_override_max_steps(self):
        mem = _make_memory(root=self.root, memory_agent_max_steps=5)
        self.assertEqual(mem._memory_max_steps, 5)


# =============================================================================
# Compaction-triggered memory update
# =============================================================================


class TestBackgroundMemoryCompaction(unittest.IsolatedAsyncioTestCase):
    """_do_compact should fire a MemoryUpdateAgent on a MemoryChunk."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    async def test_no_update_before_compaction(self):
        mem = _make_memory(root=self.root)
        for _ in range(5):
            await mem.add_message(ChatMessage.user_message("hi"))
            await mem.add_message(ChatMessage.assistant_message("hello"))
        self.assertIsNone(mem._update_task)

    @patch("motus.memory.background_memory.MemoryUpdateAgent")
    async def test_update_fires_on_do_compact(self, MockUpdateAgent):
        mock_run = AsyncMock()
        MockUpdateAgent.return_value.run = mock_run

        mem = _make_memory(root=self.root, compact_fn=lambda m, s: "summary")
        await mem.add_message(ChatMessage.user_message("hello"))
        await mem.add_message(ChatMessage.assistant_message("hi"))

        await mem._do_compact()
        await asyncio.sleep(0.01)  # let the fire-and-forget task run

        MockUpdateAgent.assert_called_once()
        mock_run.assert_awaited_once()
        chunk = mock_run.call_args[0][0]
        self.assertIsInstance(chunk, MemoryChunk)
        self.assertEqual(len(chunk.messages), 2)

    @patch("motus.memory.background_memory.MemoryUpdateAgent")
    async def test_chunk_captures_all_pre_compaction_messages(self, MockUpdateAgent):
        MockUpdateAgent.return_value.run = AsyncMock()

        mem = _make_memory(root=self.root, compact_fn=lambda m, s: "summary")
        for _ in range(5):
            await mem.add_message(ChatMessage.user_message("u"))
            await mem.add_message(ChatMessage.assistant_message("a"))

        await mem._do_compact()
        await asyncio.sleep(0.01)

        chunk = MockUpdateAgent.return_value.run.call_args[0][0]
        self.assertEqual(len(chunk.messages), 10)

    @patch("motus.memory.background_memory.MemoryUpdateAgent")
    async def test_context_compacted_after_do_compact(self, MockUpdateAgent):
        """Main agent's _messages should be compacted after _do_compact."""
        MockUpdateAgent.return_value.run = AsyncMock()

        mem = _make_memory(root=self.root, compact_fn=lambda m, s: "summary")
        for _ in range(3):
            await mem.add_message(ChatMessage.user_message("u"))
            await mem.add_message(ChatMessage.assistant_message("a"))

        await mem._do_compact()

        # Unit C (last msg is assistant) → full compact, 1 continuation message
        self.assertEqual(len(mem.messages), 1)
        self.assertIn("context_summary", mem.messages[0].content)

    @patch("motus.memory.background_memory.MemoryUpdateAgent")
    async def test_update_task_handle_stored(self, MockUpdateAgent):
        """_update_task should be set to the background asyncio task."""
        MockUpdateAgent.return_value.run = AsyncMock()

        mem = _make_memory(root=self.root, compact_fn=lambda m, s: "summary")
        await mem.add_message(ChatMessage.user_message("hi"))
        await mem.add_message(ChatMessage.assistant_message("hello"))

        self.assertIsNone(mem._update_task)
        await mem._do_compact()
        self.assertIsNotNone(mem._update_task)
        self.assertTrue(mem._update_task.get_name().startswith("memory-update-"))
        await mem._update_task  # drain

    @patch("motus.memory.background_memory.MemoryUpdateAgent")
    async def test_second_compaction_waits_for_first(self, MockUpdateAgent):
        """Sequential update processing: second _do_compact must await the first task."""
        # First call: run returns something slow; second call: quick
        slow_event = asyncio.Event()

        async def slow_run(chunk):
            await slow_event.wait()

        MockUpdateAgent.return_value.run = AsyncMock(side_effect=slow_run)

        mem = _make_memory(root=self.root, compact_fn=lambda m, s: "summary")
        await mem.add_message(ChatMessage.user_message("hi"))
        await mem.add_message(ChatMessage.assistant_message("hello"))

        await mem._do_compact()
        first_task = mem._update_task
        self.assertFalse(first_task.done())

        # Kick off second compaction in a task so we can observe it blocks
        await mem.add_message(ChatMessage.user_message("more"))
        await mem.add_message(ChatMessage.assistant_message("response"))
        second = asyncio.create_task(mem._do_compact())

        # Let the scheduler run — second must still be waiting because first is stuck
        await asyncio.sleep(0.05)
        self.assertFalse(second.done())

        slow_event.set()
        await second  # now it can complete


# =============================================================================
# build_tools() and search_memory
# =============================================================================


class TestBackgroundMemoryTools(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_build_tools_returns_search_memory(self):
        mem = _make_memory(root=self.root)
        tools = mem.build_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].__name__, "search_memory")

    def test_build_tools_disabled(self):
        mem = _make_memory(root=self.root, enable_memory_tools=False)
        self.assertEqual(mem.build_tools(), [])

    @patch("motus.memory.background_memory.MemorySearchAgent")
    async def test_search_memory_delegates_to_search_agent(self, MockSearchAgent):
        MockSearchAgent.return_value.run = AsyncMock(return_value="the answer")

        mem = _make_memory(root=self.root)
        search_memory = mem.build_tools()[0]
        result = await search_memory("what is foo?")

        MockSearchAgent.assert_called_once()
        MockSearchAgent.return_value.run.assert_awaited_once_with("what is foo?")
        self.assertEqual(result, "the answer")

    @patch("motus.memory.background_memory.MemorySearchAgent")
    async def test_each_search_gets_fresh_agent(self, MockSearchAgent):
        """Each search_memory() call should instantiate a new MemorySearchAgent."""
        MockSearchAgent.return_value.run = AsyncMock(return_value="ok")

        mem = _make_memory(root=self.root)
        search_memory = mem.build_tools()[0]
        await search_memory("q1")
        await search_memory("q2")

        self.assertEqual(MockSearchAgent.call_count, 2)


# =============================================================================
# Lifecycle hooks
# =============================================================================


class TestBackgroundMemoryLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    async def test_on_agent_start_is_noop(self):
        mem = _make_memory(root=self.root)
        await mem.on_agent_start()  # must not raise

    async def test_on_agent_complete_without_task_is_noop(self):
        mem = _make_memory(root=self.root)
        await mem.on_agent_complete()  # no update task exists

    @patch("motus.memory.background_memory.MemoryUpdateAgent")
    async def test_on_agent_complete_awaits_pending_update(self, MockUpdateAgent):
        MockUpdateAgent.return_value.run = AsyncMock()

        mem = _make_memory(root=self.root, compact_fn=lambda m, s: "summary")
        await mem.add_message(ChatMessage.user_message("hi"))
        await mem.add_message(ChatMessage.assistant_message("hello"))
        await mem._do_compact()
        self.assertIsNotNone(mem._update_task)

        await mem.on_agent_complete()
        self.assertTrue(mem._update_task.done())

    @patch("motus.memory.background_memory.MemoryUpdateAgent")
    async def test_on_agent_complete_swallows_exceptions(self, MockUpdateAgent):
        """Errors in the update task must not propagate out of on_agent_complete."""
        MockUpdateAgent.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))

        mem = _make_memory(root=self.root, compact_fn=lambda m, s: "summary")
        await mem.add_message(ChatMessage.user_message("hi"))
        await mem.add_message(ChatMessage.assistant_message("hello"))
        await mem._do_compact()

        # Must not raise
        await mem.on_agent_complete()


# =============================================================================
# Reset
# =============================================================================


class TestBackgroundMemoryReset(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    async def test_reset_clears_messages(self):
        mem = _make_memory(root=self.root)
        for _ in range(3):
            await mem.add_message(ChatMessage.user_message("hi"))
            await mem.add_message(ChatMessage.assistant_message("hello"))

        result = mem.reset()
        self.assertEqual(result["messages"], 6)
        self.assertEqual(len(mem.messages), 0)

    async def test_reset_does_not_touch_disk(self):
        """reset() should clear in-memory state but leave the on-disk tree intact."""
        mem = _make_memory(root=self.root)
        (mem._memory_root / "preferences.md").write_text("- something")
        mem.reset()
        self.assertTrue((mem._memory_root / "preferences.md").exists())


# =============================================================================
# Session state
# =============================================================================


class TestBackgroundMemorySessionState(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    async def test_get_session_state_returns_background_subclass(self):
        mem = _make_memory(root=self.root)
        mem.set_system_prompt("sys prompt")
        await mem.add_message(ChatMessage.user_message("hi"))

        state = mem.get_session_state()
        self.assertIsInstance(state, BackgroundSessionState)
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(state.system_prompt, "sys prompt")
        self.assertEqual(state.tree_root, str(self.root))

    async def test_to_dict_has_type_and_tree_root(self):
        mem = _make_memory(root=self.root)
        await mem.add_message(ChatMessage.user_message("hi"))

        d = mem.get_session_state().to_dict()
        self.assertEqual(d["type"], "BackgroundSessionState")
        self.assertEqual(d["tree_root"], str(self.root))
        self.assertIn("messages", d)
        self.assertIn("system_prompt", d)

    async def test_roundtrip_through_session_state_from_dict(self):
        """to_dict → from_dict should reconstruct a BackgroundSessionState."""
        mem = _make_memory(root=self.root)
        mem.set_system_prompt("sys")
        await mem.add_message(ChatMessage.user_message("a"))
        await mem.add_message(ChatMessage.assistant_message("b"))

        d = mem.get_session_state().to_dict()
        restored_state = SessionState.from_dict(d)

        self.assertIsInstance(restored_state, BackgroundSessionState)
        self.assertEqual(restored_state.tree_root, str(self.root))
        self.assertEqual(restored_state.system_prompt, "sys")
        self.assertEqual(len(restored_state.messages), 2)

    def test_from_dict_registry_dispatch(self):
        """from_dict dispatches by 'type' field, not by base class."""
        d_bg = {
            "type": "BackgroundSessionState",
            "messages": [],
            "system_prompt": "",
            "tree_root": "/tmp/x",
        }
        d_comp = {
            "type": "CompactionSessionState",
            "messages": [],
            "system_prompt": "",
            "session_id": "s1",
            "log_base_path": None,
            "compaction_count": 0,
        }

        self.assertIsInstance(SessionState.from_dict(d_bg), BackgroundSessionState)
        self.assertIsInstance(SessionState.from_dict(d_comp), CompactionSessionState)

    def test_from_dict_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            SessionState.from_dict({"type": "NotARealType", "messages": []})


# =============================================================================
# Restore
# =============================================================================


class TestBackgroundMemoryRestore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    async def test_restore_rehydrates_messages_and_system_prompt(self):
        mem = _make_memory(root=self.root)
        mem.set_system_prompt("original")
        await mem.add_message(ChatMessage.user_message("hi"))
        await mem.add_message(ChatMessage.assistant_message("hello"))

        state = mem.get_session_state()
        restored = BackgroundMemory.restore(
            state,
            memory_client=MagicMock(),
            memory_model_name="test-model",
        )
        self.assertEqual(restored._system_prompt, "original")
        self.assertEqual(len(restored.messages), 2)

    async def test_restore_reconnects_to_tree_root(self):
        """Restored memory should point at the same tree_root from the state."""
        mem = _make_memory(root=self.root)
        (mem._memory_root / "preferences.md").write_text("- vim")

        state = mem.get_session_state()
        restored = BackgroundMemory.restore(
            state,
            memory_client=MagicMock(),
            memory_model_name="test-model",
        )

        self.assertEqual(restored._memory_root, self.root)
        # Fact file should still be reachable via the restored memory
        self.assertTrue((restored._memory_root / "preferences.md").exists())

    def test_restore_rejects_wrong_state_type(self):
        """Passing a non-BackgroundSessionState must raise TypeError."""
        with self.assertRaises(TypeError):
            BackgroundMemory.restore(
                SessionState(messages=[], system_prompt=""),
                memory_client=MagicMock(),
                memory_model_name="test-model",
            )


if __name__ == "__main__":
    unittest.main()
