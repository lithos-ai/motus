"""
Unit tests for CompactionMemory.

Tests cover:
- Basic message management (add, clear, get_context)
- Token estimation
- Manual compaction
- Continuation message format
- Custom compact_fn injection
- Short-term/long-term NotImplementedError
- Trace logging
- Reset behavior
- Auto-compact safety (skip when mid-tool-call)
- Conversation log persistence
- Search tools
"""

import json
import tempfile
import unittest
from typing import List

from motus.memory.compaction_memory import CompactionMemory
from motus.memory.config import CompactionMemoryConfig
from motus.models import ChatMessage
from motus.models.base import FunctionCall, ToolCall


def mock_compact_fn(messages: List[ChatMessage], system_prompt: str) -> str:
    """Mock compaction function that returns a simple summary."""
    msg_count = len(messages)
    return f"Summary of {msg_count} messages. System context: {system_prompt[:50]}"


class TestCompactionMemoryBasic(unittest.IsolatedAsyncioTestCase):
    """Tests for basic working memory operations."""

    def setUp(self):
        self.memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.memory.set_system_prompt("You are a test assistant.")

    async def test_add_and_get_messages(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        self.assertEqual(len(self.memory.messages), 1)
        self.assertEqual(self.memory.messages[0].content, "Hello")

    async def test_get_context_includes_system_prompt(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        context = self.memory.get_context()
        self.assertEqual(len(context), 2)
        self.assertEqual(context[0].role, "system")
        self.assertEqual(context[0].content, "You are a test assistant.")
        self.assertEqual(context[1].role, "user")

    async def test_clear_messages(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        self.memory.clear_messages()
        self.assertEqual(len(self.memory.messages), 0)

    def test_set_system_prompt(self):
        self.memory.set_system_prompt("New prompt")
        self.assertEqual(self.memory.construct_system_prompt().content, "New prompt")

    async def test_messages_returns_copy(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        msgs = self.memory.messages
        msgs.append(ChatMessage.user_message("Injected"))
        # Original unchanged
        self.assertEqual(len(self.memory.messages), 1)


class TestCompactionMemoryCompaction(unittest.IsolatedAsyncioTestCase):
    """Tests for the compaction mechanism."""

    def setUp(self):
        self.memory = CompactionMemory(
            config=CompactionMemoryConfig(safety_ratio=0.75),
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.memory.set_system_prompt("You are a test assistant.")

    async def test_manual_compact(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.add_message(ChatMessage.assistant_message("Hi there!"))
        summary = await self.memory.compact()

        self.assertIsNotNone(summary)
        # Unit C (assistant, no tool calls): full compact, no replay
        self.assertIn("2 messages", summary)

        # After compaction: [continuation] only — Unit C summarizes everything
        msgs = self.memory.messages
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, "user")
        self.assertIn("continued from a previous conversation", msgs[0].content)

    async def test_compact_empty_returns_none(self):
        result = await self.memory.compact()
        self.assertIsNone(result)

    async def test_continuation_message_format(self):
        await self.memory.add_message(ChatMessage.user_message("Do X"))
        await self.memory.compact()

        content = self.memory.messages[0].content
        self.assertIn("<context_summary>", content)
        self.assertIn("</context_summary>", content)
        self.assertIn("continue the conversation", content.lower())

    async def test_compaction_count_tracked(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.compact()
        self.assertEqual(self.memory._compaction_count, 1)

        await self.memory.add_message(ChatMessage.user_message("More"))
        await self.memory.compact()
        self.assertEqual(self.memory._compaction_count, 2)

    async def test_custom_compact_fn(self):
        """Test that custom compact_fn is used."""

        def custom_fn(msgs, sys_prompt):
            return "CUSTOM SUMMARY"

        memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=custom_fn,
        )
        await memory.add_message(ChatMessage.user_message("Hello"))
        summary = await memory.compact()
        self.assertEqual(summary, "CUSTOM SUMMARY")

    async def test_on_compact_callback(self):
        """Test that on_compact callback is invoked with stats."""
        callback_calls = []

        memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
            on_compact=lambda stats: callback_calls.append(stats),
        )
        memory.set_system_prompt("Test")
        await memory.add_message(ChatMessage.user_message("Hello"))
        await memory.add_message(ChatMessage.assistant_message("Hi"))
        await memory.compact()

        self.assertEqual(len(callback_calls), 1)
        stats = callback_calls[0]
        # Unit C: full compact — both messages summarized, no replay
        self.assertEqual(stats["messages_compacted"], 2)
        self.assertEqual(stats["messages_replayed"], 0)
        self.assertEqual(stats["compaction_number"], 1)
        self.assertIn("summary_tokens", stats)
        self.assertIn("summary", stats)
        self.assertIn("2 messages", stats["summary"])

    async def test_on_compact_not_called_when_empty(self):
        """on_compact should NOT be called when compact() returns None."""
        callback_calls = []

        memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
            on_compact=lambda stats: callback_calls.append(stats),
        )
        result = await memory.compact()

        self.assertIsNone(result)
        self.assertEqual(len(callback_calls), 0)

    async def test_context_after_compaction(self):
        """After compaction, get_context returns [system, continuation]."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.add_message(ChatMessage.assistant_message("Hi"))
        await self.memory.add_message(ChatMessage.user_message("Do something"))
        await self.memory.compact()

        context = self.memory.get_context()
        self.assertEqual(len(context), 2)
        self.assertEqual(context[0].role, "system")
        self.assertEqual(context[1].role, "user")
        self.assertIn("<context_summary>", context[1].content)

    async def test_compact_preserves_system_prompt_in_summary(self):
        """Compact fn receives the system prompt for context."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        summary = await self.memory.compact()
        self.assertIn("You are a test assistant", summary)


class TestCompactionMemorySafety(unittest.TestCase):
    """Tests for auto-compact safety (skip when mid-tool-call)."""

    def setUp(self):
        self.memory = CompactionMemory(
            config=CompactionMemoryConfig(safety_ratio=0.75),
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.memory.set_system_prompt("Test")

    def test_is_safe_to_compact_empty(self):
        self.assertFalse(self.memory._is_at_boundary())

    def test_is_safe_to_compact_after_user_message(self):
        self.memory._messages.append(ChatMessage.user_message("Hello"))
        self.assertTrue(self.memory._is_at_boundary())

    def test_not_safe_to_compact_after_assistant_with_tool_calls(self):
        """Should NOT compact when last msg is assistant with pending tool results."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test_tool", arguments="{}"),
        )
        self.memory._messages.append(
            ChatMessage.assistant_message(content=None, tool_calls=[tool_call])
        )
        self.memory._pending_tool_calls = 1
        self.assertFalse(self.memory._is_at_boundary())

    def test_safe_to_compact_after_tool_result(self):
        """Should be a boundary after all tool results are received."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test_tool", arguments="{}"),
        )
        self.memory._messages.append(
            ChatMessage.assistant_message(content=None, tool_calls=[tool_call])
        )
        self.memory._messages.append(
            ChatMessage.tool_message("result", "call_123", "test_tool")
        )
        self.memory._pending_tool_calls = 0
        self.assertTrue(self.memory._is_at_boundary())

    def test_safe_to_compact_after_assistant_without_tool_calls(self):
        """Should be a boundary after assistant message without tool_calls."""
        self.memory._messages.append(
            ChatMessage.assistant_message("Just text, no tools")
        )
        self.assertTrue(self.memory._is_at_boundary())


class TestCompactionMemoryDefaults(unittest.TestCase):
    """Tests for CompactionMemory default behaviors."""

    def setUp(self):
        self.memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

    def test_build_tools_empty_without_log_store(self):
        """build_tools should return empty list when no log store is set."""
        tools = self.memory.build_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 0)

    def test_build_tools_disabled(self):
        """build_tools should return empty list when enable_memory_tools=False."""
        memory = CompactionMemory(
            config=CompactionMemoryConfig(log_base_path=tempfile.mkdtemp()),
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
            enable_memory_tools=False,
        )
        self.assertEqual(memory.build_tools(), [])


class TestCompactionMemoryTrace(unittest.IsolatedAsyncioTestCase):
    """Tests for trace/debugging support."""

    def setUp(self):
        self.memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

    async def test_trace_records_messages(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        trace = self.memory.get_memory_trace()
        self.assertEqual(trace["total_events"], 1)
        self.assertEqual(trace["events"][0]["role"], "user")

    async def test_trace_records_compaction(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.compact()
        trace = self.memory.get_memory_trace()
        self.assertEqual(trace["total_compactions"], 1)

    async def test_trace_records_tool_calls(self):
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="bash", arguments='{"command": "ls"}'),
        )
        await self.memory.add_message(
            ChatMessage.assistant_message(content=None, tool_calls=[tool_call])
        )
        trace = self.memory.get_memory_trace()
        event = trace["events"][0]
        self.assertIn("tool_calls", event)
        self.assertEqual(event["tool_calls"][0]["name"], "bash")

    async def test_trace_records_tool_results(self):
        await self.memory.add_message(
            ChatMessage.tool_message("file1.py\nfile2.py", "call_123", "bash")
        )
        trace = self.memory.get_memory_trace()
        event = trace["events"][0]
        self.assertEqual(event["tool_name"], "bash")

    async def test_reset_clears_trace(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        self.memory.reset()
        trace = self.memory.get_memory_trace()
        self.assertEqual(trace["total_events"], 0)


class TestCompactionMemoryReset(unittest.IsolatedAsyncioTestCase):
    """Tests for reset behavior."""

    def setUp(self):
        self.memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

    async def test_reset_returns_counts(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.add_message(ChatMessage.assistant_message("Hi"))
        result = self.memory.reset()
        self.assertEqual(result["messages"], 2)

    async def test_reset_clears_compaction_count(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.compact()
        self.memory.reset()
        self.assertEqual(self.memory._compaction_count, 0)

    async def test_reset_clears_messages(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        self.memory.reset()
        self.assertEqual(len(self.memory.messages), 0)


class TestCompactionMemoryTokenEstimation(unittest.IsolatedAsyncioTestCase):
    """Tests for token estimation."""

    def setUp(self):
        self.memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.memory.set_system_prompt("System")

    def test_estimate_message_tokens_basic(self):
        msg = ChatMessage.user_message("Hello world")
        tokens = self.memory.estimate_message_tokens(msg)
        self.assertGreater(tokens, 0)

    async def test_estimate_working_memory_tokens(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        tokens = self.memory.estimate_working_memory_tokens()
        # Should include system prompt + user message + overhead
        self.assertGreater(tokens, 0)

    async def test_token_estimate_grows_with_messages(self):
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        tokens1 = self.memory.estimate_working_memory_tokens()

        await self.memory.add_message(
            ChatMessage.assistant_message("Hi there! How can I help you today?")
        )
        tokens2 = self.memory.estimate_working_memory_tokens()

        self.assertGreater(tokens2, tokens1)


class TestCompactionMemoryDefaultCompact(unittest.IsolatedAsyncioTestCase):
    """Tests for default compaction (no client provided)."""

    async def test_no_client_no_compact_fn_raises(self):
        memory = CompactionMemory(
            model_name="gpt-4o",
            # No client, no compact_fn
        )
        await memory.add_message(ChatMessage.user_message("Hello"))
        with self.assertRaises(ValueError) as cm:
            await memory.compact()
        self.assertIn("client", str(cm.exception).lower())


class TestCompactionMemorySetModel(unittest.TestCase):
    """Tests for set_model() — agent injecting model/client params."""

    def test_fills_model_name_when_none(self):
        mem = CompactionMemory(compact_fn=lambda m, s: "summary")
        self.assertIsNone(mem._model_name)
        mem.set_model(client=None, model_name="gpt-4o")
        self.assertEqual(mem._model_name, "gpt-4o")

    def test_preserves_user_model_name(self):
        mem = CompactionMemory(
            model_name="my-custom-model",
            compact_fn=lambda m, s: "summary",
        )
        mem.set_model(client=None, model_name="gpt-4o")
        self.assertEqual(mem._model_name, "my-custom-model")

    def test_fills_client(self):
        mem = CompactionMemory(compact_fn=lambda m, s: "summary")
        self.assertIsNone(mem._client)
        sentinel_client = object()
        mem.set_model(client=sentinel_client, model_name="gpt-4o")
        self.assertIs(mem._client, sentinel_client)

    def test_preserves_user_client(self):
        user_client = object()
        mem = CompactionMemory(
            client=user_client,
            compact_fn=lambda m, s: "summary",
        )
        mem.set_model(client=object(), model_name="gpt-4o")
        self.assertIs(mem._client, user_client)

    def test_fills_compact_model_name(self):
        mem = CompactionMemory(compact_fn=lambda m, s: "summary")
        mem.set_model(client=None, model_name="gpt-4o")
        self.assertEqual(mem._compact_model_name, "gpt-4o")

    def test_preserves_user_compact_model_name(self):
        mem = CompactionMemory(
            config=CompactionMemoryConfig(compact_model_name="cheap-model"),
            compact_fn=lambda m, s: "summary",
        )
        mem.set_model(client=None, model_name="gpt-4o")
        self.assertEqual(mem._compact_model_name, "cheap-model")


class TestCompactionMemoryLogging(unittest.IsolatedAsyncioTestCase):
    """Tests for conversation log persistence and search tools."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        from motus.memory.stores.local_conversation_log import (
            LocalConversationLogStore,
        )

        self.log_store = LocalConversationLogStore(self.temp_dir)
        self.memory = CompactionMemory(
            config=CompactionMemoryConfig(session_id="test-session"),
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.memory.set_system_prompt("You are a test assistant.")

    def _read_log_entries(self):
        return self.log_store.read_entries(self.memory.session_id)

    def _read_log_messages(self):
        """Return only 'message' type entries (skip session_meta etc.)."""
        return [e for e in self._read_log_entries() if e.get("type") == "message"]

    async def test_log_file_created_on_first_message(self):
        """Log file should be created when the first message is added."""
        self.assertFalse(self.log_store.exists("test-session"))

        await self.memory.add_message(ChatMessage.user_message("Hello"))
        self.assertTrue(self.log_store.exists("test-session"))

    async def test_messages_written_to_log(self):
        """Messages should be written to the log with full content."""
        await self.memory.add_message(ChatMessage.user_message("Hello world"))
        await self.memory.add_message(
            ChatMessage.assistant_message("Hi there! How can I help?")
        )

        messages = self._read_log_messages()
        self.assertEqual(len(messages), 2)

        self.assertEqual(messages[0]["type"], "message")
        self.assertEqual(messages[0]["message"]["role"], "user")
        self.assertEqual(messages[0]["message"]["content"], "Hello world")
        self.assertIn("ts", messages[0])

        self.assertEqual(messages[1]["message"]["role"], "assistant")
        self.assertEqual(messages[1]["message"]["content"], "Hi there! How can I help?")

    async def test_session_meta_written_first(self):
        """session_meta should be the first entry in the log."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))

        entries = self._read_log_entries()
        self.assertGreater(len(entries), 0)
        self.assertEqual(entries[0]["type"], "session_meta")
        self.assertEqual(entries[0]["session_id"], "test-session")
        self.assertEqual(entries[0]["system_prompt"], "You are a test assistant.")
        self.assertIn("config", entries[0])

    async def test_full_content_not_truncated(self):
        """Both trace and log should contain full content without truncation."""
        long_content = "A" * 500
        await self.memory.add_message(ChatMessage.user_message(long_content))

        # In-memory trace keeps full content
        trace = self.memory.get_memory_trace()
        self.assertEqual(trace["events"][0]["content"], long_content)

        # Log keeps full content
        messages = self._read_log_messages()
        self.assertEqual(messages[0]["message"]["content"], long_content)

    async def test_compaction_summary_written_to_log(self):
        """Compaction summaries should be written to the log."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.add_message(ChatMessage.assistant_message("Hi"))
        await self.memory.compact()

        entries = self._read_log_entries()
        compaction_entries = [e for e in entries if e.get("type") == "compaction"]
        self.assertEqual(len(compaction_entries), 1)

        ce = compaction_entries[0]
        self.assertEqual(ce["compaction_number"], 1)
        # Unit C: full compact — both messages summarized
        self.assertEqual(ce["messages_compacted"], 2)
        self.assertIn("2 messages", ce["summary"])

    async def test_search_tool_finds_matching_messages(self):
        """Search tool should find messages matching the query."""
        await self.memory.add_message(
            ChatMessage.user_message("Fix the bug in auth.py")
        )
        await self.memory.add_message(
            ChatMessage.assistant_message("I'll look at auth.py now")
        )
        await self.memory.add_message(ChatMessage.user_message("Also check main.py"))

        result = json.loads(await self.memory._tool_search_conversation_log("auth.py"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["results"][0]["role"], "user")
        self.assertIn("auth.py", result["results"][0]["content"])

    async def test_search_tool_case_insensitive(self):
        """Search should be case-insensitive."""
        await self.memory.add_message(ChatMessage.user_message("Fix the ERROR"))

        result = json.loads(await self.memory._tool_search_conversation_log("error"))
        self.assertEqual(result["count"], 1)

    async def test_search_tool_max_results(self):
        """Search should respect max_results limit."""
        for i in range(5):
            await self.memory.add_message(
                ChatMessage.user_message(f"Message {i} about topic")
            )

        result = json.loads(
            await self.memory._tool_search_conversation_log("topic", max_results=3)
        )
        self.assertEqual(result["count"], 3)

    async def test_search_tool_no_results(self):
        """Search should return empty results for non-matching query."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))

        result = json.loads(
            await self.memory._tool_search_conversation_log("nonexistent")
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 0)

    async def test_search_tool_empty_log(self):
        """Search on a fresh memory with no messages should return empty results."""
        memory = CompactionMemory(
            config=CompactionMemoryConfig(
                log_base_path=self.temp_dir,
                session_id="empty-search",
            ),
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        result = json.loads(await memory._tool_search_conversation_log("anything"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 0)

    async def test_read_compaction_summary_latest(self):
        """Should read the latest compaction summary with -1."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.compact()
        await self.memory.add_message(ChatMessage.user_message("More"))
        await self.memory.compact()

        result = json.loads(await self.memory._tool_read_compaction_summary(-1))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["compaction_number"], 2)

    async def test_read_compaction_summary_by_number(self):
        """Should read a specific compaction summary by 1-indexed number."""
        await self.memory.add_message(ChatMessage.user_message("First batch"))
        await self.memory.compact()
        await self.memory.add_message(ChatMessage.user_message("Second batch"))
        await self.memory.compact()

        result = json.loads(await self.memory._tool_read_compaction_summary(1))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["compaction_number"], 1)

    async def test_read_compaction_summary_not_found(self):
        """Should return not_found for invalid compaction number."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.compact()

        result = json.loads(await self.memory._tool_read_compaction_summary(99))
        self.assertEqual(result["status"], "not_found")

    async def test_read_compaction_summary_none_available(self):
        """Should return no_summaries when no compactions have occurred."""
        result = json.loads(await self.memory._tool_read_compaction_summary(-1))
        self.assertEqual(result["status"], "no_summaries")

    def test_build_tools_returns_tools_when_logging_enabled(self):
        """build_tools should return tool callables when log store is set."""
        tools = self.memory.build_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 2)

    def test_fork_gets_new_session_id(self):
        """Forked memory should get a new session_id, same log store."""
        clone = self.memory.fork()
        self.assertNotEqual(clone._session_id, self.memory._session_id)
        self.assertIs(clone._log_store, self.memory._log_store)
        self.assertEqual(clone._compaction_summaries, [])

    def test_fork_without_logging(self):
        """Forking a memory without logging should assign new session_id, no log store."""
        memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
            config=CompactionMemoryConfig(log_base_path=None),
        )
        clone = memory.fork()
        self.assertNotEqual(clone.session_id, memory.session_id)
        self.assertIsNone(clone._log_store)

    async def test_reset_clears_summaries_keeps_log(self):
        """Reset should clear in-memory summaries but keep the log."""
        await self.memory.add_message(ChatMessage.user_message("Hello"))
        await self.memory.compact()

        self.assertTrue(self.log_store.exists("test-session"))
        self.assertEqual(len(self.memory._compaction_summaries), 1)

        self.memory.reset()

        self.assertEqual(len(self.memory._compaction_summaries), 0)
        # Log should still exist
        self.assertTrue(self.log_store.exists("test-session"))

    def test_session_id_auto_generated(self):
        """session_id should be auto-generated when not provided."""
        memory = CompactionMemory(
            config=CompactionMemoryConfig(log_base_path=self.temp_dir),
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.assertIsNotNone(memory.session_id)
        self.assertIsNotNone(memory.log_store)

    def test_session_id_from_config(self):
        """session_id should use the config value when provided."""
        self.assertEqual(self.memory.session_id, "test-session")

    def test_log_base_path_creates_store(self):
        """log_base_path in config should create a LocalConversationLogStore."""
        memory = CompactionMemory(
            config=CompactionMemoryConfig(
                log_base_path=self.temp_dir,
                session_id="path-test",
            ),
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.assertIsNotNone(memory.log_store)
        self.assertEqual(memory.session_id, "path-test")


class TestCompactionMemoryRestore(unittest.IsolatedAsyncioTestCase):
    """Tests for restore_from_log() session restoration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        from motus.memory.stores.local_conversation_log import (
            LocalConversationLogStore,
        )

        self.log_store = LocalConversationLogStore(self.temp_dir)

    async def _create_session_with_messages(self, session_id="restore-test"):
        """Helper: create a memory, add messages, return it."""
        memory = CompactionMemory(
            config=CompactionMemoryConfig(session_id=session_id),
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        memory.set_system_prompt("You are a test assistant.")
        await memory.add_message(ChatMessage.user_message("Hello"))
        await memory.add_message(ChatMessage.assistant_message("Hi there!"))
        await memory.add_message(ChatMessage.user_message("Do task X"))
        return memory

    async def test_restore_basic(self):
        """Restored memory should have the same messages."""
        await self._create_session_with_messages()

        restored = CompactionMemory.restore_from_log(
            session_id="restore-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

        self.assertEqual(len(restored.messages), 3)
        self.assertEqual(restored.messages[0].content, "Hello")
        self.assertEqual(restored.messages[1].content, "Hi there!")
        self.assertEqual(restored.messages[2].content, "Do task X")

    async def test_restore_system_prompt_from_meta(self):
        """system_prompt should be restored from session_meta."""
        await self._create_session_with_messages()

        restored = CompactionMemory.restore_from_log(
            session_id="restore-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

        self.assertEqual(
            restored.construct_system_prompt().content,
            "You are a test assistant.",
        )

    async def test_restore_config_from_meta(self):
        """Config values should be restored from session_meta."""
        memory = CompactionMemory(
            config=CompactionMemoryConfig(
                session_id="config-test",
                safety_ratio=0.6,
                token_threshold=5000,
            ),
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        memory.set_system_prompt("Test")
        await memory.add_message(ChatMessage.user_message("Hello"))

        restored = CompactionMemory.restore_from_log(
            session_id="config-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

        self.assertAlmostEqual(restored.config.safety_ratio, 0.6)
        self.assertEqual(restored.config.token_threshold, 5000)

    async def test_restore_replays_compaction(self):
        """After restore with compaction, messages should be continuation only + post-compaction."""
        original = await self._create_session_with_messages()
        await original.compact()
        await original.add_message(ChatMessage.user_message("After compaction"))

        restored = CompactionMemory.restore_from_log(
            session_id="restore-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

        # Should have: continuation message + "After compaction"
        self.assertEqual(len(restored.messages), 2)
        self.assertIn("context_summary", restored.messages[0].content)
        self.assertEqual(restored.messages[1].content, "After compaction")

    async def test_restore_compaction_count(self):
        """Compaction count should be correctly restored."""
        original = await self._create_session_with_messages()
        await original.compact()
        await original.add_message(ChatMessage.user_message("More"))
        await original.compact()

        restored = CompactionMemory.restore_from_log(
            session_id="restore-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

        self.assertEqual(restored._compaction_count, 2)
        self.assertEqual(len(restored._compaction_summaries), 2)

    async def test_restore_preserves_session_id(self):
        """Restored memory should keep the same session_id."""
        await self._create_session_with_messages()

        restored = CompactionMemory.restore_from_log(
            session_id="restore-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

        self.assertEqual(restored.session_id, "restore-test")

    async def test_restore_then_continue(self):
        """Should be able to add messages after restore and they go to the same log."""
        await self._create_session_with_messages()

        restored = CompactionMemory.restore_from_log(
            session_id="restore-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )

        await restored.add_message(ChatMessage.user_message("Continuing"))

        self.assertEqual(len(restored.messages), 4)
        # New message should be in the log too
        entries = self.log_store.read_entries("restore-test")
        message_entries = [e for e in entries if e.get("type") == "message"]
        self.assertEqual(message_entries[-1]["message"]["content"], "Continuing")

    async def test_restore_nonexistent_raises(self):
        """Should raise ValueError for non-existent session."""
        with self.assertRaises(ValueError):
            CompactionMemory.restore_from_log(
                session_id="does-not-exist",
                conversation_log_store=self.log_store,
            )

    async def test_restore_with_explicit_overrides(self):
        """Explicit system_prompt and config should override meta."""
        await self._create_session_with_messages()

        custom_config = CompactionMemoryConfig(safety_ratio=0.5)
        restored = CompactionMemory.restore_from_log(
            session_id="restore-test",
            conversation_log_store=self.log_store,
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
            system_prompt="Overridden prompt",
            config=custom_config,
        )

        self.assertEqual(
            restored.construct_system_prompt().content, "Overridden prompt"
        )
        self.assertAlmostEqual(restored.config.safety_ratio, 0.5)


class TestCompactionMemorySplitCompaction(unittest.IsolatedAsyncioTestCase):
    """Tests for split-compaction: summarize old history, replay last turn."""

    def setUp(self):
        self.memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
        )
        self.memory.set_system_prompt("You are a test assistant.")

    async def test_unit_b_replays_last_turn(self):
        """Unit B boundary: assistant+tool turn is replayed verbatim."""
        tool_call = ToolCall(
            id="c1", function=FunctionCall(name="search", arguments="{}")
        )
        await self.memory.add_message(ChatMessage.user_message("hello"))
        await self.memory.add_message(
            ChatMessage.assistant_message(content=None, tool_calls=[tool_call])
        )
        await self.memory.add_message(
            ChatMessage.tool_message("result", "c1", "search")
        )

        await self.memory.compact()

        # Should have: continuation + assistant + tool_result = 3 messages
        msgs = self.memory.messages
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0].role, "user")  # continuation
        self.assertIn("context_summary", msgs[0].content)
        self.assertEqual(msgs[1].role, "assistant")
        self.assertEqual(msgs[2].role, "tool")

    async def test_unit_a_pending_request_embedded_in_continuation(self):
        """Unit A boundary: pending user message is embedded in continuation."""
        await self.memory.add_message(ChatMessage.user_message("first message"))
        await self.memory.add_message(ChatMessage.assistant_message("response"))
        await self.memory.add_message(ChatMessage.user_message("pending request"))

        await self.memory.compact()

        # Should have exactly 1 message (user message embedded in continuation)
        msgs = self.memory.messages
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, "user")
        self.assertIn("context_summary", msgs[0].content)
        self.assertIn("pending request", msgs[0].content)
        self.assertIn("pending_request", msgs[0].content)

    async def test_unit_c_full_compact_no_replay(self):
        """Unit C boundary: full compaction, no replay — conversation at rest."""
        await self.memory.add_message(ChatMessage.user_message("hello"))
        await self.memory.add_message(ChatMessage.assistant_message("final answer"))

        await self.memory.compact()

        msgs = self.memory.messages
        self.assertEqual(len(msgs), 1)
        self.assertIn("context_summary", msgs[0].content)

    async def test_fallback_full_compact_when_single_message(self):
        """When entire history is one message, fall back to full compaction."""
        await self.memory.add_message(ChatMessage.user_message("only message"))

        await self.memory.compact()

        # Fallback: entire history summarized, 1 continuation only
        msgs = self.memory.messages
        self.assertEqual(len(msgs), 1)
        self.assertIn("context_summary", msgs[0].content)

    async def test_summarized_count_excludes_replayed(self):
        """messages_compacted in callback reflects only summarized messages."""
        callback_calls = []
        memory = CompactionMemory(
            model_name="gpt-4o",
            compact_fn=mock_compact_fn,
            on_compact=lambda s: callback_calls.append(s),
        )
        memory.set_system_prompt("Test")
        # Add [user, assistant(no tools), user] — last unit is second user msg
        await memory.add_message(ChatMessage.user_message("first"))
        await memory.add_message(ChatMessage.assistant_message("resp"))
        await memory.add_message(ChatMessage.user_message("second"))

        await memory.compact()

        stats = callback_calls[0]
        # to_summarize = [user("first"), assistant("resp")] = 2 messages
        self.assertEqual(stats["messages_compacted"], 2)
        self.assertEqual(
            stats["messages_replayed"], 1
        )  # the pending user msg (before embed)


if __name__ == "__main__":
    unittest.main()
