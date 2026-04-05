"""
CompactionMemory - Claude Code-style context compaction.

When context approaches the token threshold, the entire conversation is sent
to an LLM with a structured compaction prompt. The model returns a detailed
"working state" summary that becomes the first user message in the new context.

The agent loop continues seamlessly with:
    [system_prompt, continuation_user_message(summary)]
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from motus.models import BaseChatClient, ChatMessage

from .base_memory import BaseMemory
from .compaction_prompts import COMPACTION_USER_PROMPT, CONTINUATION_TEMPLATE
from .config import CompactionMemoryConfig
from .interfaces import ConversationLogStore
from .model_limits import estimate_compaction_threshold
from .stores.local_conversation_log import LocalConversationLogStore

logger = logging.getLogger("CompactionMemory")

# Type alias for custom compaction functions
CompactFn = Callable[[List[ChatMessage], str], str]
# Signature: (messages, system_prompt) -> summary_text


class CompactionMemory(BaseMemory):
    """
    Memory implementation using full-context compaction.

    Instead of summarizing-and-storing like the original Memory class,
    CompactionMemory:
    1. Monitors token count on each add_message()
    2. When threshold is reached, sends ALL messages + system prompt
       to the LLM with a structured compaction prompt
    3. The LLM returns a detailed "working state" summary
    4. The messages list is replaced with a single continuation user
       message containing the summary
    5. The agent loop continues seamlessly

    See ``apps/deep_research/researcher.py`` for a full working example.

    Usage:
        # Simple — agent injects client, model_name, system_prompt via bind()
        agent = ReActAgent(
            client=my_client,
            model_name="gpt-4o",
            system_prompt="You are a helpful assistant.",
            memory=CompactionMemory(
                config=CompactionMemoryConfig(safety_ratio=0.8),
            ),
        )

        # Advanced — explicit client/model for compaction
        agent = ReActAgent(
            ...,
            memory=CompactionMemory(
                config=CompactionMemoryConfig(
                    compact_model_name="gpt-4o-mini",
                    safety_ratio=0.75,
                ),
                on_compact=my_callback,
            ),
        )
    """

    def __init__(
        self,
        *,
        config: Optional[CompactionMemoryConfig] = None,
        compact_fn: Optional[CompactFn] = None,
        on_compact: Optional[Callable[[Dict[str, Any]], None]] = None,
        conversation_log_store: Optional[ConversationLogStore] = None,
        enable_memory_tools: bool = True,
        # Runtime params — optional, filled by agent.set_model() if not provided
        client: Optional[BaseChatClient] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize CompactionMemory.

        Args:
            config: Optional CompactionMemoryConfig with tuning parameters.
            compact_fn: Optional custom compaction function. If provided,
                        this is called instead of the default LLM-based compaction.
                        Signature: (messages: List[ChatMessage], system_prompt: str) -> str
            on_compact: Optional callback invoked after each compaction completes.
                        Receives a dict with: messages_compacted, summary_tokens,
                        compaction_number, summary.
            conversation_log_store: Optional ConversationLogStore backend for
                        session log persistence. If not provided, resolved from config.
            enable_memory_tools: Whether build_tools() returns memory tools.
            client: Optional BaseChatClient for compaction LLM calls.
                    If not provided, injected by the agent via set_model().
            model_name: Optional model identifier for token estimation.
                        If not provided, injected by the agent via set_model().
        """
        self.config = config or CompactionMemoryConfig()

        super().__init__(
            model_name=model_name,
            max_tool_result_tokens=self.config.max_tool_result_tokens,
            tool_result_truncation_suffix=self.config.tool_result_truncation_suffix,
            enable_memory_tools=enable_memory_tools,
        )

        # CompactionMemory-specific
        self._client = client
        self._compact_model_name = self.config.compact_model_name
        self._compact_fn = compact_fn
        self._token_threshold = self.config.token_threshold
        self._safety_ratio = self.config.safety_ratio
        self._on_compact = on_compact
        self._compaction_count: int = 0

        # Session identity
        self._session_id: str = self.config.session_id or str(uuid.uuid4())

        # Resolve conversation log store (None = logging disabled)
        if conversation_log_store is not None:
            self._log_store: Optional[ConversationLogStore] = conversation_log_store
        elif self.config.log_base_path is not None:
            self._log_store = LocalConversationLogStore(self.config.log_base_path)
        else:
            self._log_store = None

        self._session_meta_written: bool = False
        self._parent_session_id: Optional[str] = None
        self._compaction_summaries: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Model / client injection (called by agent during init)
    # -------------------------------------------------------------------------

    def set_model(self, *, client, model_name) -> None:
        """Fill model-related params not provided at init."""
        if self._model_name is None:
            self._model_name = model_name
        if self._client is None:
            self._client = client
        if self._compact_model_name is None:
            self._compact_model_name = self._model_name

    # -------------------------------------------------------------------------
    # Conversation log persistence
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The session identifier for this memory instance."""
        return self._session_id

    @property
    def log_store(self) -> Optional[ConversationLogStore]:
        """The conversation log store backend, or None if logging is disabled."""
        return self._log_store

    def _add_trace_event(self, message: ChatMessage) -> None:
        """Log message to in-memory trace and append to log store."""
        super()._add_trace_event(message)
        if self._log_store is None:
            return
        self._write_log_entry(
            {
                "type": "message",
                "ts": datetime.now(timezone.utc).isoformat(),
                "message": message.model_dump(exclude_none=True),
            }
        )

    def _write_log_entry(self, entry: dict) -> None:
        """Append a single log entry via the conversation log store."""
        if self._log_store is None:
            return
        if not self._session_meta_written:
            self._write_session_meta()
        self._log_store.append(self._session_id, entry)

    def _write_session_meta(self) -> None:
        """Write session metadata as the first log entry."""
        meta = {
            "type": "session_meta",
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "system_prompt": self._system_prompt or "",
            "config": {
                "safety_ratio": self.config.safety_ratio,
                "token_threshold": self.config.token_threshold,
                "compact_model_name": self.config.compact_model_name,
                "max_tool_result_tokens": self.config.max_tool_result_tokens,
            },
            "parent_session_id": self._parent_session_id,
        }
        self._log_store.append(self._session_id, meta)
        self._session_meta_written = True

    # -------------------------------------------------------------------------
    # Compaction
    # -------------------------------------------------------------------------

    def _is_safe_to_compact(self) -> bool:
        """Check if it's safe to compact right now.

        Compaction is NOT safe if the last message is an assistant message
        with tool_calls — we need to wait for the corresponding tool results
        to be added before compacting, otherwise the continuation would
        produce orphaned tool results.
        """
        if not self._messages:
            return False
        last = self._messages[-1]
        if last.role == "assistant" and last.tool_calls:
            return False
        return True

    async def _auto_compact(self) -> None:
        """Check if context is near the threshold and trigger compaction."""
        if not self._is_safe_to_compact():
            return

        if self._token_threshold is not None:
            token_threshold = self._token_threshold
        else:
            token_threshold = estimate_compaction_threshold(
                model_id=self._model_name,
                safety_ratio=self._safety_ratio,
            )
            if token_threshold is None:
                token_threshold = int(128_000 * self._safety_ratio)
                logger.warning(
                    f"Model '{self._model_name}' not found, "
                    f"using fallback threshold: {token_threshold}"
                )

        estimated_tokens = self.estimate_working_memory_tokens()
        logger.debug(
            f"Auto-compact check: {estimated_tokens} tokens "
            f"(threshold: {token_threshold})"
        )

        if estimated_tokens > token_threshold:
            await self.compact()

    async def compact(self, **kwargs) -> Optional[str]:
        """
        Perform CC-style compaction.

        Sends ALL current messages to the LLM (or custom compact_fn) with
        a structured compaction prompt. The summary replaces the entire
        message history as a single continuation user message.

        Returns:
            The summary text, or None if compaction was not possible.
        """
        if not self._messages:
            return None

        logger.info(
            f"Beginning compaction of {len(self._messages)} messages "
            f"({self.estimate_working_memory_tokens()} estimated tokens)"
        )

        # Generate summary
        if self._compact_fn is not None:
            summary = self._compact_fn(self._messages, self._system_prompt)
        else:
            summary = await self._default_compact(self._messages, self._system_prompt)

        # Build continuation message
        continuation_content = CONTINUATION_TEMPLATE.format(summary=summary)

        # Replace all messages with the continuation message
        old_message_count = len(self._messages)
        self._messages = [ChatMessage.user_message(continuation_content)]

        self._compaction_count += 1

        summary_tokens = self.estimate_message_tokens(self._messages[0])

        # Log compaction event (in-memory trace)
        compaction_event = {
            "type": "compaction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages_compacted": old_message_count,
            "summary_tokens": summary_tokens,
            "compaction_number": self._compaction_count,
        }
        self._trace_log.append(compaction_event)

        # Persist compaction summary to disk log and in-memory list
        summary_record = {
            "compaction_number": self._compaction_count,
            "messages_compacted": old_message_count,
            "summary": summary,
        }
        self._compaction_summaries.append(summary_record)
        self._write_log_entry(
            {
                "type": "compaction",
                "ts": compaction_event["timestamp"],
                "compaction_number": self._compaction_count,
                "messages_compacted": old_message_count,
                "summary": summary,
            }
        )

        logger.info(
            f"Compaction complete: {old_message_count} messages -> 1 continuation "
            f"message ({summary_tokens} tokens)"
        )

        # Notify via callback
        if self._on_compact is not None:
            self._on_compact(
                {
                    "messages_compacted": old_message_count,
                    "summary_tokens": summary_tokens,
                    "compaction_number": self._compaction_count,
                    "summary": summary,
                }
            )

        return summary

    async def _default_compact(
        self, messages: List[ChatMessage], system_prompt: str
    ) -> str:
        """Default compaction using the agent's BaseChatClient.

        Keeps the original system prompt and messages intact, then appends
        a final user message asking the model to summarize the conversation.
        This is the most natural approach — the model sees the exact conversation
        as it happened and then gets an explicit instruction to summarize.
        """
        if self._client is None:
            raise ValueError(
                "CompactionMemory requires either a 'client' or a 'compact_fn'. "
                "Neither was provided."
            )

        # Build the compaction request:
        # [system_prompt, *original_messages, user(COMPACTION_USER_PROMPT)]
        compaction_messages = [
            ChatMessage.system_message(system_prompt),
            *messages,
            ChatMessage.user_message(COMPACTION_USER_PROMPT),
        ]

        completion = await self._client.create(
            model=self._compact_model_name,
            messages=compaction_messages,
        )

        return completion.content or "Unable to generate summary."

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def reset(self) -> Dict[str, int]:
        """Reset memory to initial state.

        Clears in-memory state (messages, compaction count, trace log,
        compaction summaries). The log file on disk is NOT deleted —
        it's an append-only historical record.
        """
        count = len(self._messages)
        self._messages.clear()
        self._compaction_count = 0
        self._trace_log.clear()
        self._compaction_summaries.clear()
        return {"messages": count, "short_term": 0, "long_term": 0}

    # -------------------------------------------------------------------------
    # Fork
    # -------------------------------------------------------------------------

    def fork(self) -> "CompactionMemory":
        """Create an independent copy with a new session log.

        The forked memory shares the same log store backend but gets a
        new session_id. It does not inherit the parent's compaction summaries.
        """
        clone = super().fork()
        clone._parent_session_id = self._session_id
        clone._session_id = str(uuid.uuid4())
        clone._session_meta_written = False
        clone._compaction_summaries = []
        return clone

    # -------------------------------------------------------------------------
    # Memory tools
    # -------------------------------------------------------------------------

    def build_tools(self) -> list:
        """Provide conversation log tools when logging is enabled."""
        if not self._enable_memory_tools or self._log_store is None:
            return []
        return [self._tool_search_conversation_log, self._tool_read_compaction_summary]

    async def _tool_search_conversation_log(
        self, query: str, max_results: int = 10
    ) -> str:
        """Search the conversation log for messages matching a query.

        Performs case-insensitive substring matching on message content.
        Only use this tool when the current conversation is a continuation
        of a previous session (i.e., context compaction has occurred and
        earlier messages are no longer in the active context). This lets
        you recover specific details — error messages, file paths, code
        snippets — that the compaction summary may have omitted.

        Args:
            query: The search term to look for in message content.
            max_results: Maximum number of matching messages to return.
        """
        results = self._log_store.search_messages(self._session_id, query, max_results)
        return json.dumps(
            {"status": "ok", "query": query, "count": len(results), "results": results}
        )

    async def _tool_read_compaction_summary(self, compaction_number: int = -1) -> str:
        """Read a compaction summary by number.

        Only use this tool when context compaction has occurred and you
        need to review the full summary from a previous compaction event.
        Use -1 for the most recent summary, 1 for the first, etc.

        Args:
            compaction_number: Which compaction summary to read.
                              -1 = latest, 1 = first, 2 = second, etc.
        """
        if not self._compaction_summaries:
            return json.dumps(
                {
                    "status": "no_summaries",
                    "message": "No compactions have occurred yet.",
                }
            )

        if compaction_number == -1:
            record = self._compaction_summaries[-1]
        else:
            idx = compaction_number - 1
            if idx < 0 or idx >= len(self._compaction_summaries):
                return json.dumps(
                    {
                        "status": "not_found",
                        "message": f"Compaction #{compaction_number} not found. "
                        f"Available: 1-{len(self._compaction_summaries)}.",
                    }
                )
            record = self._compaction_summaries[idx]

        return json.dumps(
            {
                "status": "ok",
                "compaction_number": record["compaction_number"],
                "messages_compacted": record["messages_compacted"],
                "summary": record["summary"],
            }
        )

    # -------------------------------------------------------------------------
    # Session restore
    # -------------------------------------------------------------------------

    @classmethod
    def restore_from_log(
        cls,
        session_id: str,
        *,
        conversation_log_store: Optional[ConversationLogStore] = None,
        log_base_path: Optional[str] = None,
        client: Optional[BaseChatClient] = None,
        model_name: Optional[str] = None,
        compact_fn: Optional[CompactFn] = None,
        on_compact: Optional[Callable[[Dict[str, Any]], None]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[CompactionMemoryConfig] = None,
    ) -> "CompactionMemory":
        """Restore a CompactionMemory instance from a saved session log.

        Reads all entries from the log store and replays them to rebuild
        the in-memory state (messages, compaction count, summaries).
        The restored instance continues appending to the same session log.

        Args:
            session_id: The session to restore.
            conversation_log_store: The log store backend containing the session.
                If not provided, a LocalConversationLogStore is created from
                log_base_path (or config.log_base_path).
            log_base_path: Directory path for LocalConversationLogStore. Used only when
                conversation_log_store is not provided. None uses LocalConversationLogStore default.
            client: Optional BaseChatClient for future compaction calls.
            model_name: Optional model name for token estimation.
            compact_fn: Optional custom compaction function.
            on_compact: Optional compaction callback.
            system_prompt: Override system prompt. If None, read from session_meta.
            config: Override config. If None, read from session_meta.

        Returns:
            A fully reconstructed CompactionMemory ready to continue the session.

        Raises:
            ValueError: If session_id not found in the log store.
        """
        if conversation_log_store is None:
            resolved_path = log_base_path or (config.log_base_path if config else None)
            conversation_log_store = (
                LocalConversationLogStore(resolved_path)
                if resolved_path
                else LocalConversationLogStore()
            )
        entries = conversation_log_store.read_entries(session_id)
        if not entries:
            raise ValueError(f"No log entries found for session '{session_id}'")

        # Extract session_meta if present
        meta = None
        for entry in entries:
            if entry.get("type") == "session_meta":
                meta = entry
                break

        if meta is None:
            logger.warning(
                f"No session_meta found for session '{session_id}'. "
                "Using default config and empty system prompt."
            )

        # Resolve config from meta or fallback
        if config is None and meta and "config" in meta:
            meta_config = meta["config"]
            config = CompactionMemoryConfig(
                safety_ratio=meta_config.get("safety_ratio", 0.75),
                token_threshold=meta_config.get("token_threshold"),
                compact_model_name=meta_config.get("compact_model_name"),
                max_tool_result_tokens=meta_config.get("max_tool_result_tokens", 50000),
            )
        config = config or CompactionMemoryConfig()

        # Resolve system_prompt from meta or fallback
        resolved_system_prompt = system_prompt
        if resolved_system_prompt is None and meta:
            resolved_system_prompt = meta.get("system_prompt", "")
        resolved_system_prompt = resolved_system_prompt or ""

        # Create instance — pass the same store so future writes go to same log
        memory = cls(
            config=config,
            compact_fn=compact_fn,
            on_compact=on_compact,
            conversation_log_store=conversation_log_store,
            client=client,
            model_name=model_name,
        )
        memory._session_id = session_id
        memory._session_meta_written = True  # don't re-write meta
        memory.set_system_prompt(resolved_system_prompt)

        # Replay entries
        for entry in entries:
            entry_type = entry.get("type")
            if entry_type == "message":
                msg = ChatMessage(**entry["message"])
                memory._messages.append(msg)
            elif entry_type == "compaction":
                # Compaction replaces all accumulated messages with continuation
                summary = entry.get("summary", "")
                continuation = CONTINUATION_TEMPLATE.format(summary=summary)
                memory._messages = [ChatMessage.user_message(continuation)]
                memory._compaction_count += 1
                memory._compaction_summaries.append(
                    {
                        "compaction_number": entry.get(
                            "compaction_number", memory._compaction_count
                        ),
                        "messages_compacted": entry.get("messages_compacted", 0),
                        "summary": summary,
                    }
                )

        return memory
