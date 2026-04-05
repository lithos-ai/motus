"""
DatabaseMemory - Database-backed memory for agent applications.

Provides a unified interface for:
- Working memory: Current conversation messages
- Short-term memory: File-based ephemeral storage
- Long-term memory: Vector database persistent storage
"""

import copy
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .summarization import SummarizationResult

from motus.models import ChatMessage

from .base_memory import BaseMemory
from .config import DatabaseMemoryConfig
from .embedding import get_embedding_fn
from .interfaces import MemoryStore, VectorStore
from .model_limits import estimate_compaction_threshold
from .models import (
    MemoryEntry,
    MemoryPriority,
    MemoryScope,
    MemoryType,
    PromotableEntry,
)
from .prompt_templates import (
    LONG_TERM_MEMORY_PROMPT,
    format_memory_prompt,
    format_short_term_memory_prompt,
)
from .stores import FileSystemStore, InMemoryVectorStore
from .summarization import get_summarization_fn

logger = logging.getLogger("MemoryHelper")


class DatabaseMemory(BaseMemory):
    """
    Main memory interface for agent applications.

    See ``apps/motusbot/superagent.py`` for a full working example.

    Provides a unified interface for:
    - Working memory: Current conversation messages
    - Short-term memory: File-based ephemeral storage
    - Long-term memory: Vector database persistent storage

    Usage:
        memory = Memory(
            scope=MemoryScope(user_id="user123", session_id="sess456"),
            config=DatabaseMemoryConfig(),
        )

        # Add messages to working memory
        memory.add_message(ChatMessage.user_message("Hello"))

        # Store in short-term memory
        memory.remember_short_term("user_name", "Alice")

        # Store in long-term memory
        memory.remember_long_term(
            MemoryEntry(content="User prefers dark mode", memory_type=MemoryType.PREFERENCE)
        )

        # Search long-term memory
        results = memory.recall("What are the user preferences?")
    """

    def __init__(
        self,
        scope: MemoryScope,
        model_name: str,
        config: Optional[DatabaseMemoryConfig] = None,
        short_term_store: Optional[MemoryStore] = None,
        long_term_store: Optional[VectorStore] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        summarize_fn: Optional[
            Callable[[List[ChatMessage]], "SummarizationResult"]
        ] = None,
        system_prompt: Optional[str] = "",
        enable_memory_tools: bool = True,
    ):
        self.scope = scope
        self.config = config or DatabaseMemoryConfig()

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or "",
            max_tool_result_tokens=self.config.max_tool_result_tokens,
            tool_result_truncation_suffix=self.config.tool_result_truncation_suffix,
            enable_memory_tools=enable_memory_tools,
        )

        # Embedding function for text->vector conversion
        self._embed_fn = embed_fn or get_embedding_fn(
            model_name=config.embedding_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Summarization function for compacting messages
        self._summarize_fn = summarize_fn or get_summarization_fn(
            model_name=config.summarization_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Short-term memory store
        self._short_term = short_term_store or FileSystemStore(
            self.config.short_term_base_path
        )

        # Long-term memory store
        self._long_term = long_term_store or InMemoryVectorStore(
            embed_fn=self._embed_fn
        )

        # Promotion thread state
        self._promotion_thread: Optional[threading.Thread] = None
        self._promotion_running = False

    # -------------------------------------------------------------------------
    # Working Memory — overrides
    # -------------------------------------------------------------------------

    def construct_system_prompt(self) -> ChatMessage:
        """Override: append memory context to system prompt."""
        memory_prompt = self.get_memory_prompt()
        system_content = self._system_prompt + "\n\n" + memory_prompt
        return ChatMessage.system_message(system_content)

    def reset(self) -> Dict[str, int]:
        """
        Reset all memory stores (working, short-term, and long-term).

        This clears:
        - Working memory: All current conversation messages
        - Short-term memory: All ephemeral storage entries
        - Long-term memory: All persistent vector storage entries

        Returns:
            Dictionary with counts of cleared items:
            - 'messages': Number of working memory messages cleared
            - 'short_term': Number of short-term entries cleared
            - 'long_term': Number of long-term entries cleared
        """
        result = {
            "messages": len(self._messages),
            "short_term": 0,
            "long_term": 0,
        }

        # Clear working memory
        self._messages.clear()

        # Clear short-term memory
        result["short_term"] = self._short_term.clear(self.scope)

        # Clear long-term memory
        result["long_term"] = self._long_term.clear(self.scope)

        logger.info(
            f"Memory reset: cleared {result['messages']} messages, "
            f"{result['short_term']} short-term entries, "
            f"{result['long_term']} long-term entries"
        )

        return result

    def fork(self) -> "DatabaseMemory":
        """Create an independent copy with a new session scope.

        Creates a new Memory instance with a fresh session_id so
        the fork has isolated short-term and long-term storage.
        Messages are deep-copied into the new instance.
        """
        new_scope = MemoryScope(
            user_id=self.scope.user_id,
            session_id=str(uuid.uuid4()),
            org_id=self.scope.org_id,
            agent_id=self.scope.agent_id,
            namespace=self.scope.namespace,
        )
        new_memory = DatabaseMemory(
            scope=new_scope,
            model_name=self._model_name,
            config=self.config,
            embed_fn=self._embed_fn,
            summarize_fn=self._summarize_fn,
            system_prompt=self._system_prompt or "",
        )
        for msg in self._messages:
            new_memory._messages.append(copy.deepcopy(msg))
        return new_memory

    # -------------------------------------------------------------------------
    # Short-Term Memory (File-based)
    # -------------------------------------------------------------------------

    def remember_short_term(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Store a value in short-term memory.

        Args:
            key: Unique key for the value
            value: Value to store (must be JSON-serializable)
            ttl_seconds: Time-to-live in seconds (uses default if not specified)
            description: Optional short description of the stored value
        """
        data = {
            "value": value,
            "ttl_seconds": ttl_seconds or self.config.short_term_default_ttl,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
        }
        self._short_term.store(key, data, self.scope)

    def recall_short_term(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from short-term memory.

        Args:
            key: Key to look up

        Returns:
            The stored value, or None if not found or expired
        """
        data = self._short_term.retrieve(key, self.scope)
        if data is None:
            return None

        # Handle promotable entries (stored via remember_short_term_promotable)
        if data.get("_promotable"):
            promotable = PromotableEntry(**data)
            # Return the entry as a dict for JSON serialization
            return {
                "content": promotable.entry.content,
                "memory_type": promotable.entry.memory_type.value,
                "tags": promotable.entry.tags,
                "metadata": promotable.entry.metadata,
            }

        # Check expiration for regular entries
        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
                ttl = data.get("ttl_seconds", self.config.short_term_default_ttl)
                if (
                    ttl
                    and (datetime.now(timezone.utc) - created_at).total_seconds() > ttl
                ):
                    self._short_term.delete(key, self.scope)
                    return None
            except ValueError:
                # Invalid timestamp format - skip expiration check
                pass

        return data.get("value")

    def forget_short_term(self, key: str) -> bool:
        """Remove a value from short-term memory."""
        return self._short_term.delete(key, self.scope)

    def list_short_term_keys(self, prefix: str = "") -> List[str]:
        """List all keys in short-term memory."""
        return self._short_term.list_keys(self.scope, prefix)

    def list_short_term_entries(
        self, prefix: str = ""
    ) -> List[Dict[str, Optional[str]]]:
        """
        List all entries in short-term memory with their descriptions.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of dicts with 'key', 'description', and 'promotable' for each entry.
            For promotable entries, description is extracted from the MemoryEntry metadata.
            For regular entries, description is from the stored data.
        """
        all_keys = self._short_term.list_keys(self.scope, prefix)
        entries = []

        for key in all_keys:
            data = self._short_term.retrieve(key, self.scope)
            if data is None:
                continue

            is_promotable = data.get("_promotable", False)
            description = None
            if is_promotable:
                # For promotable entries, get description from MemoryEntry metadata
                promotable = PromotableEntry(**data)
                description = promotable.entry.metadata.get("description")
            else:
                # For regular entries, get description from stored data
                description = data.get("description")

            entries.append(
                {
                    "key": key,
                    "description": description,
                    "promotable": is_promotable,
                }
            )

        return entries

    # -------------------------------------------------------------------------
    # Memory Promotion (Short-Term -> Long-Term)
    # -------------------------------------------------------------------------

    def remember_short_term_promotable(
        self,
        key: str,
        entry: MemoryEntry,
    ) -> None:
        """
        Store a memory entry in short-term memory with promotion tracking.

        The entry will be eligible for promotion to long-term memory based on:
        - Access count (promoted after N accesses)
        - Time (promoted after N seconds)
        - Priority (HIGH/CRITICAL promoted immediately)

        Args:
            key: Unique key for the entry
            entry: The memory entry to store
        """
        promotable = PromotableEntry(entry=entry)

        # Check for immediate promotion (HIGH/CRITICAL priority)
        if entry.priority in (MemoryPriority.HIGH, MemoryPriority.CRITICAL):
            self.remember_long_term(entry)
            logger.debug(f"Immediately promoted HIGH/CRITICAL entry: {key}")
            return

        # Store in short-term with _promotable metadata flag
        data = promotable.model_dump(mode="json")
        data["_promotable"] = True
        self._short_term.store(key, data, self.scope)
        logger.debug(f"Stored promotable entry in short-term: {key}")

    def access_promotable(self, key: str) -> Optional[MemoryEntry]:
        """
        Access a promotable entry and update its access tracking.

        This may trigger promotion if access threshold is reached.

        Args:
            key: Key of the promotable entry

        Returns:
            The memory entry if found, None otherwise
        """
        data = self._short_term.retrieve(key, self.scope)
        if data is None or not data.get("_promotable"):
            return None

        promotable = PromotableEntry(**data)
        promotable.touch()

        # Check if access threshold reached
        if promotable.access_count >= self.config.promotion_access_threshold:
            self._promote_entry(key, promotable)
            return promotable.entry

        # Update in short-term with new access count
        updated_data = promotable.model_dump(mode="json")
        updated_data["_promotable"] = True
        self._short_term.store(key, updated_data, self.scope)
        return promotable.entry

    def _check_promotion_eligibility(self, promotable: PromotableEntry) -> bool:
        """
        Check if a promotable entry should be promoted to long-term memory.

        Args:
            promotable: The promotable entry wrapper

        Returns:
            True if the entry should be promoted
        """
        # Note: HIGH/CRITICAL priority entries are immediately promoted in
        # remember_short_term_promotable() and never stored as promotable
        return (
            promotable.access_count >= self.config.promotion_access_threshold
            or promotable.age_seconds() >= self.config.promotion_time_threshold_seconds
        )

    def _promote_entry(
        self,
        key: str,
        promotable: PromotableEntry,
    ) -> Optional[str]:
        """
        Promote an entry from short-term to long-term memory.

        Args:
            key: The key of the promotable entry
            promotable: The promotable entry wrapper

        Returns:
            ID of the entry in long-term memory, or None on failure
        """
        # Store in long-term memory
        entry = promotable.entry
        entry.metadata["promoted_from_short_term"] = True
        entry.metadata["promotion_access_count"] = promotable.access_count
        entry.metadata["promotion_age_seconds"] = promotable.age_seconds()

        long_term_id = self.remember_long_term(entry)

        # Delete from short-term memory
        self._short_term.delete(key, self.scope)

        logger.info(
            f"Promoted entry to long-term memory: {key} -> {long_term_id} "
            f"(accesses={promotable.access_count}, age={promotable.age_seconds():.1f}s)"
        )
        return long_term_id

    def promote_entry(self, key: str) -> Optional[str]:
        """
        Manually promote an entry from short-term to long-term memory.

        Args:
            key: Key of the promotable entry

        Returns:
            ID of the entry in long-term memory, or None if not found
        """
        data = self._short_term.retrieve(key, self.scope)
        if data is None or not data.get("_promotable"):
            return None

        promotable = PromotableEntry(**data)
        return self._promote_entry(key, promotable)

    def _scan_and_promote(self) -> int:
        """
        Scan short-term memory for eligible entries and promote them.

        Returns:
            Number of entries promoted
        """
        keys = self._short_term.list_keys(self.scope)
        promoted = 0

        for key in keys:
            data = self._short_term.retrieve(key, self.scope)
            if data is None or not data.get("_promotable"):
                continue

            promotable = PromotableEntry(**data)
            if self._check_promotion_eligibility(promotable):
                self._promote_entry(key, promotable)
                promoted += 1

        if promoted > 0:
            logger.info(f"Promotion scan completed: {promoted} entries promoted")

        return promoted

    def _promotion_loop(self) -> None:
        """Background thread that periodically scans and promotes eligible entries."""
        logger.info("Memory promotion thread started")
        while self._promotion_running:
            try:
                time.sleep(self.config.promotion_check_interval_seconds)
                if self._promotion_running:
                    self._scan_and_promote()
            except Exception as e:
                logger.error(f"Error in promotion loop: {e}")
        logger.info("Memory promotion thread stopped")

    def start_promotion_task(self) -> None:
        """Start the background promotion thread."""
        if self._promotion_running:
            logger.warning("Promotion thread already running")
            return

        if not self.config.auto_promote:
            logger.debug("Auto-promotion disabled, not starting thread")
            return

        self._promotion_running = True
        self._promotion_thread = threading.Thread(
            target=self._promotion_loop,
            daemon=True,
            name="MemoryPromotionThread",
        )
        self._promotion_thread.start()

    def stop_promotion_task(self) -> None:
        """Stop the background promotion thread."""
        self._promotion_running = False
        if self._promotion_thread is not None:
            self._promotion_thread.join(timeout=5.0)
            self._promotion_thread = None
        logger.debug("Promotion thread stopped")

    # -------------------------------------------------------------------------
    # Long-Term Memory (Vector-based)
    # -------------------------------------------------------------------------

    def remember_long_term(
        self,
        entry: Union[MemoryEntry, str],
        memory_type: MemoryType = MemoryType.FACT,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a memory in long-term storage.

        Args:
            entry: MemoryEntry object or content string
            memory_type: Type of memory (if entry is a string)
            tags: Tags for categorization (if entry is a string)
            metadata: Additional metadata (if entry is a string)

        Returns:
            ID of the stored memory
        """
        if isinstance(entry, str):
            entry = MemoryEntry(
                content=entry,
                memory_type=memory_type,
                tags=tags or [],
                metadata=metadata or {},
            )

        ids = self._long_term.upsert([entry], self.scope)
        return ids[0] if ids else ""

    def remember_long_term_batch(
        self,
        entries: List[MemoryEntry],
    ) -> List[str]:
        """Store multiple memories in long-term storage."""
        return self._long_term.upsert(entries, self.scope)

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """
        Search long-term memory for relevant entries.

        Args:
            query: Search query text
            top_k: Maximum number of results
            memory_types: Filter by memory types
            tags: Filter by tags
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant memory entries
        """
        filters = {}
        if memory_types:
            filters["memory_type"] = [t.value for t in memory_types]
        if tags:
            filters["tags"] = tags

        return self._long_term.search_by_text(
            query,
            self.scope,
            top_k=top_k or self.config.long_term_max_results,
            filters=filters if filters else None,
        )

    def recall_by_ids(self, ids: List[str]) -> List[MemoryEntry]:
        """Retrieve specific memories by their IDs."""
        return self._long_term.get_by_ids(ids, self.scope)

    def forget_long_term(self, ids: List[str]) -> int:
        """Remove memories from long-term storage."""
        return self._long_term.delete(ids, self.scope)

    # -------------------------------------------------------------------------
    # Memory Management & Compaction
    # -------------------------------------------------------------------------

    def _find_safe_truncation_point(
        self, messages: List[ChatMessage], target_index: int
    ) -> int:
        """
        Find the nearest safe truncation point that doesn't split tool call groups.

        Tool messages must always be paired with their corresponding assistant message
        that contains the tool_calls. This method scans backward from target_index to
        find a point where truncation won't break this invariant.

        A safe truncation point is before:
        - A user message
        - An assistant message without tool_calls
        - An assistant message with tool_calls (keeping the entire group intact)

        Args:
            messages: List of messages to search within
            target_index: The desired truncation index to start searching from

        Returns:
            The safe index to truncate at (messages[index:] will be kept)
        """
        if target_index <= 0:
            return 0
        if target_index >= len(messages):
            return len(messages)

        # Scan backward from target to find a safe truncation point
        # Tool messages must stay paired with their initiating assistant message,
        # so only non-tool messages (user, system, assistant) are safe boundaries
        for i in range(target_index, -1, -1):
            if messages[i].role != "tool":
                return i

        # If we reach here, couldn't find a safe point - return 0 to keep everything
        return 0

    def _find_token_budget_truncation_point(
        self, messages: List[ChatMessage], token_budget: int
    ) -> int:
        """
        Find the cut index that preserves messages within the given token budget.

        Scans backward from the end of the message list, accumulating token counts
        until the budget is exceeded. Returns the index where messages[index:] will
        be kept (i.e., the preserved messages).

        Args:
            messages: List of messages to search within
            token_budget: Maximum total tokens to preserve

        Returns:
            The index to cut at - messages[index:] will be kept, messages[:index]
            will be summarized. Returns 0 if all messages fit within budget.
        """
        if not messages:
            return 0

        cumulative_tokens = 0

        # Scan backward from most recent, accumulating tokens
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = self.estimate_message_tokens(messages[i])
            if cumulative_tokens + msg_tokens > token_budget:
                # This message would exceed budget, cut here (keep i+1 onwards)
                return i + 1
            cumulative_tokens += msg_tokens

        # All messages fit within budget
        return 0

    async def _auto_compact(self) -> None:
        """Check if auto-compaction should run based on estimated token count."""
        if not self.config.auto_compact:
            return

        token_threshold = estimate_compaction_threshold(
            model_id=self._model_name,
            safety_ratio=self.config.compact_safety_ratio,
        )

        # Fallback threshold if model not found (assume 128k context)
        if token_threshold is None:
            token_threshold = int(128_000 * self.config.compact_safety_ratio)
            logger.warning(
                f"Model '{self._model_name}' not found, using fallback threshold: "
                f"{token_threshold}"
            )

        estimated_tokens = self.estimate_working_memory_tokens()

        logger.debug(
            f"Auto-compact check: {estimated_tokens} tokens "
            f"(threshold: {token_threshold})"
        )

        if estimated_tokens > token_threshold:
            await self.compact()

    async def compact(
        self,
        preserve_ratio: Optional[float] = None,
        preserve_recent: Optional[int] = None,
    ) -> Optional["SummarizationResult"]:
        """
        Compact working memory by summarizing older messages.

        Uses a token-based ratio to determine how many recent messages to preserve,
        ensuring predictable context budget management regardless of message sizes.

        Args:
            preserve_ratio: Fraction of max context to preserve for recent messages
                           (e.g., 0.1 = 10%). Uses config value if not specified.
            preserve_recent: Number of recent messages to preserve regardless of token count.
        Returns:
            SummarizationResult if compaction occurred, None otherwise
        """
        if not self._messages:
            return None

        if preserve_recent is not None:
            # Preserve a fixed number of recent messages
            target_cut_index = max(0, len(self._messages) - preserve_recent)

            logger.debug(
                f"Compacting with preserve_recent={preserve_recent}, "
                f"target_cut_index={target_cut_index}"
            )
        else:
            # Use preserve_ratio (explicit or from config)
            ratio = (
                preserve_ratio
                if preserve_ratio is not None
                else self.config.compact_preserve_ratio
            )

            # Calculate token budget for preserved messages
            from .model_limits import get_model_limits

            limits = get_model_limits(self._model_name)
            if limits is None:
                # Fallback to a reasonable default if model not found
                logger.warning(
                    f"Model '{self._model_name}' not found in limits registry, "
                    "using 128k context fallback"
                )
                max_context = 128_000
            else:
                max_context = limits.context_window

            preserve_budget = int(max_context * ratio)

            logger.debug(
                f"Compacting with preserve_ratio={ratio:.1%}, "
                f"budget={preserve_budget} tokens (max_context={max_context})"
            )

            # Find cut point based on token budget (scanning from end)
            target_cut_index = self._find_token_budget_truncation_point(
                self._messages, preserve_budget
            )

        # Apply safe truncation to respect tool call pairing
        # This may move the cut point backward (keeping more messages)
        safe_cut_index = self._find_safe_truncation_point(
            self._messages, target_cut_index
        )

        logger.debug(
            f"Cut indices: target={target_cut_index}, safe={safe_cut_index} "
            f"(total messages: {len(self._messages)})"
        )

        # If safe cut point is 0, nothing to summarize
        if safe_cut_index == 0:
            logger.debug("No messages to compact (all fit within preserve budget)")
            return None

        messages_to_summarize = self._messages[:safe_cut_index]
        messages_to_keep = self._messages[safe_cut_index:]

        logger.info(
            f"Compacting {len(messages_to_summarize)} messages, "
            f"preserving {len(messages_to_keep)} recent messages"
        )

        result = None
        if self._summarize_fn:
            result = self._summarize_fn(messages_to_summarize)

            # Store summary in short-term memory first (staged promotion)
            # so it appears in get_memory_prompt()
            # It will be promoted to long-term memory based on:
            # - Access frequency (if accessed 3+ times)
            # - Time (after 1 hour by default)
            # - Or manually via promote_entry()
            # Key format: {key_name}_{time}
            time_str = datetime.now(timezone.utc).strftime("%H%M%S")
            summary_key = f"{result.key_name}_{time_str}"

            # Ensure description is in the entry metadata
            result.entry.metadata["description"] = result.description
            result.entry.metadata["message_count"] = len(messages_to_summarize)
            result.entry.metadata["compacted_at"] = datetime.now(
                timezone.utc
            ).isoformat()

            logger.info(result.entry)

            self.remember_short_term_promotable(
                key=summary_key,
                entry=result.entry,
            )
        self._messages = messages_to_keep

        # Log compaction event for trace
        self._trace_log.append(
            {
                "type": "compaction",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "messages_compacted": len(messages_to_summarize),
                "messages_preserved": len(messages_to_keep),
                "summary_created": result is not None,
            }
        )

        return result

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Update an existing long-term memory entry.

        Args:
            memory_id: ID of the memory to update
            content: New content (if changing)
            metadata: Metadata to merge
            tags: Tags to set

        Returns:
            True if update was successful
        """
        entries = self._long_term.get_by_ids([memory_id], self.scope)
        if not entries:
            return False

        entry = entries[0]
        if content is not None:
            entry.content = content
            entry.embedding = None  # Clear embedding so it gets regenerated
        if metadata is not None:
            entry.metadata.update(metadata)
        if tags is not None:
            entry.tags = tags
        entry.updated_at = datetime.now(timezone.utc)

        self._long_term.upsert([entry], self.scope)
        return True

    # -------------------------------------------------------------------------
    # Memory tools
    # -------------------------------------------------------------------------

    def build_tools(self) -> list:
        """Provide short-term and long-term memory search tools."""
        if not self._enable_memory_tools:
            return []
        return [self._tool_search_short_term, self._tool_search_long_term]

    async def _tool_search_short_term(self, key: str) -> str:
        """Retrieve a value from short-term memory by exact key.

        Available keys are listed in the system prompt with descriptions.

        Args:
            key: Exact key to retrieve.

        Returns:
            JSON string with the value or error message.
        """
        import json

        value = self.recall_short_term(key)
        if value is not None:
            return json.dumps(
                {
                    "status": "success",
                    "key": key,
                    "value": value,
                    "note": "If this value doesn't address your needs, use other tools to complete the task.",
                }
            )
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No entry found for key: '{key}'",
                "guidance": "This key doesn't exist in short-term memory. Proceed with your other available tools to complete the task—do not retry memory search.",
            }
        )

    async def _tool_search_long_term(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> str:
        """Search long-term memory using semantic similarity.

        Args:
            query: Descriptive search query for semantic similarity matching.
            memory_types: Filter by types: fact, episode, semantic, procedural, preference.
            tags: Filter by tags.
            top_k: Max results (default: 5, max: 20).

        Returns:
            JSON string with search results.
        """
        import json

        top_k = max(1, min(top_k, 20))

        parsed_memory_types = None
        if memory_types:
            try:
                parsed_memory_types = [MemoryType(t) for t in memory_types]
            except ValueError as e:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid memory_type: {e}. Valid types: fact, episode, semantic, procedural, preference.",
                    }
                )

        try:
            results = self.recall(
                query=query,
                top_k=top_k,
                memory_types=parsed_memory_types,
                tags=tags,
            )
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Error searching long-term memory: {str(e)}",
                }
            )

        if results is None:
            results = []

        if not results:
            return json.dumps(
                {
                    "status": "no_results",
                    "message": f"No memories found matching query: '{query}'",
                    "guidance": "No relevant memories exist for this query. Do not retry—proceed with your other available tools to complete the task.",
                }
            )

        formatted = []
        for entry in results:
            formatted.append(
                {
                    "id": entry.id,
                    "content": entry.content,
                    "type": entry.memory_type.value,
                    "tags": entry.tags,
                    "created_at": entry.created_at.isoformat(),
                }
            )

        return json.dumps(
            {
                "status": "success",
                "count": len(formatted),
                "memories": formatted,
                "note": "If these results don't address your needs, use other tools to complete the task.",
            }
        )

    # -------------------------------------------------------------------------
    # System Prompt Generation
    # -------------------------------------------------------------------------

    def get_short_term_memory_prompt(self) -> str:
        """
        Generate a system prompt describing short-term memory capabilities.

        This prompt is dynamic - it includes the current keys and descriptions
        available in short-term memory so the agent knows what context is available.

        Returns:
            System prompt string for short-term memory
        """
        all_entries = self.list_short_term_entries()
        promotable_entries = []
        regular_entries = []
        for e in all_entries:
            entry = {"key": e["key"], "description": e["description"]}
            (promotable_entries if e.get("promotable") else regular_entries).append(
                entry
            )

        return format_short_term_memory_prompt(
            regular_entries=regular_entries,
            promotable_entries=promotable_entries,
        )

    def get_long_term_memory_prompt(self) -> str:
        """
        Generate a system prompt describing long-term memory capabilities.

        This prompt is static as it describes the search interface rather
        than listing specific contents.

        Returns:
            System prompt string for long-term memory
        """
        return LONG_TERM_MEMORY_PROMPT

    def get_memory_prompt(self) -> str:
        """
        Generate a complete system prompt describing all memory capabilities.

        Combines short-term and long-term memory prompts with usage guidelines.

        Returns:
            Complete system prompt string for memory capabilities
        """
        short_term_prompt = self.get_short_term_memory_prompt()
        long_term_prompt = self.get_long_term_memory_prompt()
        return format_memory_prompt(
            short_term_prompt=short_term_prompt,
            long_term_prompt=long_term_prompt,
        )

    # -------------------------------------------------------------------------
    # Trace
    # -------------------------------------------------------------------------
    def _add_trace_event(self, message: ChatMessage) -> None:
        """Override: richer trace logging with sub-agent traces and error detection."""
        import json

        event = {
            "type": "message",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": message.role,
            "content": message.content,
        }

        # Handle assistant messages with tool calls
        if message.role == "assistant" and message.tool_calls:
            event["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        # Handle tool result messages
        if message.role == "tool":
            event["tool_call_id"] = getattr(message, "tool_call_id", None)
            event["tool_name"] = message.name

            # Check for sub-agent trace (launch_sub_agent tool)
            if message.name == "launch_sub_agent":
                try:
                    result_data = json.loads(message.content or "{}")
                    if "execution_trace" in result_data:
                        event["sub_agent_trace"] = result_data["execution_trace"]
                except (json.JSONDecodeError, TypeError):
                    pass

            # Check for errors
            content = message.content or ""
            if "error" in content.lower() or "exception" in content.lower():
                event["is_error"] = True

        self._trace_log.append(event)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory state to a dictionary."""
        return {
            "scope": self.scope.model_dump(),
            "model_name": self._model_name,
            "messages": [m.model_dump() for m in self._messages],
            "config": {
                "system_prompt": self._system_prompt,
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        model_name: Optional[str] = None,
        short_term_store: Optional[MemoryStore] = None,
        long_term_store: Optional[VectorStore] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> "DatabaseMemory":
        """Deserialize memory state from a dictionary."""
        scope = MemoryScope(**data.get("scope", {}))
        config_data = data.get("config", {})
        config = DatabaseMemoryConfig(**config_data) if config_data else None
        system_prompt = config_data.get("system_prompt") if config_data else None
        resolved_model_name = model_name or data.get("model_name", "")

        memory = cls(
            scope=scope,
            model_name=resolved_model_name,
            config=config,
            short_term_store=short_term_store,
            long_term_store=long_term_store,
            embed_fn=embed_fn,
            system_prompt=system_prompt,
        )

        messages_data = data.get("messages", [])
        for msg_data in messages_data:
            memory._append_message(ChatMessage(**msg_data))

        return memory
