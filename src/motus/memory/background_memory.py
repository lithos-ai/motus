"""
BackgroundMemory — CompactionBase with long-term cross-session memory.

Extends CompactionBase with:
- On-demand memory agents that manage a hierarchical memory tree:
    MemoryUpdateAgent — files messages into the tree at compaction time
    MemorySearchAgent — searches the tree via search_memory() tool
- Boundary-aware split-compaction inherited from CompactionBase

At each compaction point, the full pre-compaction messages are packaged
as a MemoryChunk and processed by a MemoryUpdateAgent asynchronously.
The main agent continues with the compacted context immediately.

Usage::

    memory = BackgroundMemory(
        memory_client=haiku_client,
        memory_model_name="claude-haiku-4-5",
    )
    async with ReActAgent(client=sonnet_client, model_name="...", memory=memory) as agent:
        result = await agent("Hello")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from motus.models.base import ChatMessage

from .background.events import MemoryChunk
from .background.memory_agent import (
    MemorySearchAgent,
    MemoryUpdateAgent,
    initialize_memory_dir,
)
from .compaction_base import CompactFn, CompactionBase
from .config import BackgroundMemoryConfig, CompactionMemoryConfig
from .session_state import BackgroundSessionState, SessionState

logger = logging.getLogger(__name__)


class BackgroundMemory(CompactionBase):
    """
    In-session memory with compaction and background long-term memory.

    Memory update flow
    ------------------
    At each compaction point:
    1. All pre-compaction messages are packaged as a single MemoryChunk
    2. The context is compacted (split-compact from CompactionBase)
    3. A MemoryUpdateAgent processes the chunk asynchronously (fire-and-forget)

    The memory tree is only queried after compaction, when old context has
    been summarized away and the agent needs to recall specific details.
    """

    def __init__(
        self,
        memory_client: Any,
        memory_model_name: str,
        *,
        bg_config: Optional[BackgroundMemoryConfig] = None,
        config: Optional[CompactionMemoryConfig] = None,
        compact_fn: Optional[CompactFn] = None,
        on_compact: Optional[Callable[[Dict[str, Any]], None]] = None,
        # Main agent model — injected by AgentBase.set_model() if omitted
        client: Optional[Any] = None,
        model_name: Optional[str] = None,
        # Direct overrides (prefer bg_config)
        root: Optional[Path] = None,
        memory_agent_max_steps: Optional[int] = None,
        enable_memory_tools: Optional[bool] = None,
        # Custom prompts for memory agents (None = use defaults)
        memory_update_prompt: Optional[str] = None,
        memory_search_prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            memory_client: Client for the memory agents (cheap model).
            memory_model_name: Model name for the memory agents.
            bg_config: BackgroundMemoryConfig for tree location and agent settings.
            config: CompactionMemoryConfig for threshold / model tuning.
            compact_fn: Optional custom compaction function.
            on_compact: Optional callback after each compaction.
            client: Main agent client for compaction LLM calls (filled by set_model).
            model_name: Main agent model name (filled by set_model).
            root: Override bg_config.root.
            memory_agent_max_steps: Override bg_config.memory_agent_max_steps.
            enable_memory_tools: Override bg_config.enable_memory_tools.
            memory_update_prompt: Custom system prompt for MemoryUpdateAgent.
            memory_search_prompt: Custom system prompt for MemorySearchAgent.
        """
        bgc = bg_config or BackgroundMemoryConfig()
        self._memory_client = memory_client
        self._memory_model_name = memory_model_name
        self._memory_root = Path(root or bgc.root).expanduser()
        self._memory_max_steps = (
            memory_agent_max_steps
            if memory_agent_max_steps is not None
            else bgc.memory_agent_max_steps
        )
        resolved_enable_tools = (
            enable_memory_tools
            if enable_memory_tools is not None
            else bgc.enable_memory_tools
        )

        super().__init__(
            config=config,
            compact_fn=compact_fn,
            on_compact=on_compact,
            enable_memory_tools=resolved_enable_tools,
            client=client,
            model_name=model_name,
        )

        self._memory_update_prompt = memory_update_prompt
        self._memory_search_prompt = memory_search_prompt

        # Initialize the memory tree directory structure
        initialize_memory_dir(self._memory_root)

        self._update_task: Optional[asyncio.Task] = None

    # -------------------------------------------------------------------------
    # Compaction with memory update
    # -------------------------------------------------------------------------

    async def _do_compact(self) -> None:
        """Package pre-compaction messages, compact, then update memory async."""
        # Wait for any previous memory update to finish (sequential processing)
        if self._update_task and not self._update_task.done():
            try:
                await self._update_task
            except Exception:
                pass  # logged by the update agent

        # Capture all messages before compaction
        pre_compact_messages: List[ChatMessage] = list(self._messages)

        # Create a chunk from the full pre-compaction context
        chunk = MemoryChunk.create(
            messages=pre_compact_messages,
            turn_start=0,
            turn_end=len(pre_compact_messages) - 1,
        )

        # Compact the context (split-compact from CompactionBase)
        await self.compact()

        # Fire-and-forget: create an update agent and process the chunk
        update_agent = MemoryUpdateAgent(
            client=self._memory_client,
            model_name=self._memory_model_name,
            root=self._memory_root,
            max_steps=self._memory_max_steps,
            system_prompt=self._memory_update_prompt,
        )
        self._update_task = asyncio.create_task(
            update_agent.run(chunk),
            name=f"memory-update-{chunk.chunk_id}",
        )
        logger.info(
            f"Compaction complete, memory update started "
            f"(chunk {chunk.chunk_id}, {len(pre_compact_messages)} messages)"
        )

    # -------------------------------------------------------------------------
    # Memory tool
    # -------------------------------------------------------------------------

    def build_tools(self) -> list:
        """Expose search_memory() as a memory tool for the agent."""
        if not self._enable_memory_tools:
            return []

        # Capture params for the closure — search agent created on each call
        _client = self._memory_client
        _model = self._memory_model_name
        _root = self._memory_root
        _max_steps = self._memory_max_steps
        _search_prompt = self._memory_search_prompt

        async def search_memory(query: str) -> str:
            """Search long-term memory to retrieve information from past conversations.

            This is a READ-ONLY search tool — it queries the memory store for
            previously recorded facts, decisions, preferences, and context.
            It does NOT store or save new information.

            Use this when:
            - The conversation context has been compacted and you need details
              that were discussed earlier but are no longer visible
            - The user asks about something from a previous session
            - You need to recall specific facts, names, dates, or decisions

            Do NOT use this to store or remember new information — that happens
            automatically in the background as the conversation progresses.

            Args:
                query: A natural language question describing what you want
                    to look up (e.g. "What project is the user working on?",
                    "What are the user's editor preferences?").
            """
            search_agent = MemorySearchAgent(
                client=_client,
                model_name=_model,
                root=_root,
                max_steps=_max_steps,
                system_prompt=_search_prompt,
            )
            return await search_agent.run(query)

        return [search_memory]

    # -------------------------------------------------------------------------
    # Lifecycle TODO: wire to agentBase class lifecycle
    # -------------------------------------------------------------------------

    async def on_agent_start(self) -> None:
        """No-op — memory agents are created on demand."""
        pass

    async def on_agent_complete(self) -> None:
        """Wait for any pending memory update to finish."""
        if self._update_task and not self._update_task.done():
            try:
                await self._update_task
            except Exception:
                pass  # logged by the update agent

    def reset(self) -> Dict[str, int]:
        return super().reset()

    # -------------------------------------------------------------------------
    # Session state
    # -------------------------------------------------------------------------

    def get_session_state(self) -> BackgroundSessionState:
        """Capture current session state."""
        return BackgroundSessionState(
            messages=self._messages.copy(),
            system_prompt=self._system_prompt,
            tree_root=str(self._memory_root),
        )

    @classmethod
    def restore(
        cls,
        state: "SessionState",
        *,
        memory_client: Any,
        memory_model_name: str,
        bg_config: Optional[BackgroundMemoryConfig] = None,
        config: Optional[CompactionMemoryConfig] = None,
        compact_fn=None,
        on_compact=None,
        client: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> "BackgroundMemory":
        """Restore from a BackgroundSessionState snapshot.

        Reconnects to the existing memory tree on disk.

        Args:
            state: Must be a BackgroundSessionState instance.
            memory_client: Client for the memory agents.
            memory_model_name: Model name for the memory agents.
            bg_config: BackgroundMemoryConfig (root is overridden from state).
            All other args mirror BackgroundMemory.__init__().
        """
        if not isinstance(state, BackgroundSessionState):
            raise TypeError(
                f"Expected BackgroundSessionState, got {type(state).__name__}"
            )

        memory = cls(
            memory_client=memory_client,
            memory_model_name=memory_model_name,
            bg_config=bg_config,
            config=config,
            compact_fn=compact_fn,
            on_compact=on_compact,
            client=client,
            model_name=model_name,
            root=Path(state.tree_root),
        )
        memory._system_prompt = state.system_prompt
        memory._messages = list(state.messages)
        return memory
