"""
Memory agents — on-demand agents for updating and searching the memory tree.

Two agent types, both using ReActAgent with scoped file tools:
  MemoryUpdateAgent — files a chunk of messages into the tree (WRITE)
  MemorySearchAgent — searches the tree and returns an answer (READ)

Created on demand by BackgroundMemory — no persistent agent instances.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from motus.models.base import BaseChatClient

from .events import MemoryChunk
from .memory_tools import make_memory_tools
from .prompts import MEMORY_SEARCH_AGENT_PROMPT, MEMORY_UPDATE_AGENT_PROMPT

logger = logging.getLogger(__name__)


def initialize_memory_dir(root: Path) -> None:
    """Create the memory directory with memory.md entry point and raw_logs/."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw_logs").mkdir(exist_ok=True)

    entry = root / "memory.md"
    if not entry.exists():
        entry.write_text("# Memory\n\nNo facts stored yet.\n")


def write_raw_chunk(root: Path, chunk: MemoryChunk) -> None:
    """Write a raw chunk to raw_logs/. Immutable — skips if already exists."""
    path = root / "raw_logs" / f"{chunk.chunk_id}.md"
    if path.exists():
        return
    meta = {
        "chunk_id": chunk.chunk_id,
        "timestamp": chunk.timestamp.isoformat(),
        "num_messages": len(chunk.messages),
    }
    # Index messages so the memory agent can reference by index range
    indexed_lines = []
    for i, m in enumerate(chunk.messages):
        text = (m.content or "").replace("\n", "\\n")
        if m.tool_calls:
            tool_names = ", ".join(tc.function.name for tc in m.tool_calls)
            indexed_lines.append(f"[{i}] {m.role} [tool_calls: {tool_names}]")
        else:
            indexed_lines.append(f"[{i}] {m.role}: {text}")
    content = (
        f"<meta>\n{json.dumps(meta, indent=2)}\n</meta>\n\n"
        f"<messages>\n" + "\n".join(indexed_lines) + "\n</messages>\n"
    )
    path.write_text(content)
    logger.debug(f"Wrote raw chunk: {chunk.chunk_id}")


def _make_react_agent(
    client: BaseChatClient,
    model_name: str,
    root: Path,
    system_prompt: str,
    max_steps: int,
):
    """Create a fresh ReActAgent with scoped file tools for the memory tree."""
    # Lazy import to avoid circular: memory → memory_agent → ReActAgent → base_agent → memory
    from motus.agent.react_agent import ReActAgent

    return ReActAgent(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        tools=make_memory_tools(root),
        max_steps=max_steps,
    )


class MemoryUpdateAgent:
    """Files a chunk of conversation messages into the memory tree.

    Created on demand at each compaction point. Writes the raw chunk to
    disk, then uses a ReActAgent to navigate the tree, create leaf
    annotations (ref_<chunk_id>.md), and update node summaries.
    """

    def __init__(
        self,
        client: BaseChatClient,
        model_name: str,
        root: Path,
        max_steps: int = 30,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = model_name
        self._root = root
        self._max_steps = max_steps
        self._system_prompt = system_prompt or MEMORY_UPDATE_AGENT_PROMPT

    async def run(self, chunk: MemoryChunk) -> None:
        """Write raw chunk and file it into the memory tree."""
        write_raw_chunk(self._root, chunk)
        prompt = (
            f"A new conversation chunk has been stored at "
            f"raw_logs/{chunk.chunk_id}.md ({len(chunk.messages)} messages, indexed [0]-[{len(chunk.messages) - 1}]).\n\n"
            f"Read the chunk, identify distinct topics, and for each topic:\n"
            f"- Create a leaf file ref_{chunk.chunk_id}_<msg_start>_<msg_end>.md with an annotation\n"
            f"- Place it under the appropriate tree node\n"
            f"- Update parent summaries up to root.md"
        )
        try:
            agent = _make_react_agent(
                self._client,
                self._model,
                self._root,
                self._system_prompt,
                max_steps=None,  # unlimited — runs async, won't block main agent
            )
            await agent(prompt)
            logger.debug(f"Processed chunk {chunk.chunk_id}")
        except Exception as e:
            logger.warning(f"Failed to process chunk {chunk.chunk_id}: {e}")


class MemorySearchAgent:
    """Searches the memory tree and returns an answer.

    Created on demand when the main agent calls search_memory().
    Uses a ReActAgent to navigate the tree, read relevant nodes,
    and compose a response.
    """

    def __init__(
        self,
        client: BaseChatClient,
        model_name: str,
        root: Path,
        max_steps: int = 30,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = model_name
        self._root = root
        self._max_steps = max_steps
        self._system_prompt = system_prompt or MEMORY_SEARCH_AGENT_PROMPT

    async def run(self, query: str) -> str:
        """Query the memory tree and return an answer."""
        prompt = (
            f"Search the memory tree for the following and return a concise answer:\n\n"
            f"{query}"
        )
        try:
            agent = _make_react_agent(
                self._client,
                self._model,
                self._root,
                self._system_prompt,
                self._max_steps,
            )
            result = await agent(prompt)
            logger.debug(f"Answered query: {query!r}")
            return result
        except Exception as e:
            logger.warning(f"Memory query failed: {e}")
            return f"Memory lookup failed: {e}"
