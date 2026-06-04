"""
Session state snapshots for memory persistence and restoration.

SessionState captures the minimum information needed to restore a memory
instance across process restarts:

  - Shared part (method-independent): current context window messages + system prompt
  - Method-specific part: how to reconnect to the backing store

Serialization: each state has to_dict() / from_dict() for filesystem persistence.
SessionState.from_dict() dispatches to the correct subclass based on the "type" field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from motus.models.base import ChatMessage


@dataclass
class SessionState:
    """Method-independent session snapshot.

    Contains the current context window (messages) and system prompt.
    Subclasses add the backing-store metadata needed to reconnect after restore.
    """

    messages: List[ChatMessage]
    system_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Deserialize from dict, dispatching to the correct subclass."""
        state_type = data.get("type", "SessionState")
        target_cls = _STATE_TYPE_REGISTRY.get(state_type)
        if target_cls is None:
            raise ValueError(f"Unknown session state type: {state_type}")
        return target_cls._from_dict_impl(data)

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> "SessionState":
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return cls(messages=messages, system_prompt=data.get("system_prompt", ""))


@dataclass
class CompactionSessionState(SessionState):
    """Session state for CompactionMemory.

    Captures session identity and log store location for cross-session
    continuity via the conversation log.
    """

    session_id: str = ""
    log_base_path: Optional[str] = None
    compaction_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["session_id"] = self.session_id
        d["log_base_path"] = self.log_base_path
        d["compaction_count"] = self.compaction_count
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> "CompactionSessionState":
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return cls(
            messages=messages,
            system_prompt=data.get("system_prompt", ""),
            session_id=data.get("session_id", ""),
            log_base_path=data.get("log_base_path"),
            compaction_count=data.get("compaction_count", 0),
        )


@dataclass
class BackgroundSessionState(SessionState):
    """Session state for BackgroundMemory.

    Captures the memory tree root so the MemoryAgent can continue
    building the existing tree on disk.
    """

    tree_root: str = "~/.motus/memory"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["tree_root"] = self.tree_root
        return d

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> "BackgroundSessionState":
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return cls(
            messages=messages,
            system_prompt=data.get("system_prompt", ""),
            tree_root=data.get("tree_root", "~/.motus/memory"),
        )


_STATE_TYPE_REGISTRY: Dict[str, type] = {
    "SessionState": SessionState,
    "CompactionSessionState": CompactionSessionState,
    "BackgroundSessionState": BackgroundSessionState,
}
