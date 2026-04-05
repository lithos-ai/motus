"""
Local filesystem implementation of ConversationLogStore.

Stores JSONL log files directly under the given directory:
    {path}/{session_id}.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from ..interfaces import ConversationLogStore

logger = logging.getLogger(__name__)


class LocalConversationLogStore(ConversationLogStore):
    """Local filesystem conversation log store.

    Stores one JSONL file per session directly under the given path:
        {path}/{session_id}.jsonl
    """

    DEFAULT_PATH = "~/.motus/conversation_logs"

    def __init__(self, path: Union[str, Path] = DEFAULT_PATH):
        self._path = Path(path).expanduser()

    def _session_path(self, session_id: str) -> Path:
        return self._path / f"{session_id}.jsonl"

    def append(self, session_id: str, entry: Dict[str, Any]) -> None:
        path = self._session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_entries(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed log line in {path}")
                    continue
        return entries

    def search_messages(
        self, session_id: str, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()
        with open(path, "r") as f:
            for line in f:
                if len(results) >= max_results:
                    break
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "message":
                    continue
                msg = entry.get("message", {})
                content = msg.get("content", "")
                if (
                    isinstance(content, str)
                    and content
                    and query_lower in content.lower()
                ):
                    results.append(
                        {
                            "role": msg.get("role"),
                            "content": content,
                            "ts": entry.get("ts"),
                        }
                    )
        return results

    def exists(self, session_id: str) -> bool:
        return self._session_path(session_id).exists()

    def list_sessions(self) -> List[str]:
        if not self._path.exists():
            return []
        return sorted(p.stem for p in self._path.glob("*.jsonl"))
