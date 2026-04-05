"""
File-system based storage implementation.

Provides a file-based storage backend for short-term memory,
storing entries as JSON files organized by scope.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..interfaces import MemoryStore
from ..models import MemoryScope

logger = logging.getLogger("MemoryHelper")


class FileSystemStore(MemoryStore[Dict[str, Any]]):
    """
    File-system based storage for short-term memory.
    Stores entries as JSON files organized by scope.
    """

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str, scope: MemoryScope) -> Path:
        scope_path = self.base_path / scope.to_path_prefix()
        scope_path.mkdir(parents=True, exist_ok=True)
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return scope_path / f"{safe_key}.json"

    def store(self, key: str, value: Dict[str, Any], scope: MemoryScope) -> None:
        path = self._get_path(key, scope)
        data = {
            "key": key,
            "value": value,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    def retrieve(self, key: str, scope: MemoryScope) -> Optional[Dict[str, Any]]:
        path = self._get_path(key, scope)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data.get("value")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read memory file {path}: {e}")
            return None

    def delete(self, key: str, scope: MemoryScope) -> bool:
        path = self._get_path(key, scope)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_keys(self, scope: MemoryScope, prefix: str = "") -> List[str]:
        scope_path = self.base_path / scope.to_path_prefix()
        if not scope_path.exists():
            return []
        keys = []
        for file_path in scope_path.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                key = data.get("key", "")
                if key.startswith(prefix):
                    keys.append(key)
            except (json.JSONDecodeError, IOError):
                continue
        return keys

    def clear(self, scope: MemoryScope) -> int:
        scope_path = self.base_path / scope.to_path_prefix()
        if not scope_path.exists():
            return 0
        count = 0
        for file_path in scope_path.glob("*.json"):
            file_path.unlink()
            count += 1
        return count
