"""Configuration for the tracing system."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from motus.auth.credentials import get_api_key, get_api_url
from motus.config import CONFIG


class CollectionLevel(str, Enum):
    """What level of data to collect during tracing.

    Levels (from least to most overhead):
    - DISABLED: No data collection
    - BASIC: Task timing and names only (minimal overhead)
    - DETAILED: + Full messages, tool args, model outputs (user debugging mode)
    """

    DISABLED = "disabled"
    BASIC = "basic"
    DETAILED = "detailed"


def _get_collection_level_default() -> CollectionLevel:
    """Determine collection level from environment variables.

    Priority:
    1. MOTUS_COLLECTION_LEVEL (explicit level)
    2. MOTUS_TRACING=1 → DETAILED (user debugging)
    3. Default → BASIC (minimal overhead)
    """
    # Explicit collection level
    if "MOTUS_COLLECTION_LEVEL" in os.environ:
        level = os.getenv("MOTUS_COLLECTION_LEVEL", "").lower()
        try:
            return CollectionLevel(level)
        except ValueError:
            return CollectionLevel.BASIC

    # MOTUS_TRACING=1 or MOTUS_TRACING_ONLINE=1 → user debugging mode (detailed + export)
    for var in ("MOTUS_TRACING", "MOTUS_TRACING_ONLINE"):
        if os.getenv(var, "").lower() in ("1", "true", "yes"):
            return CollectionLevel.DETAILED

    # Default: basic (minimal overhead)
    return CollectionLevel.BASIC


def _get_export_enabled_default() -> bool:
    """Determine if export should be enabled from environment.

    Returns True if:
    - MOTUS_TRACING=1 (user debugging mode)
    - OR MOTUS_TRACING_ONLINE=1 (live tracing implies persistence)
    - OR MOTUS_TRACING_EXPORT=1 (explicit export)
    """
    for var in ("MOTUS_TRACING", "MOTUS_TRACING_ONLINE", "MOTUS_TRACING_EXPORT"):
        if os.getenv(var, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _get_online_tracing_default() -> bool:
    """Check environment variable for online tracing setting."""
    return os.getenv("MOTUS_TRACING_ONLINE", "").lower() in ("1", "true", "yes")


def _get_cloud_api_url() -> str | None:
    return get_api_url()


def _get_cloud_api_key() -> str | None:
    return get_api_key()


def _get_project_default() -> str | None:
    """Resolve project identifier from env vars, then motus.toml."""
    val = os.getenv("MOTUS_PROJECT")
    if val:
        return val
    return CONFIG.get("project_id")


def _get_build_default() -> str | None:
    """Resolve build identifier from env vars, then motus.toml."""
    val = os.getenv("MOTUS_BUILD")
    if val:
        return val
    return CONFIG.get("build_id")


def _get_log_dir_default() -> Path:
    """Determine the log directory from environment or generate a timestamped default.

    Priority:
    1. MOTUS_TRACING_DIR env var → use as-is (no timestamping)
    2. Default → traces/trace_<YYYYMMDD_HHMMSS>/
    """
    env_dir = os.getenv("MOTUS_TRACING_DIR")
    if env_dir:
        return Path(env_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"traces/trace_{timestamp}")


@dataclass
class TraceConfig:
    """Configuration for motus tracing.

    Usage Patterns:
    ---------------

    1. User Debugging (via env vars):
        $ MOTUS_TRACING=1 python script.py
        → collection_level=DETAILED, export_enabled=True

        $ MOTUS_TRACING_ONLINE=1 python script.py
        → collection_level=DETAILED, export_enabled=True, online_tracing=True

    2. Meta-Agent (programmatic):
        config = TraceConfig(
            collection_level=CollectionLevel.DETAILED,  # for full trace data
            export_enabled=False,
            online_tracing=False
        )

    3. Production (minimal overhead):
        config = TraceConfig(collection_level=CollectionLevel.DISABLED)
        # Or: $ MOTUS_COLLECTION_LEVEL=disabled python script.py

    Attributes:
        collection_level: What data to collect (disabled/basic/detailed)
        export_enabled: Whether to enable export_trace() file writing
        online_tracing: Whether to enable live tracing via SSE push
        log_dir: Directory to store trace files. Defaults to
            MOTUS_TRACING_DIR env var if set, otherwise
            traces/trace_<YYYYMMDD_HHMMSS>/
        json_path: Filename for the JSON trace state
    """

    # Core settings
    collection_level: CollectionLevel = field(
        default_factory=_get_collection_level_default
    )
    export_enabled: bool = field(default_factory=_get_export_enabled_default)
    online_tracing: bool = field(default_factory=_get_online_tracing_default)

    # Output settings
    log_dir: Path = field(default_factory=_get_log_dir_default)
    json_path: str = "tracer_state.json"

    # Cloud trace store settings
    cloud_api_url: str | None = field(default_factory=_get_cloud_api_url)
    cloud_api_key: str | None = field(default_factory=_get_cloud_api_key)

    # Project identity (from env vars or motus.toml)
    project: str | None = field(default_factory=_get_project_default)
    build: str | None = field(default_factory=_get_build_default)

    # Session identity (set programmatically by serve2, env var as fallback)
    session_id: str | None = field(
        default_factory=lambda: os.getenv("MOTUS_SESSION_ID")
    )

    def __post_init__(self):
        # Ensure log_dir is a Path object
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)

    @property
    def is_collecting(self) -> bool:
        """Whether any data collection is enabled."""
        return self.collection_level != CollectionLevel.DISABLED

    @property
    def collect_metrics(self) -> bool:
        """Whether to collect metrics (failures, bottlenecks)."""
        return self.collection_level == CollectionLevel.DETAILED

    @property
    def collect_full_traces(self) -> bool:
        """Whether to collect full traces (messages, tool args, outputs)."""
        return self.collection_level == CollectionLevel.DETAILED

    @property
    def cloud_enabled(self) -> bool:
        """Whether cloud trace upload is configured."""
        return bool(self.cloud_api_url and self.cloud_api_key)
