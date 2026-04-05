"""
Shared fixtures for example integration tests using vcrpy.

Usage:
    # Replay from cassettes (CI / fast local run, no API keys needed)
    pytest tests/integration/examples/ -v

    # Re-record cassettes (slow, requires real API keys)
    pytest tests/integration/examples/ -v --vcr-record=all
"""

import gzip
import json
import os
import re

import pytest
import vcr

# ---------------------------------------------------------------------------
# Ensure API keys exist at import time (before example modules are imported
# during test collection).  Real keys are only needed when recording.
# Track which keys we injected so the autouse fixture can clean them up.
# ---------------------------------------------------------------------------
_INJECTED_KEYS: set[str] = set()
for _key in (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "BRAVE_API_KEY",
    "ANTHROPIC_API_KEY",
):
    if _key not in os.environ:
        os.environ[_key] = "fake-key-for-collection"
        _INJECTED_KEYS.add(_key)
# Ensure OpenAI client targets OpenRouter so cassettes recorded against
# openrouter.ai replay correctly even without real credentials.
if "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    _INJECTED_KEYS.add("OPENAI_BASE_URL")


# ---------------------------------------------------------------------------
# Scrubbing helpers
# ---------------------------------------------------------------------------
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)
MAX_BLOB_SIZE = 50_000  # strings > 50 KB are candidates for replacement


def _replace_large_blobs(obj):
    """Recursively replace large base64 / opaque strings with tiny placeholders."""
    if isinstance(obj, str) and len(obj) > MAX_BLOB_SIZE:
        if obj.startswith("data:image/"):
            return "data:image/png;base64," + TINY_PNG_B64
        if re.match(r"^[A-Za-z0-9+/=\s]{1000,}$", obj[:2000]):
            return TINY_PNG_B64
    elif isinstance(obj, dict):
        return {k: _replace_large_blobs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_large_blobs(v) for v in obj]
    return obj


def _strip_reasoning(obj):
    """Strip all reasoning fields from a message-like dict.

    Removes:
    - ``reasoning`` field (readable reasoning text)
    - ``reasoning_details`` field entirely (opaque round-trip data)

    Both are non-deterministic and irrelevant to request/response matching.
    """
    if isinstance(obj, dict):
        obj = {
            k: _strip_reasoning(v)
            for k, v in obj.items()
            if k not in ("reasoning", "reasoning_details")
        }
        return obj
    elif isinstance(obj, list):
        return [_strip_reasoning(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# VCR callbacks — scrub responses and request bodies before saving
# ---------------------------------------------------------------------------
def _decode_body(body_raw):
    """Decode (possibly gzipped) bytes to (text, is_gzip)."""
    if not isinstance(body_raw, bytes):
        body_raw = body_raw.encode("utf-8")
    try:
        return gzip.decompress(body_raw).decode("utf-8"), True
    except (gzip.BadGzipFile, OSError):
        return body_raw.decode("utf-8", errors="replace"), False


def scrub_response(response):
    """VCR before_record_response — strip large blobs and reasoning."""
    body_raw = response.get("body", {}).get("string", b"")
    body_text, is_gzip = _decode_body(body_raw)

    try:
        data = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        return response

    data = _replace_large_blobs(data)
    data = _strip_reasoning(data)
    new_body = json.dumps(data).encode("utf-8")
    response["body"]["string"] = gzip.compress(new_body) if is_gzip else new_body
    return response


def scrub_request(request):
    """VCR before_record_request — strip reasoning from request bodies."""
    if not request.body:
        return request

    body = request.body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return request

    data = _strip_reasoning(data)
    request.body = json.dumps(data).encode("utf-8")
    return request


# ---------------------------------------------------------------------------
# Custom body matcher — normalizes JSON whitespace before comparison
# ---------------------------------------------------------------------------
def _json_body_matcher(r1, r2):
    """Compare request bodies by parsed JSON content, ignoring formatting."""
    b1 = r1.body or b""
    b2 = r2.body or b""

    if isinstance(b1, str):
        b1 = b1.encode("utf-8")
    if isinstance(b2, str):
        b2 = b2.encode("utf-8")

    try:
        j1 = json.loads(b1)
        j2 = json.loads(b2)
        assert _strip_reasoning(j1) == _strip_reasoning(j2)
    except (json.JSONDecodeError, UnicodeDecodeError):
        assert b1 == b2


# ---------------------------------------------------------------------------
# VCR fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configured_vcr(request):
    """A VCR instance that respects --vcr-record CLI flag."""
    record_mode = request.config.getoption("--vcr-record", default=None) or "none"
    my_vcr = vcr.VCR(
        record_mode=record_mode,
        match_on=["method", "host", "path", "query", "body"],
        filter_headers=[
            "authorization",
            "api-key",
            "x-api-key",
            "X-Subscription-Token",
        ],
        before_record_response=scrub_response,
        before_record_request=scrub_request,
    )
    my_vcr.register_matcher("body", _json_body_matcher)
    return my_vcr


# ---------------------------------------------------------------------------
# Fake API keys for replay mode
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fake_api_keys(request):
    """Set fake API keys for replay mode. Real keys are only needed when recording."""
    record_mode = request.config.getoption("--vcr-record", default=None)
    if record_mode and record_mode != "none":
        yield
        return

    saved = {}
    for key in (
        "OPENROUTER_API_KEY",
        "BRAVE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        # Keys in _INJECTED_KEYS were absent before conftest loaded — treat
        # them as originally absent so we remove them on teardown instead of
        # restoring the fake-key-for-collection sentinel.
        if key in _INJECTED_KEYS:
            saved[key] = None
        else:
            saved[key] = os.environ.get(key)
        os.environ[key] = "fake-key-for-replay"

    yield

    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
