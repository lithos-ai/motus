# Integration Tests (VCR Cassettes)

These tests replay recorded HTTP interactions (LLM calls, web searches, etc.)
so they run in seconds with **zero network access and zero token cost**.

## Quick Start

```bash
# Run all integration tests (replay mode, no API keys needed)
pytest tests/integration/examples/ -v

# Run a specific test
pytest tests/integration/examples/test_omni_integration.py::TestOmniDemo -v
```

## Recording Cassettes

When you add a new example or need to refresh recorded data:

```bash
OPENROUTER_API_KEY=... BRAVE_API_KEY=... \
    pytest tests/integration/examples/test_omni_integration.py -v --vcr-record=all
```

This runs the real workflow, captures every HTTP request/response into a
YAML cassette file, and automatically:

- **Filters sensitive headers** (`authorization`, `api-key`, etc.)
- **Scrubs large blobs** (base64 images replaced with a 1x1 pixel placeholder)
- **Strips encrypted reasoning** (`reasoning_details` with type `reasoning.encrypted` removed from both requests and responses; summaries are kept)

Cassettes are stored in each example's `cassettes_vcrpy/` directory
(e.g. `examples/omni/cassettes_vcrpy/`).

## How It Works

| Component | Role |
|---|---|
| `conftest.py` | Shared VCR config, request/response scrubbing, custom body matcher, mock fixtures |
| `--vcr-record=all` | CLI flag to switch from replay to recording |
| `cassettes_vcrpy/*.yaml` | Recorded HTTP interactions (committed to repo) |

### Parallel Agent Matching

Agents run in parallel (e.g. multiple slides generated concurrently), so request
order is non-deterministic. VCR matches requests using `method + host + path + query + body`:

- **`body`** distinguishes OpenRouter POST requests (each has different conversation history)
- **`query`** distinguishes Brave GET requests (each has different `?q=...` search term)
- A custom body matcher normalizes JSON whitespace (SDK vs VCR formatting differences)

### Adding a New Example Test

1. Write a test that uses the `configured_vcr` fixture from `conftest.py`:
   ```python
   def test_my_example(self, configured_vcr):
       with configured_vcr.use_cassette("path/to/cassette.yaml"):
           # code that makes HTTP calls
           ...
   ```
2. Record the cassette: `pytest path/to/test.py -v --vcr-record=all`
3. Verify replay: `pytest path/to/test.py -v`
4. Commit the cassette YAML alongside the test.

## CI

In CI, tests run in replay mode by default (`record_mode="none"`).
No API keys or network access required. Fake keys are injected automatically.
