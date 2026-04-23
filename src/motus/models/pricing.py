"""Model pricing registry for cost calculation.

Provides per-model token pricing and a cost calculation function.
Used by ReActAgent to track cost automatically.
"""

# Pricing per million tokens (USD)
_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-4-7": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    "claude-opus-4-6": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    # MiniMax (via OpenRouter)
    "minimax/minimax-m2.5": {
        "input": 0.118,
        "output": 0.99,
        "cache_write": 0.118,
        "cache_read": 0.059,
    },
    "minimax/minimax-m2.7": {
        "input": 0.30,
        "output": 1.20,
        "cache_write": 0.30,
        "cache_read": 0.06,
    },
    # Kimi (via OpenRouter)
    "moonshotai/kimi-k2.5": {
        "input": 0.38,
        "output": 1.72,
        "cache_write": 0.38,
        "cache_read": 0.19,
    },
    # OpenAI (via OpenRouter)
    "openai/gpt-5-mini": {
        "input": 0.25,
        "output": 2.00,
        "cache_write": 0.25,
        "cache_read": 0.025,
    },
    "openai/gpt-5.3-codex": {
        "input": 1.75,
        "output": 14.00,
        "cache_write": 1.75,
        "cache_read": 0.175,
    },
}


def get_pricing(model: str) -> dict[str, float] | None:
    """Get pricing for a model. Tries exact match, then prefix match."""
    pricing = _PRICING.get(model)
    if pricing:
        return pricing
    return next(
        (v for k, v in _PRICING.items() if model.startswith(k) or k.startswith(model)),
        None,
    )


def calculate_cost(model: str | None, usage: dict) -> float | None:
    """Calculate cost in USD from token usage and model pricing.

    If the response includes an explicit ``cost`` field in ``usage`` (emitted
    by OpenAI-compatible gateways like OpenRouter or LithosAI's model proxy),
    use that value directly — it's the provider/gateway's authoritative
    number. Otherwise fall back to computing from tokens using the local
    pricing table.

    Args:
        model: Model identifier.
        usage: Dict with token counts (prompt_tokens, completion_tokens,
               cache_creation_input_tokens, cache_read_input_tokens). May also
               include a pre-computed ``cost`` field.

    Returns:
        Cost in USD, or None if no cost can be determined.
    """
    if not usage:
        return None
    # Provider/gateway-reported cost takes precedence — single source of truth.
    if usage.get("cost"):
        try:
            return float(usage["cost"])
        except (TypeError, ValueError):
            pass  # fall through to local computation
    if not model:
        return None
    pricing = get_pricing(model)
    if not pricing:
        return None
    cost = (
        usage.get("prompt_tokens", 0) * pricing["input"]
        + usage.get("completion_tokens", 0) * pricing["output"]
        + usage.get("cache_creation_input_tokens", 0) * pricing["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * pricing["cache_read"]
    ) / 1_000_000
    return round(cost, 8)
