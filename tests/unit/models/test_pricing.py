"""Tests for motus.models.pricing — cost resolution logic."""

import pytest

from motus.models.pricing import calculate_cost, get_pricing


class TestGetPricing:
    def test_exact_match(self):
        pricing = get_pricing("claude-haiku-4-5")
        assert pricing is not None
        assert pricing["input"] == 1.00
        assert pricing["output"] == 5.00

    def test_unknown_model_returns_none(self):
        assert get_pricing("totally-fake-model") is None


class TestCalculateCost:
    def test_basic_from_tokens(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = calculate_cost("claude-haiku-4-5", usage)
        # haiku: input=$1/M, output=$5/M → (1000*1 + 500*5)/1M = 0.0035
        assert cost == pytest.approx(0.0035, rel=1e-6)

    def test_prefers_response_cost_over_tokens(self):
        """When the gateway (model_proxy / OpenRouter) has already computed cost
        and put it in usage.cost, motus must trust that number and skip the
        local pricing table — single source of truth."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.42}
        cost = calculate_cost("claude-haiku-4-5", usage)
        assert cost == 0.42

    def test_response_cost_wins_even_for_unknown_model(self):
        """Gateway-provided cost works even when we have no local pricing."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.42}
        cost = calculate_cost("totally-fake-model", usage)
        assert cost == 0.42

    def test_response_cost_zero_falls_back_to_tokens(self):
        """cost=0 means provider didn't report cost; fall back to local table."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0}
        cost = calculate_cost("claude-haiku-4-5", usage)
        assert cost == pytest.approx(0.0035, rel=1e-6)

    def test_malformed_cost_falls_back_to_tokens(self):
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cost": "not-a-number",
        }
        cost = calculate_cost("claude-haiku-4-5", usage)
        assert cost == pytest.approx(0.0035, rel=1e-6)

    def test_no_model_and_no_response_cost_returns_none(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        assert calculate_cost(None, usage) is None

    def test_no_model_with_response_cost_still_works(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.42}
        assert calculate_cost(None, usage) == 0.42

    def test_empty_usage_returns_none(self):
        assert calculate_cost("claude-haiku-4-5", {}) is None

    def test_unknown_model_no_response_cost_returns_none(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        assert calculate_cost("totally-fake-model", usage) is None
