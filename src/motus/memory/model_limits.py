"""
Model context length limits for various LLM providers.

Contains max token limits for models from OpenAI, Anthropic, Google, and xAI.
Used to determine thresholds for memory compaction and context management.

Usage:
    from motus.memory import get_model_limits, ModelLimits

    limits = get_model_limits("gpt-4o")
    if limits:
        print(f"Context window: {limits.context_window}")
        print(f"Max output: {limits.max_output_tokens}")
"""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Dict, Optional


class ModelProvider(StrEnum):
    """LLM provider identifiers."""

    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    XAI = auto()


@dataclass(frozen=True)
class ModelLimits:
    """Token limits and metadata for a specific model."""

    model_id: str
    provider: ModelProvider
    context_window: int  # Total context window size in tokens
    max_output_tokens: int  # Maximum tokens in completion/response
    description: str = ""

    @property
    def max_input_tokens(self) -> int:
        """Approximate max input tokens (context - output)."""
        return self.context_window - self.max_output_tokens

    def get_compaction_threshold(self, ratio: float = 0.75) -> int:
        """
        Get recommended token count to trigger compaction.

        Args:
            ratio: Fraction of context window to use as threshold (default 75%)

        Returns:
            Token count threshold for triggering compaction
        """
        return int(self.context_window * ratio)


# =============================================================================
# OpenAI Models
# =============================================================================

OPENAI_MODELS: Dict[str, ModelLimits] = {
    # GPT-5.2 series
    "gpt-5.2": ModelLimits(
        model_id="gpt-5.2",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5.2 - most capable coding and agentic model",
    ),
    "gpt-5.2-2025-12-11": ModelLimits(
        model_id="gpt-5.2-2025-12-11",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5.2 December 2025 snapshot",
    ),
    "gpt-5.2-pro": ModelLimits(
        model_id="gpt-5.2-pro",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5.2 Pro - enterprise with max context and agent support",
    ),
    "gpt-5.2-chat-latest": ModelLimits(
        model_id="gpt-5.2-chat-latest",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=16_384,
        description="GPT-5.2 Instant - fast lookups and rapid support",
    ),
    "gpt-5.2-codex": ModelLimits(
        model_id="gpt-5.2-codex",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5.2 Codex - optimized for code generation",
    ),
    # GPT-5 series
    "gpt-5": ModelLimits(
        model_id="gpt-5",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5 - coding, reasoning, and agentic tasks",
    ),
    "gpt-5-thinking": ModelLimits(
        model_id="gpt-5-thinking",
        provider=ModelProvider.OPENAI,
        context_window=196_000,
        max_output_tokens=128_000,
        description="GPT-5 Thinking - reasoning-focused variant",
    ),
    "gpt-5-mini": ModelLimits(
        model_id="gpt-5-mini",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5 Mini - fast and cost-efficient",
    ),
    "gpt-5-nano": ModelLimits(
        model_id="gpt-5-nano",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5 Nano - lightweight and affordable",
    ),
    # GPT-4o series
    "gpt-4o": ModelLimits(
        model_id="gpt-4o",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=16_384,
        description="GPT-4o - multimodal flagship model",
    ),
    "gpt-4o-2024-11-20": ModelLimits(
        model_id="gpt-4o-2024-11-20",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=16_384,
        description="GPT-4o November 2024 snapshot",
    ),
    "gpt-4o-2024-08-06": ModelLimits(
        model_id="gpt-4o-2024-08-06",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=16_384,
        description="GPT-4o August 2024 snapshot",
    ),
    "gpt-4o-mini": ModelLimits(
        model_id="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=16_384,
        description="GPT-4o Mini - affordable small model",
    ),
    "gpt-4o-mini-2024-07-18": ModelLimits(
        model_id="gpt-4o-mini-2024-07-18",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=16_384,
        description="GPT-4o Mini July 2024 snapshot",
    ),
    # o1 reasoning series
    "o1": ModelLimits(
        model_id="o1",
        provider=ModelProvider.OPENAI,
        context_window=200_000,
        max_output_tokens=100_000,
        description="o1 - reasoning model",
    ),
    "o1-2024-12-17": ModelLimits(
        model_id="o1-2024-12-17",
        provider=ModelProvider.OPENAI,
        context_window=200_000,
        max_output_tokens=100_000,
        description="o1 December 2024 snapshot",
    ),
    "o1-mini": ModelLimits(
        model_id="o1-mini",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=65_536,
        description="o1-mini - fast reasoning model",
    ),
    "o1-mini-2024-09-12": ModelLimits(
        model_id="o1-mini-2024-09-12",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=65_536,
        description="o1-mini September 2024 snapshot",
    ),
    "o1-preview": ModelLimits(
        model_id="o1-preview",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=32_768,
        description="o1-preview - reasoning preview model",
    ),
    "o3-mini": ModelLimits(
        model_id="o3-mini",
        provider=ModelProvider.OPENAI,
        context_window=200_000,
        max_output_tokens=100_000,
        description="o3-mini - latest reasoning model",
    ),
    # GPT-4 Turbo series
    "gpt-4-turbo": ModelLimits(
        model_id="gpt-4-turbo",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=4_096,
        description="GPT-4 Turbo with vision",
    ),
    "gpt-4-turbo-2024-04-09": ModelLimits(
        model_id="gpt-4-turbo-2024-04-09",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=4_096,
        description="GPT-4 Turbo April 2024 snapshot",
    ),
    "gpt-4-turbo-preview": ModelLimits(
        model_id="gpt-4-turbo-preview",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=4_096,
        description="GPT-4 Turbo preview",
    ),
    "gpt-4-0125-preview": ModelLimits(
        model_id="gpt-4-0125-preview",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=4_096,
        description="GPT-4 Turbo January 2024 preview",
    ),
    "gpt-4-1106-preview": ModelLimits(
        model_id="gpt-4-1106-preview",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
        max_output_tokens=4_096,
        description="GPT-4 Turbo November 2023 preview",
    ),
    # GPT-4 series
    "gpt-4": ModelLimits(
        model_id="gpt-4",
        provider=ModelProvider.OPENAI,
        context_window=8_192,
        max_output_tokens=8_192,
        description="GPT-4 base model",
    ),
    "gpt-4-0613": ModelLimits(
        model_id="gpt-4-0613",
        provider=ModelProvider.OPENAI,
        context_window=8_192,
        max_output_tokens=8_192,
        description="GPT-4 June 2023 snapshot",
    ),
    "gpt-4-32k": ModelLimits(
        model_id="gpt-4-32k",
        provider=ModelProvider.OPENAI,
        context_window=32_768,
        max_output_tokens=8_192,
        description="GPT-4 with 32k context",
    ),
    "gpt-4-32k-0613": ModelLimits(
        model_id="gpt-4-32k-0613",
        provider=ModelProvider.OPENAI,
        context_window=32_768,
        max_output_tokens=8_192,
        description="GPT-4 32k June 2023 snapshot",
    ),
    # GPT-3.5 Turbo series
    "gpt-3.5-turbo": ModelLimits(
        model_id="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        context_window=16_385,
        max_output_tokens=4_096,
        description="GPT-3.5 Turbo",
    ),
    "gpt-3.5-turbo-0125": ModelLimits(
        model_id="gpt-3.5-turbo-0125",
        provider=ModelProvider.OPENAI,
        context_window=16_385,
        max_output_tokens=4_096,
        description="GPT-3.5 Turbo January 2024",
    ),
    "gpt-3.5-turbo-1106": ModelLimits(
        model_id="gpt-3.5-turbo-1106",
        provider=ModelProvider.OPENAI,
        context_window=16_385,
        max_output_tokens=4_096,
        description="GPT-3.5 Turbo November 2023",
    ),
    "gpt-3.5-turbo-16k": ModelLimits(
        model_id="gpt-3.5-turbo-16k",
        provider=ModelProvider.OPENAI,
        context_window=16_385,
        max_output_tokens=4_096,
        description="GPT-3.5 Turbo 16k context",
    ),
}


# =============================================================================
# Anthropic Models
# =============================================================================

ANTHROPIC_MODELS: Dict[str, ModelLimits] = {
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": ModelLimits(
        model_id="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=8_192,
        description="Claude 3.5 Sonnet - latest balanced model",
    ),
    "claude-3-5-sonnet-20240620": ModelLimits(
        model_id="claude-3-5-sonnet-20240620",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=8_192,
        description="Claude 3.5 Sonnet June 2024",
    ),
    "claude-3-5-sonnet-latest": ModelLimits(
        model_id="claude-3-5-sonnet-latest",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=8_192,
        description="Claude 3.5 Sonnet latest alias",
    ),
    "claude-3-5-haiku-20241022": ModelLimits(
        model_id="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=8_192,
        description="Claude 3.5 Haiku - fast and affordable",
    ),
    "claude-3-5-haiku-latest": ModelLimits(
        model_id="claude-3-5-haiku-latest",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=8_192,
        description="Claude 3.5 Haiku latest alias",
    ),
    # Claude 3 series
    "claude-3-opus-20240229": ModelLimits(
        model_id="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=4_096,
        description="Claude 3 Opus - most capable",
    ),
    "claude-3-opus-latest": ModelLimits(
        model_id="claude-3-opus-latest",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=4_096,
        description="Claude 3 Opus latest alias",
    ),
    "claude-3-sonnet-20240229": ModelLimits(
        model_id="claude-3-sonnet-20240229",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=4_096,
        description="Claude 3 Sonnet - balanced",
    ),
    "claude-3-haiku-20240307": ModelLimits(
        model_id="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=4_096,
        description="Claude 3 Haiku - fast",
    ),
    # Claude Opus 4 series
    "claude-opus-4-20250514": ModelLimits(
        model_id="claude-opus-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=32_000,
        description="Claude Opus 4 - most capable model",
    ),
    "claude-opus-4-5-20251101": ModelLimits(
        model_id="claude-opus-4-5-20251101",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=16_000,
        description="Claude Opus 4.5 - frontier intelligence",
    ),
    "claude-opus-4-6": ModelLimits(
        model_id="claude-opus-4-6",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,  # 1M with beta header, 200k default
        max_output_tokens=128_000,
        description="Claude Opus 4.6 - latest most capable model",
    ),
    # Claude Sonnet 4 series
    "claude-sonnet-4-20250514": ModelLimits(
        model_id="claude-sonnet-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=16_000,
        description="Claude Sonnet 4 - balanced capability",
    ),
    "claude-sonnet-4-5-20250929": ModelLimits(
        model_id="claude-sonnet-4-5-20250929",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=64_000,
        description="Claude Sonnet 4.5 - strong coding and reasoning",
    ),
    "claude-sonnet-4-6": ModelLimits(
        model_id="claude-sonnet-4-6",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=64_000,
        description="Claude Sonnet 4.6 - latest balanced model",
    ),
    # Claude Haiku 4 series
    "claude-haiku-4-5-20251001": ModelLimits(
        model_id="claude-haiku-4-5-20251001",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=64_000,
        description="Claude Haiku 4.5 - fast and affordable",
    ),
    # Convenience aliases
    "claude-3-5-sonnet": ModelLimits(
        model_id="claude-3-5-sonnet",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=8_192,
        description="Claude 3.5 Sonnet alias",
    ),
    "claude-3-5-haiku": ModelLimits(
        model_id="claude-3-5-haiku",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=8_192,
        description="Claude 3.5 Haiku alias",
    ),
    "claude-3-opus": ModelLimits(
        model_id="claude-3-opus",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=4_096,
        description="Claude 3 Opus alias",
    ),
    "claude-3-sonnet": ModelLimits(
        model_id="claude-3-sonnet",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=4_096,
        description="Claude 3 Sonnet alias",
    ),
    "claude-3-haiku": ModelLimits(
        model_id="claude-3-haiku",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
        max_output_tokens=4_096,
        description="Claude 3 Haiku alias",
    ),
}


# =============================================================================
# Google Models
# =============================================================================

GOOGLE_MODELS: Dict[str, ModelLimits] = {
    # Gemini 2.5 series
    "gemini-2.5-pro": ModelLimits(
        model_id="gemini-2.5-pro",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=65_536,
        description="Gemini 2.5 Pro - advanced reasoning and coding",
    ),
    "gemini-2.5-flash": ModelLimits(
        model_id="gemini-2.5-flash",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=65_535,
        description="Gemini 2.5 Flash - fast reasoning",
    ),
    "gemini-2.5-flash-lite": ModelLimits(
        model_id="gemini-2.5-flash-lite",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=65_535,
        description="Gemini 2.5 Flash-Lite - fastest and most affordable",
    ),
    # Gemini 2.0 series
    "gemini-2.0-flash": ModelLimits(
        model_id="gemini-2.0-flash",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=8_192,
        description="Gemini 2.0 Flash - fast multimodal",
    ),
    "gemini-2.0-flash-exp": ModelLimits(
        model_id="gemini-2.0-flash-exp",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=8_192,
        description="Gemini 2.0 Flash experimental",
    ),
    "gemini-2.0-flash-thinking-exp": ModelLimits(
        model_id="gemini-2.0-flash-thinking-exp",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=8_192,
        description="Gemini 2.0 Flash with thinking",
    ),
    # Gemini 1.5 series
    "gemini-1.5-pro": ModelLimits(
        model_id="gemini-1.5-pro",
        provider=ModelProvider.GOOGLE,
        context_window=2_097_152,
        max_output_tokens=8_192,
        description="Gemini 1.5 Pro - 2M context",
    ),
    "gemini-1.5-pro-latest": ModelLimits(
        model_id="gemini-1.5-pro-latest",
        provider=ModelProvider.GOOGLE,
        context_window=2_097_152,
        max_output_tokens=8_192,
        description="Gemini 1.5 Pro latest",
    ),
    "gemini-1.5-pro-002": ModelLimits(
        model_id="gemini-1.5-pro-002",
        provider=ModelProvider.GOOGLE,
        context_window=2_097_152,
        max_output_tokens=8_192,
        description="Gemini 1.5 Pro version 002",
    ),
    "gemini-1.5-flash": ModelLimits(
        model_id="gemini-1.5-flash",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=8_192,
        description="Gemini 1.5 Flash - fast 1M context",
    ),
    "gemini-1.5-flash-latest": ModelLimits(
        model_id="gemini-1.5-flash-latest",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=8_192,
        description="Gemini 1.5 Flash latest",
    ),
    "gemini-1.5-flash-002": ModelLimits(
        model_id="gemini-1.5-flash-002",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=8_192,
        description="Gemini 1.5 Flash version 002",
    ),
    "gemini-1.5-flash-8b": ModelLimits(
        model_id="gemini-1.5-flash-8b",
        provider=ModelProvider.GOOGLE,
        context_window=1_048_576,
        max_output_tokens=8_192,
        description="Gemini 1.5 Flash 8B - smallest",
    ),
    # Gemini 1.0 series
    "gemini-1.0-pro": ModelLimits(
        model_id="gemini-1.0-pro",
        provider=ModelProvider.GOOGLE,
        context_window=32_760,
        max_output_tokens=8_192,
        description="Gemini 1.0 Pro",
    ),
    "gemini-pro": ModelLimits(
        model_id="gemini-pro",
        provider=ModelProvider.GOOGLE,
        context_window=32_760,
        max_output_tokens=8_192,
        description="Gemini Pro alias",
    ),
    # Gemma models (open source)
    "gemma-2-27b-it": ModelLimits(
        model_id="gemma-2-27b-it",
        provider=ModelProvider.GOOGLE,
        context_window=8_192,
        max_output_tokens=8_192,
        description="Gemma 2 27B instruction tuned",
    ),
    "gemma-2-9b-it": ModelLimits(
        model_id="gemma-2-9b-it",
        provider=ModelProvider.GOOGLE,
        context_window=8_192,
        max_output_tokens=8_192,
        description="Gemma 2 9B instruction tuned",
    ),
}


# =============================================================================
# xAI Models
# =============================================================================

XAI_MODELS: Dict[str, ModelLimits] = {
    # Grok 2 series
    "grok-2": ModelLimits(
        model_id="grok-2",
        provider=ModelProvider.XAI,
        context_window=131_072,
        max_output_tokens=4_096,
        description="Grok 2 - flagship model",
    ),
    "grok-2-1212": ModelLimits(
        model_id="grok-2-1212",
        provider=ModelProvider.XAI,
        context_window=131_072,
        max_output_tokens=4_096,
        description="Grok 2 December 2024",
    ),
    "grok-2-latest": ModelLimits(
        model_id="grok-2-latest",
        provider=ModelProvider.XAI,
        context_window=131_072,
        max_output_tokens=4_096,
        description="Grok 2 latest alias",
    ),
    # Grok 2 Vision
    "grok-2-vision": ModelLimits(
        model_id="grok-2-vision",
        provider=ModelProvider.XAI,
        context_window=32_768,
        max_output_tokens=4_096,
        description="Grok 2 Vision - multimodal",
    ),
    "grok-2-vision-1212": ModelLimits(
        model_id="grok-2-vision-1212",
        provider=ModelProvider.XAI,
        context_window=32_768,
        max_output_tokens=4_096,
        description="Grok 2 Vision December 2024",
    ),
    "grok-2-vision-latest": ModelLimits(
        model_id="grok-2-vision-latest",
        provider=ModelProvider.XAI,
        context_window=32_768,
        max_output_tokens=4_096,
        description="Grok 2 Vision latest alias",
    ),
    # Grok Beta
    "grok-beta": ModelLimits(
        model_id="grok-beta",
        provider=ModelProvider.XAI,
        context_window=131_072,
        max_output_tokens=4_096,
        description="Grok Beta",
    ),
    "grok-vision-beta": ModelLimits(
        model_id="grok-vision-beta",
        provider=ModelProvider.XAI,
        context_window=8_192,
        max_output_tokens=4_096,
        description="Grok Vision Beta",
    ),
}


# =============================================================================
# MiniMax Models (via OpenRouter)
# =============================================================================

MINIMAX_MODELS: Dict[str, ModelLimits] = {
    "minimax/minimax-m2.5": ModelLimits(
        model_id="minimax/minimax-m2.5",
        provider=ModelProvider.OPENAI,  # OpenRouter uses OpenAI-compatible API
        context_window=196_608,
        max_output_tokens=65_536,
        description="MiniMax M2.5 - cost-efficient reasoning model",
    ),
    "minimax/minimax-m2.7": ModelLimits(
        model_id="minimax/minimax-m2.7",
        provider=ModelProvider.OPENAI,
        context_window=204_800,
        max_output_tokens=131_072,
        description="MiniMax M2.7 - next-gen reasoning model",
    ),
    "openai/gpt-5-mini": ModelLimits(
        model_id="openai/gpt-5-mini",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5 Mini - fast, cheap, with caching on OpenRouter",
    ),
    "openai/gpt-5.3-codex": ModelLimits(
        model_id="openai/gpt-5.3-codex",
        provider=ModelProvider.OPENAI,
        context_window=400_000,
        max_output_tokens=128_000,
        description="GPT-5.3 Codex - strong coding model via OpenRouter",
    ),
}


# =============================================================================
# Combined Registry
# =============================================================================

ALL_MODELS: Dict[str, ModelLimits] = {
    **OPENAI_MODELS,
    **ANTHROPIC_MODELS,
    **GOOGLE_MODELS,
    **XAI_MODELS,
    **MINIMAX_MODELS,
}


def get_model_limits(model_id: str) -> Optional[ModelLimits]:
    """
    Get the limits for a specific model.

    Handles provider-prefixed names (e.g., "anthropic/claude-opus-4-6")
    by stripping the prefix before lookup.

    Args:
        model_id: The model identifier (e.g., "gpt-4o", "claude-3-5-sonnet",
                  "anthropic/claude-opus-4-6")

    Returns:
        ModelLimits if found, None otherwise
    """
    result = ALL_MODELS.get(model_id)
    if result is not None:
        return result

    # Strip provider prefix (e.g., "anthropic/claude-opus-4-6" -> "claude-opus-4-6")
    if "/" in model_id:
        bare_id = model_id.rsplit("/", 1)[-1]
        return ALL_MODELS.get(bare_id)

    return None


def estimate_compaction_threshold(
    model_id: str,
    safety_ratio: float = 0.75,
) -> Optional[int]:
    """
    Estimate the token threshold at which to trigger memory compaction.

    The threshold is a fraction of the full context window. With split-compaction,
    only the history before the last turn is summarized, so the summarization call
    is always bounded regardless of output size.

    Args:
        model_id: The model identifier
        safety_ratio: Fraction of context window to use as threshold (default 75%)

    Returns:
        Token threshold for triggering compaction, or None if model not found
    """
    limits = get_model_limits(model_id)
    if limits is None:
        return None

    return int(limits.context_window * safety_ratio)
