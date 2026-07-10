from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TextModelPricing:
    input_per_1k: float
    output_per_1k: float


TEXT_MODEL_PRICING: Dict[str, TextModelPricing] = {
    # === OpenAI ===
    # GPT-5 family (approx per 1K tokens)
    "gpt-5": TextModelPricing(input_per_1k=0.00125, output_per_1k=0.0100),
    "gpt-5-mini": TextModelPricing(input_per_1k=0.00025, output_per_1k=0.00200),
    "gpt-5-nano": TextModelPricing(input_per_1k=0.00005, output_per_1k=0.00040),
    "gpt-5-nano-2025-08-07": TextModelPricing(
        input_per_1k=0.00025,
        output_per_1k=0.00100,
    ),
    # GPT-4 variants
    "gpt-4o": TextModelPricing(input_per_1k=0.00250, output_per_1k=0.01000),
    "gpt-4o-mini": TextModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
    "gpt-4-turbo": TextModelPricing(input_per_1k=0.0100, output_per_1k=0.0300),
    "gpt-4": TextModelPricing(input_per_1k=0.0300, output_per_1k=0.0600),
    # Budget older models
    "gpt-3.5-turbo": TextModelPricing(input_per_1k=0.00050, output_per_1k=0.00150),
    # === Anthropic Claude models ===
    "claude-opus-4": TextModelPricing(input_per_1k=0.0150, output_per_1k=0.0750),
    "claude-sonnet-4": TextModelPricing(input_per_1k=0.0030, output_per_1k=0.0150),
    "claude-haiku": TextModelPricing(input_per_1k=0.00025, output_per_1k=0.00125),
    # Claude 4.5 (both naming conventions)
    "claude-opus-4.5": TextModelPricing(input_per_1k=0.0150, output_per_1k=0.0750),
    "claude-opus-4-5": TextModelPricing(input_per_1k=0.0150, output_per_1k=0.0750),
    # === Google Gemini ===
    "gemini-2.5-pro": TextModelPricing(input_per_1k=0.00125, output_per_1k=0.0100),
    "gemini-2.5-flash": TextModelPricing(input_per_1k=0.00030, output_per_1k=0.00250),
    # === Meta / LLaMA (via API or cloud) ===
    # Note: open-source hosting cost varies; these are typical marketplace rates
    "llama-3-70b": TextModelPricing(input_per_1k=0.00099, output_per_1k=0.00132),
    # === xAI Grok models ===
    # Pricing here is illustrative — check provider docs if available
    "grok-4-fast": TextModelPricing(input_per_1k=0.00025, output_per_1k=0.00250),
    "grok-4": TextModelPricing(input_per_1k=0.0030, output_per_1k=0.0150),
    # === Misc / placeholder ===
    # You can add other popular open models here (e.g., Mistral, Cohere, Alibaba, DeepSeek), with their actual pricing
}
