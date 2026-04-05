"""
Embedding function factory for the memory system.

Creates embedding functions from model names that can be used with the memory system.
Supports OpenAI embedding models with plans for additional providers.
"""

import os
from enum import StrEnum, auto
from functools import lru_cache
from typing import Callable, Dict, List, NamedTuple, Optional


class EmbeddingProvider(StrEnum):
    """Supported embedding providers."""

    OPENAI = auto()
    # Future providers can be added here:
    # COHERE = auto()
    # GOOGLE = auto()
    # LOCAL = auto()


class ModelConfig(NamedTuple):
    """Configuration for an embedding model."""

    provider: EmbeddingProvider
    dimension: int
    max_tokens: int


# Model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # OpenAI models
    "text-embedding-3-small": ModelConfig(
        provider=EmbeddingProvider.OPENAI,
        dimension=1536,
        max_tokens=8191,
    ),
    "text-embedding-3-large": ModelConfig(
        provider=EmbeddingProvider.OPENAI,
        dimension=3072,
        max_tokens=8191,
    ),
    "text-embedding-ada-002": ModelConfig(
        provider=EmbeddingProvider.OPENAI,
        dimension=1536,
        max_tokens=8191,
    ),
}

# Default model to use if none specified
DEFAULT_MODEL = "text-embedding-3-small"


def get_embedding_dimension(model_name: str) -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model_name: The name of the embedding model.

    Returns:
        The embedding dimension (vector size) for the model.

    Raises:
        ValueError: If the model is not supported.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown embedding model: {model_name}. "
            f"Supported models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name].dimension


def get_embedding_fn(
    model_name: str,
    api_key: Optional[str] = None,
) -> Optional[Callable[[str], List[float]]]:
    """
    Create a synchronous embedding function for the given model.

    Args:
        model_name: The name of the embedding model (e.g., "text-embedding-3-small").
        api_key: Optional API key. If not provided, uses environment variables.

    Returns:
        A function that takes a string and returns its embedding as a list of floats,
        or None if the API key is not provided.

    Raises:
        ValueError: If the model is not supported.
        ImportError: If required dependencies are not installed.

    Example:
        embed_fn = get_embedding_fn("text-embedding-3-small")
        if embed_fn:
            embedding = embed_fn("Hello, world!")
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown embedding model: {model_name}. "
            f"Supported models: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]

    if config.provider == EmbeddingProvider.OPENAI:
        return _create_openai_embed_fn(model_name, api_key)

    raise ValueError(f"Provider {config.provider} not yet implemented")


def get_embedding_fn_async(
    model_name: str,
    api_key: Optional[str] = None,
) -> Optional[Callable[[str], List[float]]]:
    """
    Create an async-compatible embedding function for the given model.

    Note: This returns a synchronous function that can be called from async code.
    For true async support, the memory system would need to be updated to support
    async embedding functions.

    Args:
        model_name: The name of the embedding model.
        api_key: Optional API key.

    Returns:
        A function that takes a string and returns its embedding,
        or None if the API key is not provided.
    """
    # For now, return the sync version
    # True async support would require changes to the memory system interface
    return get_embedding_fn(model_name, api_key)


def _create_openai_embed_fn(
    model_name: str,
    api_key: Optional[str] = None,
) -> Callable[[str], List[float]]:
    """
    Create an OpenAI embedding function.

    Args:
        model_name: The OpenAI embedding model name.
        api_key: Optional API key. Falls back to OPENAI_API_KEY env var.

    Returns:
        A function that embeds text using OpenAI's API.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI package is required for OpenAI embeddings. "
            "Install it with: pip install openai"
        ) from e

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        return None

    client = OpenAI(api_key=resolved_api_key)

    @lru_cache(maxsize=1024)
    def embed(text: str) -> tuple:
        """Embed text and return as tuple for caching."""
        response = client.embeddings.create(
            model=model_name,
            input=text,
        )
        return tuple(response.data[0].embedding)

    def embed_fn(text: str) -> List[float]:
        """Embed text using OpenAI's API."""
        return list(embed(text))

    return embed_fn
