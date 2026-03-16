"""
LLM Provider abstraction layer for structured extraction.

This module provides a unified interface for different LLM providers
(OpenAI, Anthropic, and future: Groq, Fireworks, Gemini) with automatic
failover and provider selection based on configuration.

Usage:
    from src.extraction.providers import get_provider, get_providers
    from config import get_settings

    settings = get_settings()

    # Get primary provider
    primary = get_provider("openai", settings)

    # Get all configured providers in failover order
    providers = get_providers(settings)
"""

import logging
from typing import Literal

from config import Settings
from src.extraction.providers.anthropic_provider import AnthropicProvider
from src.extraction.providers.base import (
    BaseLLMProvider,
    LLMProvider,
    ProviderConfig,
)
from src.extraction.providers.fireworks_provider import FireworksProvider
from src.extraction.providers.gemini_provider import GeminiProvider
from src.extraction.providers.groq_provider import GroqProvider
from src.extraction.providers.openai_provider import OpenAIProvider

logger = logging.getLogger("vendor_quote_extractor.providers")

# Type alias for supported provider names (includes future providers)
ProviderName = Literal["openai", "anthropic", "groq", "fireworks", "gemini"]

# Registry of provider classes
PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "fireworks": FireworksProvider,
    "gemini": GeminiProvider,
}


def get_provider_config(provider_name: str, settings: Settings) -> ProviderConfig:
    """
    Get provider configuration from settings.

    Args:
        provider_name: Name of the provider (openai, anthropic, groq, fireworks, gemini)
        settings: Application settings instance

    Returns:
        ProviderConfig for the specified provider

    Raises:
        ValueError: If provider is not supported
    """
    if provider_name == "openai":
        return ProviderConfig(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            vision_model=settings.llm_model,  # GPT-4o handles both
        )
    elif provider_name == "anthropic":
        return ProviderConfig(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            vision_model=settings.anthropic_model,  # Claude 3.5 handles both
        )
    elif provider_name == "groq":
        return ProviderConfig(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            vision_model=settings.groq_model,  # Llama-4-Scout handles both
        )
    elif provider_name == "fireworks":
        return ProviderConfig(
            api_key=settings.fireworks_api_key,
            model=settings.fireworks_model,
            vision_model=settings.fireworks_model,
        )
    elif provider_name == "gemini":
        return ProviderConfig(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            vision_model=settings.gemini_model,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def get_provider(provider_name: str, settings: Settings) -> BaseLLMProvider:
    """
    Get a configured provider instance by name.

    Args:
        provider_name: Name of the provider (openai, anthropic)
        settings: Application settings instance

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider is not supported or not configured
    """
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported: {list(PROVIDER_REGISTRY.keys())}"
        )

    config = get_provider_config(provider_name, settings)
    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(config)


def get_providers(settings: Settings) -> list[BaseLLMProvider]:
    """
    Get all configured providers in failover order.

    The primary provider comes first, followed by any available
    failover providers.

    Args:
        settings: Application settings instance

    Returns:
        List of configured provider instances in priority order
    """
    providers: list[BaseLLMProvider] = []

    # Primary provider first
    primary_name = settings.llm_provider
    try:
        primary = get_provider(primary_name, settings)
        providers.append(primary)
        logger.info(f"Primary provider: {primary_name}")
    except Exception as e:
        logger.error(f"Failed to initialize primary provider {primary_name}: {e}")

    # Add failover providers from registry
    for name in PROVIDER_REGISTRY.keys():
        if name == primary_name:
            continue

        # Check if provider has API key configured
        if not is_provider_available(name, settings):
            logger.debug(f"{name.capitalize()} failover not configured (no API key)")
            continue

        try:
            provider = get_provider(name, settings)
            providers.append(provider)
            logger.info(f"Failover provider: {name}")
        except Exception as e:
            logger.warning(f"Failed to initialize failover provider {name}: {e}")

    return providers


def is_provider_available(provider_name: str, settings: Settings) -> bool:
    """
    Check if a provider is available and properly configured.

    Args:
        provider_name: Name of the provider to check
        settings: Application settings instance

    Returns:
        True if provider can be initialized and has valid config
    """
    # Provider must be in registry to be available
    # Note: groq, fireworks, gemini not in registry yet (Phase 2-4)
    if provider_name == "openai":
        return bool(settings.openai_api_key)
    elif provider_name == "anthropic":
        return settings.has_anthropic_key()
    elif provider_name == "groq":
        return settings.has_groq_key()
    elif provider_name == "fireworks":
        return settings.has_fireworks_key()
    elif provider_name == "gemini":
        return settings.has_gemini_key()

    return False


__all__ = [
    "LLMProvider",
    "BaseLLMProvider",
    "ProviderConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "GroqProvider",
    "FireworksProvider",
    "GeminiProvider",
    "get_provider",
    "get_providers",
    "get_provider_config",
    "is_provider_available",
    "ProviderName",
    "PROVIDER_REGISTRY",
]
