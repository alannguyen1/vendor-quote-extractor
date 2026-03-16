"""
Configuration module for Vendor Quote Extractor.

Provides application settings loaded from environment variables.

Usage:
    from config import get_settings, Settings

    settings = get_settings()
    print(settings.llm_provider)  # "openai"
"""

from config.settings import Settings, configure_logging, get_settings

__all__ = ["Settings", "get_settings", "configure_logging"]
