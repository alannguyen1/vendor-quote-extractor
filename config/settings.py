"""
Application settings module using pydantic-settings.

Settings are loaded from environment variables and .env file.
Use get_settings() to access the singleton instance.
"""

import logging
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Supported LLM providers
LLMProviderType = Literal["openai", "anthropic", "groq", "fireworks", "gemini"]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM API Keys - All optional for flexible deployment
    # Demo mode uses Groq by default; other providers available if keys are set
    openai_api_key: str = ""
    anthropic_api_key: str = ""  # Optional - needed for failover

    # Provider API keys for Speed Optimization (Phase 2-4)
    groq_api_key: str = ""
    fireworks_api_key: str = ""
    gemini_api_key: str = ""

    # LLM Configuration
    # Default to Groq for demo - fast response times, free tier available
    llm_provider: LLMProviderType = "groq"
    llm_model: str = "gpt-4o-2024-08-06"
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Provider models for Speed Optimization (Phase 2-4)
    # Use llama-3.3-70b-versatile for better reasoning on TCV vs Net distinctions
    # The 70B model has much better instruction following than the 17B scout model
    groq_model: str = "llama-3.3-70b-versatile"
    # Phase A1: Default to 8B model for speed (~3s faster than 70B)
    # Use 70B only for complex docs via routing
    fireworks_model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    gemini_model: str = "gemini-2.5-flash"

    # Provider failover chain (primary first, then fallbacks in order)
    # Demo mode: Groq first, then OpenAI/Anthropic as backups
    llm_fallback_chain: list[str] = ["groq", "openai", "anthropic"]

    # Smart Routing Configuration (Speed Phase 3)
    routing_enabled: bool = True  # Feature enabled (v42.0)
    routing_strategy: Literal["auto", "speed", "accuracy", "cost"] = "speed"

    # Speed Optimization Settings (Phase A: Safe Optimizations)
    # Phase A3: Lower DPI from 300 to 150 (~1.5s faster for scanned PDFs)
    render_dpi: int = 150  # DPI for rendering scanned PDFs (lower = faster)
    # Phase A2: Reduce truncation from 30K to 15K (~0.5s faster, less tokens)
    max_text_chars: int = 15000  # Maximum characters for text extraction truncation
    max_table_chars: int = 10000  # Maximum characters for table extraction truncation
    enable_prompt_caching: bool = True  # Enable prompt caching for supported providers
    parallel_parsing: bool = True  # Run text and table extraction in parallel
    # Phase A4: Disable few-shot by default (~0.3s faster, skip DB queries)
    enable_few_shot: bool = False  # Enable few-shot learning from user corrections

    # H6: Per-Provider Context Window Management
    # Different providers have different context window sizes:
    # - Groq: 128K tokens (70B model) - fast with large context
    # - OpenAI: 128K tokens (~250K chars) - balanced
    # - Gemini: 1M+ tokens (~2M chars) - largest context
    # - Anthropic: 200K tokens (~400K chars) - large context
    # - Fireworks: model-dependent (~60K chars typical)
    provider_max_text_chars: dict[str, int] = {
        "groq": 15000,
        "openai": 250000,
        "gemini": 500000,  # Conservative limit for Gemini's 1M+ context
        "anthropic": 200000,
        "fireworks": 60000,
    }
    provider_max_table_chars: dict[str, int] = {
        "groq": 15000,
        "openai": 50000,
        "gemini": 100000,
        "anthropic": 50000,
        "fireworks": 20000,
    }

    # H6: Document complexity threshold for routing
    # Route to large-context provider (Gemini) if table row count exceeds this
    complexity_routing_threshold: int = 40  # Route to Gemini if >40 table rows

    # Chunked Extraction Settings (for large documents with many line items)
    chunking_enabled: bool = True  # Enable chunked line-item extraction
    chunk_rows_per_call: int = 5  # Max rows per LLM call in chunked mode
    chunking_min_rows: int = 30  # Min total rows to trigger chunked extraction
    large_doc_text_budget: int = 8000  # Phase 2 H0c
    chunk_parallel_concurrency: int = 3  # Parallel chunking

    # Provider-aware single-pass bypass: skip chunking if provider context
    # window can handle the total row count in a single call.
    # Key = provider name, value = max rows for single-pass extraction.
    provider_single_pass_max_rows: dict[str, int] = {
        "groq": 30,
        "openai": 40,
        "gemini": 150,
        "anthropic": 60,
        "fireworks": 30,
    }

    # Application Settings
    max_file_size_mb: int = 10
    max_pdf_pages: int = (
        20  # Demo constraint to prevent memory issues on Streamlit Cloud
    )
    processing_timeout_seconds: int = 45
    confidence_threshold: float = 0.90

    # UI/API Configuration
    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: int = 10  # Timeout for API requests from UI
    max_recent_extractions: int = 50  # Max extractions to return in list endpoint
    cors_origins: list[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",  # React dev server
    ]

    # Validation Thresholds
    tax_rate_warning_threshold: float = 0.20  # Warn if tax exceeds 20% of subtotal
    math_tolerance: float = 0.01  # $0.01 tolerance for math validation
    msrp_variance_threshold: float = 0.05  # 5% variance triggers multiple totals UI

    # UI Display Settings
    max_error_causes: int = 5  # Max possible causes to display in validation errors
    llm_bottleneck_threshold: int = 80  # % of extract time to flag as LLM bottleneck

    # Line Item Confidence Settings
    line_item_low_confidence_threshold: float = 0.9  # Below this triggers logging

    # Post-extraction quantity correction
    quantity_auto_correction: bool = True  # Enable post-extraction quantity correction

    # Logging
    log_level: str = "INFO"

    def has_anthropic_key(self) -> bool:
        """Check if a valid Anthropic API key is configured for failover."""
        return bool(
            self.anthropic_api_key
            and self.anthropic_api_key != "sk-ant-PLACEHOLDER_ADD_YOUR_KEY"
        )

    def has_groq_key(self) -> bool:
        """Check if a valid Groq API key is configured."""
        return bool(self.groq_api_key and len(self.groq_api_key) > 10)

    def has_fireworks_key(self) -> bool:
        """Check if a valid Fireworks API key is configured."""
        return bool(self.fireworks_api_key and len(self.fireworks_api_key) > 10)

    def has_gemini_key(self) -> bool:
        """Check if a valid Gemini API key is configured."""
        return bool(self.gemini_api_key and len(self.gemini_api_key) > 10)

    def has_openai_key(self) -> bool:
        """Check if a valid OpenAI API key is configured."""
        return bool(self.openai_api_key and len(self.openai_api_key) > 10)

    def get_available_providers(self) -> list[str]:
        """Get list of providers with valid API keys configured."""
        available = []
        if self.has_openai_key():
            available.append("openai")
        if self.has_anthropic_key():
            available.append("anthropic")
        if self.has_groq_key():
            available.append("groq")
        if self.has_fireworks_key():
            available.append("fireworks")
        if self.has_gemini_key():
            available.append("gemini")
        return available


@lru_cache
def get_settings() -> Settings:
    """
    Get the singleton Settings instance.

    Uses lru_cache to ensure only one instance is created.
    The settings are loaded once from .env and environment variables.
    """
    # pyright doesn't understand pydantic-settings loads from env
    return Settings()  # type: ignore[call-arg]


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """
    Configure application logging.

    Args:
        settings: Optional Settings instance. If not provided, uses get_settings().

    Returns:
        The configured root logger for the application.
    """
    if settings is None:
        settings = get_settings()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get our application logger
    logger = logging.getLogger("vendor_quote_extractor")
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Log startup info - show provider-specific model
    logger.info(f"LLM provider: {settings.llm_provider}")
    # Show the model for the configured provider
    model_map = {
        "openai": settings.llm_model,
        "anthropic": settings.anthropic_model,
        "groq": settings.groq_model,
        "fireworks": settings.fireworks_model,
        "gemini": settings.gemini_model,
    }
    active_model = model_map.get(settings.llm_provider, settings.llm_model)
    logger.info(f"LLM model: {active_model}")
    logger.info(f"Anthropic failover available: {settings.has_anthropic_key()}")
    logger.info(f"Max file size: {settings.max_file_size_mb}MB")
    logger.info(f"Processing timeout: {settings.processing_timeout_seconds}s")
    logger.info(f"Confidence threshold: {settings.confidence_threshold}")

    return logger
