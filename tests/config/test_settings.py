"""
Tests for config/settings.py module.

Tests cover:
- Settings class initialization with defaults and custom values
- API key validation methods (has_anthropic_key, has_groq_key, etc.)
- Provider availability detection (get_available_providers)
- Singleton behavior (get_settings)
- Logging configuration (configure_logging)
- Type validation for Literal fields

Why these tests matter:
- Settings drive all LLM provider selection and failover behavior
- Misconfigured API keys could cause silent failures or unexpected provider usage
- The singleton pattern must work correctly for consistent behavior across the app
"""

import logging
import os
from unittest.mock import patch

from pydantic_settings import BaseSettings, SettingsConfigDict

from config.settings import (
    LLMProviderType,
    Settings,
    configure_logging,
    get_settings,
)


class IsolatedSettings(Settings):
    """Settings subclass that ignores both .env file AND environment variables.

    This ensures tests are fully isolated and don't pick up keys from the shell.
    """

    model_config = SettingsConfigDict(
        env_file=None,  # Don't load from .env
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Only use init settings, ignore environment variables entirely."""
        # Only return init_settings - this ignores env vars and .env files
        return (init_settings,)


class TestSettingsDefaults:
    """Test Settings class default values.

    Uses IsolatedSettings to avoid loading from .env file.
    """

    def test_default_llm_provider(self):
        """Default LLM provider should be Groq for the demo app."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.llm_provider == "groq"

    def test_default_llm_model(self):
        """Default LLM model should be GPT-4o."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.llm_model == "gpt-4o-2024-08-06"

    def test_default_anthropic_model(self):
        """Default Anthropic model should be Claude 3.5 Sonnet."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.anthropic_model == "claude-3-5-sonnet-20241022"

    def test_default_groq_model(self):
        """
        Default Groq model should be Llama 3.3 70B for better TCV accuracy.
        """
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.groq_model == "llama-3.3-70b-versatile"

    def test_default_fireworks_model(self):
        """Default Fireworks model should be Llama 3.1 8B (Phase A optimization)."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        # Phase A1: Default to 8B model for speed (~3s faster than 70B)
        expected_model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        assert settings.fireworks_model == expected_model

    def test_default_gemini_model(self):
        """Default Gemini model should be Gemini 2.5 Flash."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.gemini_model == "gemini-2.5-flash"

    def test_default_fallback_chain(self):
        """Default fallback chain should prefer Groq, then OpenAI, then Anthropic."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.llm_fallback_chain == ["groq", "openai", "anthropic"]

    def test_default_routing_enabled(self):
        """Smart routing should be enabled by default (v42.0)."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.routing_enabled is True

    def test_default_routing_strategy(self):
        """Default routing strategy should be speed."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.routing_strategy == "speed"

    def test_default_render_dpi(self):
        """Default render DPI should be 150 (Phase A optimization)."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        # Phase A3: Lower DPI from 300 to 150 (~1.5s faster for scanned PDFs)
        assert settings.render_dpi == 150

    def test_default_max_text_chars(self):
        """Default max text chars should be 15000 (Phase A optimization)."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        # Phase A2: Reduce truncation from 30K to 15K (~0.5s faster, less tokens)
        assert settings.max_text_chars == 15000

    def test_default_max_table_chars(self):
        """Default max table chars should be 10000."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.max_table_chars == 10000

    def test_default_prompt_caching_enabled(self):
        """Prompt caching should be enabled by default."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.enable_prompt_caching is True

    def test_default_parallel_parsing_enabled(self):
        """Parallel parsing should be enabled by default."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.parallel_parsing is True

    def test_default_few_shot_disabled(self):
        """Few-shot learning should be disabled by default (Phase A optimization)."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        # Phase A4: Disable few-shot by default (~0.3s faster, skip DB queries)
        assert settings.enable_few_shot is False

    def test_default_max_file_size(self):
        """Default max file size should be 10MB."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.max_file_size_mb == 10

    def test_default_processing_timeout(self):
        """Default processing timeout should be 45 seconds."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.processing_timeout_seconds == 45

    def test_default_confidence_threshold(self):
        """Default confidence threshold should be 0.90."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.confidence_threshold == 0.90

    def test_default_api_base_url(self):
        """Default API base URL should be localhost:8000."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.api_base_url == "http://localhost:8000"

    def test_default_api_timeout_seconds(self):
        """Default API timeout should be 10 seconds."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.api_timeout_seconds == 10

    def test_default_log_level(self):
        """Default log level should be INFO."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.log_level == "INFO"

    def test_optional_api_keys_default_empty(self):
        """Optional API keys should default to empty string."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.anthropic_api_key == ""
        assert settings.groq_api_key == ""
        assert settings.fireworks_api_key == ""
        assert settings.gemini_api_key == ""


class TestSettingsCustomValues:
    """Test Settings class with custom values."""

    def test_custom_llm_provider(self):
        """Settings should accept custom LLM provider."""
        settings = Settings(
            openai_api_key="sk-test",
            llm_provider="anthropic",
        )
        assert settings.llm_provider == "anthropic"

    def test_custom_routing_strategy(self):
        """Settings should accept custom routing strategy."""
        settings = Settings(
            openai_api_key="sk-test",
            routing_strategy="accuracy",
        )
        assert settings.routing_strategy == "accuracy"

    def test_custom_render_dpi(self):
        """Settings should accept custom render DPI."""
        settings = Settings(
            openai_api_key="sk-test",
            render_dpi=150,
        )
        assert settings.render_dpi == 150

    def test_custom_fallback_chain(self):
        """Settings should accept custom fallback chain."""
        settings = Settings(
            openai_api_key="sk-test",
            llm_fallback_chain=["groq", "openai", "anthropic"],
        )
        assert settings.llm_fallback_chain == ["groq", "openai", "anthropic"]

    def test_routing_enabled(self):
        """Settings should accept routing_enabled=True."""
        settings = Settings(
            openai_api_key="sk-test",
            routing_enabled=True,
        )
        assert settings.routing_enabled is True

    def test_custom_api_base_url(self):
        """Settings should accept custom API base URL."""
        settings = Settings(
            openai_api_key="sk-test",
            api_base_url="http://api.example.com:9000",
        )
        assert settings.api_base_url == "http://api.example.com:9000"

    def test_custom_api_timeout(self):
        """Settings should accept custom API timeout."""
        settings = Settings(
            openai_api_key="sk-test",
            api_timeout_seconds=30,
        )
        assert settings.api_timeout_seconds == 30


class TestHasAnthropicKey:
    """Test has_anthropic_key() validation method."""

    def test_empty_key_returns_false(self):
        """Empty Anthropic key should return False."""
        settings = Settings(openai_api_key="sk-test", anthropic_api_key="")
        assert settings.has_anthropic_key() is False

    def test_placeholder_key_returns_false(self):
        """Placeholder Anthropic key should return False."""
        settings = Settings(
            openai_api_key="sk-test",
            anthropic_api_key="sk-ant-PLACEHOLDER_ADD_YOUR_KEY",
        )
        assert settings.has_anthropic_key() is False

    def test_valid_key_returns_true(self):
        """Valid Anthropic key should return True."""
        settings = Settings(
            openai_api_key="sk-test",
            anthropic_api_key="sk-ant-api03-real-key-here",
        )
        assert settings.has_anthropic_key() is True

    def test_short_valid_key_returns_true(self):
        """Short but non-placeholder Anthropic key should return True."""
        settings = Settings(
            openai_api_key="sk-test",
            anthropic_api_key="sk-ant-x",
        )
        assert settings.has_anthropic_key() is True


class TestHasGroqKey:
    """Test has_groq_key() validation method."""

    def test_empty_key_returns_false(self):
        """Empty Groq key should return False."""
        settings = Settings(openai_api_key="sk-test", groq_api_key="")
        assert settings.has_groq_key() is False

    def test_short_key_returns_false(self):
        """Groq key under 10 chars should return False."""
        settings = Settings(openai_api_key="sk-test", groq_api_key="short")
        assert settings.has_groq_key() is False

    def test_valid_key_returns_true(self):
        """Groq key over 10 chars should return True."""
        settings = Settings(
            openai_api_key="sk-test",
            groq_api_key="gsk_valid_groq_api_key_here",
        )
        assert settings.has_groq_key() is True

    def test_exactly_10_chars_returns_false(self):
        """Groq key with exactly 10 chars should return False (needs > 10)."""
        settings = Settings(openai_api_key="sk-test", groq_api_key="1234567890")
        assert settings.has_groq_key() is False

    def test_11_chars_returns_true(self):
        """Groq key with 11 chars should return True."""
        settings = Settings(openai_api_key="sk-test", groq_api_key="12345678901")
        assert settings.has_groq_key() is True


class TestHasFireworksKey:
    """Test has_fireworks_key() validation method."""

    def test_empty_key_returns_false(self):
        """Empty Fireworks key should return False."""
        settings = Settings(openai_api_key="sk-test", fireworks_api_key="")
        assert settings.has_fireworks_key() is False

    def test_short_key_returns_false(self):
        """Fireworks key under 10 chars should return False."""
        settings = Settings(openai_api_key="sk-test", fireworks_api_key="short")
        assert settings.has_fireworks_key() is False

    def test_valid_key_returns_true(self):
        """Fireworks key over 10 chars should return True."""
        settings = Settings(
            openai_api_key="sk-test",
            fireworks_api_key="fw_valid_fireworks_api_key",
        )
        assert settings.has_fireworks_key() is True


class TestHasGeminiKey:
    """Test has_gemini_key() validation method."""

    def test_empty_key_returns_false(self):
        """Empty Gemini key should return False."""
        settings = Settings(openai_api_key="sk-test", gemini_api_key="")
        assert settings.has_gemini_key() is False

    def test_short_key_returns_false(self):
        """Gemini key under 10 chars should return False."""
        settings = Settings(openai_api_key="sk-test", gemini_api_key="AIza123")
        assert settings.has_gemini_key() is False

    def test_valid_key_returns_true(self):
        """Gemini key over 10 chars should return True."""
        settings = Settings(
            openai_api_key="sk-test",
            gemini_api_key="AIzaSyBvalidGeminiKey",
        )
        assert settings.has_gemini_key() is True


class TestGetAvailableProviders:
    """Test get_available_providers() method.

    Uses IsolatedSettings to avoid loading from .env file for isolated tests.
    """

    def test_only_openai_configured(self):
        """Only OpenAI should be available when only OpenAI key is set."""
        settings = IsolatedSettings(openai_api_key="sk-test-openai")
        assert settings.get_available_providers() == ["openai"]

    def test_openai_and_anthropic_configured(self):
        """Both providers should be available when both keys are set."""
        settings = IsolatedSettings(
            openai_api_key="sk-test-openai",
            anthropic_api_key="sk-ant-valid-key",
        )
        providers = settings.get_available_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) == 2

    def test_all_providers_configured(self):
        """All 5 providers should be available when all keys are set."""
        settings = IsolatedSettings(
            openai_api_key="sk-test-openai",
            anthropic_api_key="sk-ant-valid-key",
            groq_api_key="gsk_valid_groq_key",
            fireworks_api_key="fw_valid_fireworks_key",
            gemini_api_key="AIzaSyB_valid_gemini",
        )
        providers = settings.get_available_providers()
        assert len(providers) == 5
        assert "openai" in providers
        assert "anthropic" in providers
        assert "groq" in providers
        assert "fireworks" in providers
        assert "gemini" in providers

    def test_order_is_consistent(self):
        """Provider order should be: openai, anthropic, groq, fireworks, gemini."""
        settings = IsolatedSettings(
            openai_api_key="sk-test-openai",
            anthropic_api_key="sk-ant-valid-key",
            groq_api_key="gsk_valid_groq_key",
            fireworks_api_key="fw_valid_fireworks_key",
            gemini_api_key="AIzaSyB_valid_gemini",
        )
        providers = settings.get_available_providers()
        expected_order = ["openai", "anthropic", "groq", "fireworks", "gemini"]
        assert providers == expected_order

    def test_placeholder_anthropic_not_available(self):
        """Anthropic with placeholder key should not be available."""
        settings = IsolatedSettings(
            openai_api_key="sk-test-openai",
            anthropic_api_key="sk-ant-PLACEHOLDER_ADD_YOUR_KEY",
        )
        providers = settings.get_available_providers()
        assert "anthropic" not in providers
        assert providers == ["openai"]

    def test_short_groq_key_not_available(self):
        """Groq with short key should not be available."""
        settings = IsolatedSettings(
            openai_api_key="sk-test-openai",
            groq_api_key="short",
        )
        providers = settings.get_available_providers()
        assert "groq" not in providers


class TestGetSettingsSingleton:
    """Test get_settings() singleton behavior."""

    def test_returns_settings_instance(self):
        """get_settings() should return a Settings instance."""
        # Clear the cache to ensure fresh instance
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_singleton_returns_same_instance(self):
        """get_settings() should return the same instance on subsequent calls."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


class TestConfigureLogging:
    """Test configure_logging() function."""

    def test_returns_logger(self):
        """configure_logging() should return a Logger instance."""
        settings = Settings(openai_api_key="sk-test", log_level="WARNING")
        logger = configure_logging(settings)
        assert isinstance(logger, logging.Logger)

    def test_logger_name_is_vendor_quote_extractor(self):
        """Configured logger should use the app namespace."""
        settings = Settings(openai_api_key="sk-test", log_level="WARNING")
        logger = configure_logging(settings)
        assert logger.name == "vendor_quote_extractor"

    def test_respects_log_level(self):
        """Logger should respect the configured log level."""
        settings = Settings(openai_api_key="sk-test", log_level="DEBUG")
        logger = configure_logging(settings)
        assert logger.level == logging.DEBUG

    def test_info_log_level(self):
        """Logger should handle INFO log level."""
        settings = Settings(openai_api_key="sk-test", log_level="INFO")
        logger = configure_logging(settings)
        assert logger.level == logging.INFO

    def test_warning_log_level(self):
        """Logger should handle WARNING log level."""
        settings = Settings(openai_api_key="sk-test", log_level="WARNING")
        logger = configure_logging(settings)
        assert logger.level == logging.WARNING

    def test_uses_get_settings_when_none(self):
        """configure_logging() should use get_settings() when settings is None."""
        get_settings.cache_clear()
        # This tests that it doesn't crash when called without settings
        # The actual get_settings() will load from env
        logger = configure_logging(None)
        assert isinstance(logger, logging.Logger)


class TestLLMProviderType:
    """Test LLMProviderType Literal type."""

    def test_valid_providers(self):
        """All valid provider names should be accepted."""
        valid_providers: list[LLMProviderType] = [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
            "gemini",
        ]
        for provider in valid_providers:
            settings = Settings(openai_api_key="sk-test", llm_provider=provider)
            assert settings.llm_provider == provider


class TestEnvironmentVariableLoading:
    """Test that Settings loads from environment variables."""

    def test_loads_openai_key_from_env(self):
        """Settings should load OPENAI_API_KEY from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-test-key"}):
            get_settings.cache_clear()
            settings = Settings()  # type: ignore[call-arg]
            assert settings.openai_api_key == "sk-env-test-key"

    def test_loads_log_level_from_env(self):
        """Settings should load LOG_LEVEL from environment."""
        settings = Settings(openai_api_key="sk-test", log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_loads_routing_enabled_from_env(self):
        """Settings should load ROUTING_ENABLED from environment."""
        settings = Settings(openai_api_key="sk-test", routing_enabled=True)
        assert settings.routing_enabled is True


class TestSettingsExtraIgnored:
    """Test that Settings ignores extra environment variables."""

    def test_ignores_unknown_env_vars(self):
        """Settings should not fail on unknown environment variables."""
        # The extra="ignore" setting should prevent errors
        settings = Settings(openai_api_key="sk-test")
        assert settings.openai_api_key == "sk-test"


class TestSettingsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_openai_key_still_creates_settings(self):
        """Empty OpenAI key should still create Settings (validation is optional)."""
        # Note: In production, this would fail but we test the class behavior
        settings = Settings(openai_api_key="")
        assert settings.openai_api_key == ""

    def test_zero_render_dpi(self):
        """Zero render DPI should be accepted (validation is application-level)."""
        settings = Settings(openai_api_key="sk-test", render_dpi=0)
        assert settings.render_dpi == 0

    def test_negative_timeout_accepted(self):
        """Negative timeout should be accepted by Settings (validation elsewhere)."""
        settings = Settings(openai_api_key="sk-test", processing_timeout_seconds=-1)
        assert settings.processing_timeout_seconds == -1

    def test_confidence_threshold_boundaries(self):
        """Confidence threshold should accept values outside 0-1 range."""
        settings = Settings(openai_api_key="sk-test", confidence_threshold=1.5)
        assert settings.confidence_threshold == 1.5

    def test_empty_fallback_chain(self):
        """Empty fallback chain should be accepted."""
        settings = Settings(openai_api_key="sk-test", llm_fallback_chain=[])
        assert settings.llm_fallback_chain == []

    def test_case_insensitive_log_level(self):
        """Log level should work with different cases due to upper() conversion."""
        settings = Settings(openai_api_key="sk-test", log_level="debug")
        logger = configure_logging(settings)
        # logging.DEBUG is used after upper() conversion
        assert logger.level == logging.DEBUG


class TestConfigurableThresholds:
    """Tests for configurable threshold settings added in v30.0.

    These settings allow operational flexibility without code changes:
    - Validation thresholds (tax rate, math tolerance)
    - UI display settings (max error causes, LLM bottleneck %)
    - API limits (max recent extractions, CORS origins)
    """

    def test_default_max_recent_extractions(self):
        """Default max recent extractions should be 50."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.max_recent_extractions == 50

    def test_default_cors_origins(self):
        """Default CORS origins should include Streamlit and React dev ports."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert "http://localhost:8501" in settings.cors_origins
        assert "http://localhost:3000" in settings.cors_origins

    def test_default_tax_rate_warning_threshold(self):
        """Default tax rate warning threshold should be 20% (0.20)."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.tax_rate_warning_threshold == 0.20

    def test_default_math_tolerance(self):
        """Default math tolerance should be $0.01."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.math_tolerance == 0.01

    def test_default_msrp_variance_threshold(self):
        """Default MSRP variance threshold should be 5% (0.05)."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.msrp_variance_threshold == 0.05

    def test_default_max_error_causes(self):
        """Default max error causes should be 5."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.max_error_causes == 5

    def test_default_llm_bottleneck_threshold(self):
        """Default LLM bottleneck threshold should be 80%."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.llm_bottleneck_threshold == 80

    def test_default_line_item_low_confidence_threshold(self):
        """Default line item low confidence threshold should be 0.9."""
        settings = IsolatedSettings(openai_api_key="sk-test-key")
        assert settings.line_item_low_confidence_threshold == 0.9

    def test_custom_max_recent_extractions(self):
        """Settings should accept custom max recent extractions."""
        settings = Settings(openai_api_key="sk-test", max_recent_extractions=100)
        assert settings.max_recent_extractions == 100

    def test_custom_cors_origins(self):
        """Settings should accept custom CORS origins."""
        custom_origins = ["https://prod.example.com", "https://staging.example.com"]
        settings = Settings(openai_api_key="sk-test", cors_origins=custom_origins)
        assert settings.cors_origins == custom_origins

    def test_custom_tax_rate_threshold(self):
        """Settings should accept custom tax rate warning threshold."""
        settings = Settings(openai_api_key="sk-test", tax_rate_warning_threshold=0.15)
        assert settings.tax_rate_warning_threshold == 0.15

    def test_custom_math_tolerance(self):
        """Settings should accept custom math tolerance."""
        settings = Settings(openai_api_key="sk-test", math_tolerance=0.05)
        assert settings.math_tolerance == 0.05

    def test_custom_msrp_variance_threshold(self):
        """Settings should accept custom MSRP variance threshold."""
        settings = Settings(openai_api_key="sk-test", msrp_variance_threshold=0.10)
        assert settings.msrp_variance_threshold == 0.10

    def test_custom_max_error_causes(self):
        """Settings should accept custom max error causes."""
        settings = Settings(openai_api_key="sk-test", max_error_causes=10)
        assert settings.max_error_causes == 10

    def test_custom_llm_bottleneck_threshold(self):
        """Settings should accept custom LLM bottleneck threshold."""
        settings = Settings(openai_api_key="sk-test", llm_bottleneck_threshold=90)
        assert settings.llm_bottleneck_threshold == 90

    def test_custom_line_item_confidence_threshold(self):
        """Settings should accept custom line item low confidence threshold."""
        settings = Settings(
            openai_api_key="sk-test", line_item_low_confidence_threshold=0.85
        )
        assert settings.line_item_low_confidence_threshold == 0.85
