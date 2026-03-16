"""
Tests for the LLM provider abstraction layer.

Tests cover:
- Provider configuration
- Provider registry
- Provider initialization
- get_provider() and get_providers() functions
"""

from unittest.mock import MagicMock, patch

import pytest

from src.extraction.providers import (
    PROVIDER_REGISTRY,
    AnthropicProvider,
    FireworksProvider,
    GeminiProvider,
    GroqProvider,
    OpenAIProvider,
    ProviderConfig,
    get_provider,
    get_provider_config,
    get_providers,
    is_provider_available,
)

# Test constants
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings with all providers configured."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-openai-key"
    settings.anthropic_api_key = "sk-ant-test-key"
    settings.groq_api_key = "gsk-test-groq-key"
    settings.fireworks_api_key = "fw-test-fireworks-key"
    settings.gemini_api_key = "gemini-test-key"
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-4o-2024-08-06"
    settings.anthropic_model = "claude-3-5-sonnet-20241022"
    settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    settings.fireworks_model = FIREWORKS_MODEL
    settings.gemini_model = "gemini-2.0-flash-lite"
    settings.has_anthropic_key.return_value = True
    settings.has_groq_key.return_value = True
    settings.has_fireworks_key.return_value = True
    settings.has_gemini_key.return_value = True
    return settings


@pytest.fixture
def mock_settings_openai_only():
    """Create mock settings with only OpenAI configured."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-openai-key"
    settings.anthropic_api_key = ""
    settings.groq_api_key = ""
    settings.fireworks_api_key = ""
    settings.gemini_api_key = ""
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-4o-2024-08-06"
    settings.anthropic_model = "claude-3-5-sonnet-20241022"
    settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    settings.fireworks_model = FIREWORKS_MODEL
    settings.gemini_model = "gemini-2.0-flash-lite"
    settings.has_anthropic_key.return_value = False
    settings.has_groq_key.return_value = False
    settings.has_fireworks_key.return_value = False
    settings.has_gemini_key.return_value = False
    return settings


# ============================================================================
# ProviderConfig Tests
# ============================================================================


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_config_with_required_fields(self):
        """Test creating config with required fields only."""
        config = ProviderConfig(
            api_key="test-key",
            model="test-model",
        )
        assert config.api_key == "test-key"
        assert config.model == "test-model"
        assert config.vision_model is None
        assert config.base_url is None
        assert config.max_retries == 2

    def test_config_with_all_fields(self):
        """Test creating config with all fields."""
        config = ProviderConfig(
            api_key="test-key",
            model="test-model",
            vision_model="vision-model",
            base_url="https://custom.api.com",
            max_retries=5,
        )
        assert config.api_key == "test-key"
        assert config.model == "test-model"
        assert config.vision_model == "vision-model"
        assert config.base_url == "https://custom.api.com"
        assert config.max_retries == 5


# ============================================================================
# Provider Registry Tests
# ============================================================================


class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_registry_has_openai(self):
        """Test OpenAI is registered."""
        assert "openai" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["openai"] == OpenAIProvider

    def test_registry_has_anthropic(self):
        """Test Anthropic is registered."""
        assert "anthropic" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["anthropic"] == AnthropicProvider

    def test_registry_has_groq(self):
        """Test Groq is registered."""
        assert "groq" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["groq"] == GroqProvider

    def test_registry_has_fireworks(self):
        """Test Fireworks is registered."""
        assert "fireworks" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["fireworks"] == FireworksProvider

    def test_registry_has_gemini(self):
        """Test Gemini is registered."""
        assert "gemini" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["gemini"] == GeminiProvider


# ============================================================================
# get_provider_config Tests
# ============================================================================


class TestGetProviderConfig:
    """Tests for get_provider_config function."""

    def test_openai_config(self, mock_settings):
        """Test getting OpenAI provider config from settings."""
        config = get_provider_config("openai", mock_settings)

        assert config.api_key == "sk-test-openai-key"
        assert config.model == "gpt-4o-2024-08-06"
        assert config.vision_model == "gpt-4o-2024-08-06"

    def test_anthropic_config(self, mock_settings):
        """Test getting Anthropic provider config from settings."""
        config = get_provider_config("anthropic", mock_settings)

        assert config.api_key == "sk-ant-test-key"
        assert config.model == "claude-3-5-sonnet-20241022"

    def test_groq_config(self, mock_settings):
        """Test getting Groq provider config from settings."""
        config = get_provider_config("groq", mock_settings)

        assert config.api_key == "gsk-test-groq-key"
        assert config.model == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_fireworks_config(self, mock_settings):
        """Test getting Fireworks provider config from settings."""
        config = get_provider_config("fireworks", mock_settings)

        assert config.api_key == "fw-test-fireworks-key"
        assert config.model == FIREWORKS_MODEL

    def test_gemini_config(self, mock_settings):
        """Test getting Gemini provider config from settings."""
        config = get_provider_config("gemini", mock_settings)

        assert config.api_key == "gemini-test-key"
        assert config.model == "gemini-2.0-flash-lite"

    def test_unknown_provider_raises(self, mock_settings):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            get_provider_config("nonexistent", mock_settings)


# ============================================================================
# get_provider Tests
# ============================================================================


class TestGetProvider:
    """Tests for get_provider function."""

    @patch("src.extraction.providers.openai_provider.instructor")
    @patch("src.extraction.providers.openai_provider.AsyncOpenAI")
    def test_get_openai_provider(
        self, mock_async_openai, mock_instructor, mock_settings
    ):
        """Test getting OpenAI provider instance."""
        mock_instructor.from_openai.return_value = MagicMock()

        provider = get_provider("openai", mock_settings)

        assert provider.name == "openai"
        assert provider.model == "gpt-4o-2024-08-06"

    @patch("src.extraction.providers.anthropic_provider.instructor")
    @patch("src.extraction.providers.anthropic_provider.AsyncAnthropic")
    def test_get_anthropic_provider(
        self, mock_async_anthropic, mock_instructor, mock_settings
    ):
        """Test getting Anthropic provider instance."""
        mock_instructor.from_anthropic.return_value = MagicMock()

        provider = get_provider("anthropic", mock_settings)

        assert provider.name == "anthropic"
        assert provider.model == "claude-3-5-sonnet-20241022"

    @patch("src.extraction.providers.groq_provider.instructor")
    @patch("src.extraction.providers.groq_provider.AsyncGroq")
    def test_get_groq_provider(self, mock_async_groq, mock_instructor, mock_settings):
        """Test getting Groq provider instance."""
        mock_instructor.from_groq.return_value = MagicMock()

        provider = get_provider("groq", mock_settings)

        assert provider.name == "groq"
        assert provider.model == "meta-llama/llama-4-scout-17b-16e-instruct"

    @patch("src.extraction.providers.fireworks_provider.instructor")
    @patch("src.extraction.providers.fireworks_provider.AsyncFireworks")
    def test_get_fireworks_provider(
        self, mock_async_fireworks, mock_instructor, mock_settings
    ):
        """Test getting Fireworks provider instance."""
        mock_instructor.from_fireworks.return_value = MagicMock()

        provider = get_provider("fireworks", mock_settings)

        assert provider.name == "fireworks"
        assert provider.model == FIREWORKS_MODEL

    @patch("src.extraction.providers.gemini_provider.instructor")
    def test_get_gemini_provider(self, mock_instructor, mock_settings):
        """Test getting Gemini provider instance."""
        mock_instructor.from_provider.return_value = MagicMock()

        provider = get_provider("gemini", mock_settings)

        assert provider.name == "gemini"
        assert provider.model == "gemini-2.0-flash-lite"

    def test_get_unknown_provider_raises(self, mock_settings):
        """Test getting unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            get_provider("nonexistent", mock_settings)


# ============================================================================
# get_providers Tests
# ============================================================================


class TestGetProviders:
    """Tests for get_providers function."""

    @patch("src.extraction.providers.openai_provider.instructor")
    @patch("src.extraction.providers.openai_provider.AsyncOpenAI")
    @patch("src.extraction.providers.anthropic_provider.instructor")
    @patch("src.extraction.providers.anthropic_provider.AsyncAnthropic")
    def test_get_providers_with_both_configured(
        self,
        mock_async_anthropic,
        mock_anthropic_instructor,
        mock_async_openai,
        mock_openai_instructor,
        mock_settings,
    ):
        """Test getting all providers when both are configured."""
        mock_openai_instructor.from_openai.return_value = MagicMock()
        mock_anthropic_instructor.from_anthropic.return_value = MagicMock()

        providers = get_providers(mock_settings)

        # Primary provider first
        assert len(providers) >= 1
        assert providers[0].name == "openai"

        # Anthropic as fallback
        if len(providers) > 1:
            assert providers[1].name == "anthropic"

    @patch("src.extraction.providers.openai_provider.instructor")
    @patch("src.extraction.providers.openai_provider.AsyncOpenAI")
    def test_get_providers_openai_only(
        self,
        mock_async_openai,
        mock_instructor,
        mock_settings_openai_only,
    ):
        """Test getting providers with only OpenAI configured."""
        mock_instructor.from_openai.return_value = MagicMock()

        providers = get_providers(mock_settings_openai_only)

        assert len(providers) == 1
        assert providers[0].name == "openai"


# ============================================================================
# is_provider_available Tests
# ============================================================================


class TestIsProviderAvailable:
    """Tests for is_provider_available function."""

    def test_openai_available_with_key(self, mock_settings):
        """Test OpenAI is available when API key is set."""
        assert is_provider_available("openai", mock_settings) is True

    def test_anthropic_available_with_key(self, mock_settings):
        """Test Anthropic is available when API key is set."""
        assert is_provider_available("anthropic", mock_settings) is True

    def test_groq_available_with_key(self, mock_settings):
        """Test Groq is available when API key is set."""
        assert is_provider_available("groq", mock_settings) is True

    def test_fireworks_available_with_key(self, mock_settings):
        """Test Fireworks is available when API key is set."""
        assert is_provider_available("fireworks", mock_settings) is True

    def test_gemini_available_with_key(self, mock_settings):
        """Test Gemini is available when API key is set."""
        assert is_provider_available("gemini", mock_settings) is True

    def test_anthropic_not_available_without_key(self, mock_settings_openai_only):
        """Test Anthropic not available without API key."""
        assert is_provider_available("anthropic", mock_settings_openai_only) is False

    def test_fireworks_not_available_without_key(self, mock_settings_openai_only):
        """Test Fireworks not available without API key."""
        assert is_provider_available("fireworks", mock_settings_openai_only) is False

    def test_gemini_not_available_without_key(self, mock_settings_openai_only):
        """Test Gemini not available without API key."""
        assert is_provider_available("gemini", mock_settings_openai_only) is False

    def test_unknown_provider_not_available(self, mock_settings):
        """Test unknown provider is not available."""
        assert is_provider_available("nonexistent", mock_settings) is False


# ============================================================================
# OpenAIProvider Tests
# ============================================================================


class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    @patch("src.extraction.providers.openai_provider.instructor")
    @patch("src.extraction.providers.openai_provider.AsyncOpenAI")
    def test_provider_name(self, mock_async_openai, mock_instructor):
        """Test provider name property."""
        mock_instructor.from_openai.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gpt-4o")

        provider = OpenAIProvider(config)

        assert provider.name == "openai"

    @patch("src.extraction.providers.openai_provider.instructor")
    @patch("src.extraction.providers.openai_provider.AsyncOpenAI")
    def test_provider_model(self, mock_async_openai, mock_instructor):
        """Test provider model property."""
        mock_instructor.from_openai.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gpt-4o")

        provider = OpenAIProvider(config)

        assert provider.model == "gpt-4o"

    @patch("src.extraction.providers.openai_provider.instructor")
    @patch("src.extraction.providers.openai_provider.AsyncOpenAI")
    def test_provider_vision_model_fallback(self, mock_async_openai, mock_instructor):
        """Test vision model falls back to main model."""
        mock_instructor.from_openai.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gpt-4o")

        provider = OpenAIProvider(config)

        assert provider.vision_model == "gpt-4o"

    def test_provider_raises_without_api_key(self):
        """Test provider raises error without API key."""
        config = ProviderConfig(api_key="", model="gpt-4o")

        with pytest.raises(ValueError, match="API key not configured"):
            OpenAIProvider(config)


# ============================================================================
# AnthropicProvider Tests
# ============================================================================


class TestAnthropicProvider:
    """Tests for AnthropicProvider class."""

    @patch("src.extraction.providers.anthropic_provider.instructor")
    @patch("src.extraction.providers.anthropic_provider.AsyncAnthropic")
    def test_provider_name(self, mock_async_anthropic, mock_instructor):
        """Test provider name property."""
        mock_instructor.from_anthropic.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="claude-3-5-sonnet")

        provider = AnthropicProvider(config)

        assert provider.name == "anthropic"

    @patch("src.extraction.providers.anthropic_provider.instructor")
    @patch("src.extraction.providers.anthropic_provider.AsyncAnthropic")
    def test_provider_model(self, mock_async_anthropic, mock_instructor):
        """Test provider model property."""
        mock_instructor.from_anthropic.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="claude-3-5-sonnet")

        provider = AnthropicProvider(config)

        assert provider.model == "claude-3-5-sonnet"

    def test_provider_raises_without_api_key(self):
        """Test provider raises error without API key."""
        config = ProviderConfig(api_key="", model="claude-3-5-sonnet")

        with pytest.raises(ValueError, match="API key not configured"):
            AnthropicProvider(config)


# ============================================================================
# GroqProvider Tests
# ============================================================================


class TestGroqProvider:
    """Tests for GroqProvider class."""

    @patch("src.extraction.providers.groq_provider.instructor")
    @patch("src.extraction.providers.groq_provider.AsyncGroq")
    def test_provider_name(self, mock_async_groq, mock_instructor):
        """Test provider name property."""
        mock_instructor.from_groq.return_value = MagicMock()
        config = ProviderConfig(
            api_key="test-key", model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

        provider = GroqProvider(config)

        assert provider.name == "groq"

    @patch("src.extraction.providers.groq_provider.instructor")
    @patch("src.extraction.providers.groq_provider.AsyncGroq")
    def test_provider_model(self, mock_async_groq, mock_instructor):
        """Test provider model property."""
        mock_instructor.from_groq.return_value = MagicMock()
        config = ProviderConfig(
            api_key="test-key", model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

        provider = GroqProvider(config)

        assert provider.model == "meta-llama/llama-4-scout-17b-16e-instruct"

    @patch("src.extraction.providers.groq_provider.instructor")
    @patch("src.extraction.providers.groq_provider.AsyncGroq")
    def test_provider_vision_model_fallback(self, mock_async_groq, mock_instructor):
        """Test vision model falls back to main model."""
        mock_instructor.from_groq.return_value = MagicMock()
        config = ProviderConfig(
            api_key="test-key", model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

        provider = GroqProvider(config)

        assert provider.vision_model == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_provider_raises_without_api_key(self):
        """Test provider raises error without API key."""
        config = ProviderConfig(
            api_key="", model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

        with pytest.raises(ValueError, match="API key not configured"):
            GroqProvider(config)

    @patch("src.extraction.providers.groq_provider.instructor")
    @patch("src.extraction.providers.groq_provider.AsyncGroq")
    @pytest.mark.asyncio
    async def test_vision_rejects_too_many_images(
        self, mock_async_groq, mock_instructor
    ):
        """Test vision extraction rejects more than 5 images."""
        mock_instructor.from_groq.return_value = MagicMock()
        config = ProviderConfig(
            api_key="test-key", model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        provider = GroqProvider(config)

        # Create 6 fake images (exceeds Groq's 5-image limit)
        images = [b"fake-image-bytes"] * 6

        with pytest.raises(ValueError, match="maximum 5 images"):
            await provider.extract_vision(images, "Extract data")

    @patch("src.extraction.providers.groq_provider.instructor")
    @patch("src.extraction.providers.groq_provider.AsyncGroq")
    def test_provider_is_available(self, mock_async_groq, mock_instructor):
        """Test provider is available with valid config."""
        mock_instructor.from_groq.return_value = MagicMock()
        config = ProviderConfig(
            api_key="test-key", model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

        provider = GroqProvider(config)

        assert provider.is_available is True


# ============================================================================
# FireworksProvider Tests
# ============================================================================


class TestFireworksProvider:
    """Tests for FireworksProvider class."""

    @patch("src.extraction.providers.fireworks_provider.instructor")
    @patch("src.extraction.providers.fireworks_provider.AsyncFireworks")
    def test_provider_name(self, mock_async_fireworks, mock_instructor):
        """Test provider name property."""
        mock_instructor.from_fireworks.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model=FIREWORKS_MODEL)

        provider = FireworksProvider(config)

        assert provider.name == "fireworks"

    @patch("src.extraction.providers.fireworks_provider.instructor")
    @patch("src.extraction.providers.fireworks_provider.AsyncFireworks")
    def test_provider_model(self, mock_async_fireworks, mock_instructor):
        """Test provider model property."""
        mock_instructor.from_fireworks.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model=FIREWORKS_MODEL)

        provider = FireworksProvider(config)

        assert provider.model == FIREWORKS_MODEL

    @patch("src.extraction.providers.fireworks_provider.instructor")
    @patch("src.extraction.providers.fireworks_provider.AsyncFireworks")
    def test_provider_vision_model_fallback(
        self, mock_async_fireworks, mock_instructor
    ):
        """Test vision model falls back to main model."""
        mock_instructor.from_fireworks.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model=FIREWORKS_MODEL)

        provider = FireworksProvider(config)

        assert provider.vision_model == FIREWORKS_MODEL

    def test_provider_raises_without_api_key(self):
        """Test provider raises error without API key."""
        config = ProviderConfig(api_key="", model=FIREWORKS_MODEL)

        with pytest.raises(ValueError, match="API key not configured"):
            FireworksProvider(config)

    @patch("src.extraction.providers.fireworks_provider.instructor")
    @patch("src.extraction.providers.fireworks_provider.AsyncFireworks")
    @pytest.mark.asyncio
    async def test_vision_rejects_too_many_images(
        self, mock_async_fireworks, mock_instructor
    ):
        """Test vision extraction rejects more than 30 images."""
        mock_instructor.from_fireworks.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model=FIREWORKS_MODEL)
        provider = FireworksProvider(config)

        # Create 31 fake images (exceeds Fireworks' 30-image limit)
        images = [b"fake-image-bytes"] * 31

        with pytest.raises(ValueError, match="maximum 30 images"):
            await provider.extract_vision(images, "Extract data")

    @patch("src.extraction.providers.fireworks_provider.instructor")
    @patch("src.extraction.providers.fireworks_provider.AsyncFireworks")
    def test_provider_is_available(self, mock_async_fireworks, mock_instructor):
        """Test provider is available with valid config."""
        mock_instructor.from_fireworks.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model=FIREWORKS_MODEL)

        provider = FireworksProvider(config)

        assert provider.is_available is True


# ============================================================================
# GeminiProvider Tests
# ============================================================================


class TestGeminiProvider:
    """Tests for GeminiProvider class.

    Updated for google-genai SDK migration (v46.5):
    - Uses instructor.from_provider() instead of instructor.from_gemini()
    - No longer requires mocking genai module
    """

    @patch("src.extraction.providers.gemini_provider.instructor")
    def test_provider_name(self, mock_instructor):
        """Test provider name property."""
        mock_instructor.from_provider.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gemini-2.0-flash-lite")

        provider = GeminiProvider(config)

        assert provider.name == "gemini"

    @patch("src.extraction.providers.gemini_provider.instructor")
    def test_provider_model(self, mock_instructor):
        """Test provider model property."""
        mock_instructor.from_provider.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gemini-2.0-flash-lite")

        provider = GeminiProvider(config)

        assert provider.model == "gemini-2.0-flash-lite"

    @patch("src.extraction.providers.gemini_provider.instructor")
    def test_provider_vision_model_fallback(self, mock_instructor):
        """Test vision model falls back to main model."""
        mock_instructor.from_provider.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gemini-2.0-flash-lite")

        provider = GeminiProvider(config)

        assert provider.vision_model == "gemini-2.0-flash-lite"

    def test_provider_raises_without_api_key(self):
        """Test provider raises error without API key."""
        config = ProviderConfig(api_key="", model="gemini-2.0-flash-lite")

        with pytest.raises(ValueError, match="API key not configured"):
            GeminiProvider(config)

    @patch("src.extraction.providers.gemini_provider.instructor")
    def test_provider_is_available(self, mock_instructor):
        """Test provider is available with valid config."""
        mock_instructor.from_provider.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gemini-2.0-flash-lite")

        provider = GeminiProvider(config)

        assert provider.is_available is True

    @patch("src.extraction.providers.gemini_provider.instructor")
    def test_provider_uses_async_mode(self, mock_instructor):
        """
        Test provider initializes instructor with async_client=True.

        Without async_client=True, instructor.from_provider() returns a
        synchronous client that would block the event loop when awaited.
        This test ensures the async_client=True parameter is always passed.
        """
        mock_instructor.from_provider.return_value = MagicMock()
        config = ProviderConfig(api_key="test-key", model="gemini-2.0-flash-lite")

        GeminiProvider(config)

        # Verify from_provider was called with async_client=True
        mock_instructor.from_provider.assert_called_once()
        call_kwargs = mock_instructor.from_provider.call_args.kwargs
        assert call_kwargs.get("async_client") is True, (
            "instructor.from_provider must be called with async_client=True "
            "to avoid blocking the event loop in async contexts"
        )
