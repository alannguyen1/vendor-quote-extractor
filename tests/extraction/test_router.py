"""
Tests for the Smart Router module.

Tests cover:
- Routing decisions for different document types
- Routing strategies (speed, accuracy, cost)
- Provider availability handling
- Fallback chain generation
- Router singleton management
"""

from unittest.mock import MagicMock, patch

import pytest

from src.extraction.pdf_parser import ParseResult, ParseTimingBreakdown
from src.extraction.router import (
    RoutingDecision,
    SmartRouter,
    get_router,
    reset_router,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_settings_all_providers():
    """Create mock settings with all providers configured."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-openai-key"
    settings.anthropic_api_key = "sk-ant-test-key"
    settings.groq_api_key = "gsk-test-groq-key"
    settings.fireworks_api_key = "fw-test-key"
    settings.gemini_api_key = "gm-test-key"
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-4o-2024-08-06"
    settings.anthropic_model = "claude-3-5-sonnet-20241022"
    settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    settings.fireworks_model = "accounts/fireworks/models/llama-v3p3-70b-instruct"
    settings.gemini_model = "gemini-2.0-flash-lite"
    settings.llm_fallback_chain = ["openai", "anthropic"]
    settings.routing_enabled = True
    settings.routing_strategy = "speed"
    settings.complexity_routing_threshold = 40  # H6: Route to Gemini if >40 table rows
    settings.has_anthropic_key.return_value = True
    settings.has_groq_key.return_value = True
    settings.has_fireworks_key.return_value = True
    settings.has_gemini_key.return_value = True
    settings.get_available_providers.return_value = [
        "openai",
        "anthropic",
        "groq",
        "fireworks",
        "gemini",
    ]
    return settings


@pytest.fixture
def mock_settings_groq_only():
    """Create mock settings with only Groq configured."""
    settings = MagicMock()
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.groq_api_key = "gsk-test-groq-key"
    settings.fireworks_api_key = ""
    settings.gemini_api_key = ""
    settings.llm_provider = "groq"
    settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    settings.llm_fallback_chain = ["groq"]
    settings.routing_enabled = True
    settings.routing_strategy = "speed"
    settings.complexity_routing_threshold = 40  # H6
    settings.has_anthropic_key.return_value = False
    settings.has_groq_key.return_value = True
    settings.has_fireworks_key.return_value = False
    settings.has_gemini_key.return_value = False
    settings.get_available_providers.return_value = ["groq"]
    return settings


@pytest.fixture
def mock_settings_openai_anthropic():
    """Create mock settings with OpenAI and Anthropic only."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-openai-key"
    settings.anthropic_api_key = "sk-ant-test-key"
    settings.groq_api_key = ""
    settings.fireworks_api_key = ""
    settings.gemini_api_key = ""
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-4o-2024-08-06"
    settings.anthropic_model = "claude-3-5-sonnet-20241022"
    settings.llm_fallback_chain = ["openai", "anthropic"]
    settings.routing_enabled = True
    settings.routing_strategy = "speed"
    settings.complexity_routing_threshold = 40  # H6
    settings.has_anthropic_key.return_value = True
    settings.has_groq_key.return_value = False
    settings.has_fireworks_key.return_value = False
    settings.has_gemini_key.return_value = False
    settings.get_available_providers.return_value = ["openai", "anthropic"]
    return settings


@pytest.fixture
def mock_settings_routing_disabled():
    """Create mock settings with routing disabled."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-openai-key"
    settings.anthropic_api_key = "sk-ant-test-key"
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-4o-2024-08-06"
    settings.llm_fallback_chain = ["openai", "anthropic"]
    settings.routing_enabled = False
    settings.routing_strategy = "speed"
    settings.complexity_routing_threshold = 40  # H6
    settings.has_anthropic_key.return_value = True
    settings.has_groq_key.return_value = False
    settings.has_fireworks_key.return_value = False
    settings.has_gemini_key.return_value = False
    settings.get_available_providers.return_value = ["openai", "anthropic"]
    return settings


@pytest.fixture
def text_pdf_small():
    """Create ParseResult for small text PDF (5 pages)."""
    return ParseResult(
        text="Test content " * 1000,
        tables=[],
        page_count=5,
        is_scanned=False,
        images=None,
        error=None,
        timing=ParseTimingBreakdown(
            text_extract_ms=100,
            table_extract_ms=50,
        ),
    )


@pytest.fixture
def text_pdf_complex():
    """Create ParseResult for text PDF with many table rows (H6 complexity test)."""
    from src.extraction.pdf_parser import TableData

    # Create a table with 50 rows (exceeds threshold of 40)
    large_table = TableData(
        page=1,
        rows=[["Item", "Qty", "Price"]]
        + [[f"Item {i}", "1", "100.00"] for i in range(50)],
        headers=["Item", "Qty", "Price"],
    )
    return ParseResult(
        text="Test content " * 500,
        tables=[large_table],
        page_count=3,  # Small page count but complex due to table rows
        is_scanned=False,
        images=None,
        error=None,
        timing=ParseTimingBreakdown(
            text_extract_ms=100,
            table_extract_ms=150,
        ),
    )


@pytest.fixture
def text_pdf_simple_with_table():
    """Create ParseResult for text PDF with small table (below complexity threshold)."""
    from src.extraction.pdf_parser import TableData

    # Create a table with 20 rows (below threshold of 40)
    small_table = TableData(
        page=1,
        rows=[["Item", "Qty", "Price"]]
        + [[f"Item {i}", "1", "100.00"] for i in range(20)],
        headers=["Item", "Qty", "Price"],
    )
    return ParseResult(
        text="Test content " * 500,
        tables=[small_table],
        page_count=3,
        is_scanned=False,
        images=None,
        error=None,
        timing=ParseTimingBreakdown(
            text_extract_ms=100,
            table_extract_ms=80,
        ),
    )


@pytest.fixture
def text_pdf_large():
    """Create ParseResult for large text PDF (20 pages)."""
    return ParseResult(
        text="Test content " * 5000,
        tables=[],
        page_count=20,
        is_scanned=False,
        images=None,
        error=None,
        timing=ParseTimingBreakdown(
            text_extract_ms=200,
            table_extract_ms=100,
        ),
    )


@pytest.fixture
def scanned_pdf_small():
    """Create ParseResult for small scanned PDF (3 pages)."""
    return ParseResult(
        text="",
        tables=[],
        page_count=3,
        is_scanned=True,
        images=[b"image1", b"image2", b"image3"],
        error=None,
        timing=ParseTimingBreakdown(
            text_extract_ms=50,
            table_extract_ms=20,
            image_render_ms=500,
        ),
    )


@pytest.fixture
def scanned_pdf_medium():
    """Create ParseResult for medium scanned PDF (8 pages, exceeds Groq limit)."""
    return ParseResult(
        text="",
        tables=[],
        page_count=8,
        is_scanned=True,
        images=[b"image"] * 8,
        error=None,
        timing=ParseTimingBreakdown(
            text_extract_ms=50,
            table_extract_ms=20,
            image_render_ms=1000,
        ),
    )


@pytest.fixture
def scanned_pdf_large():
    """Create ParseResult for large scanned PDF (40 pages, exceeds Fireworks limit)."""
    return ParseResult(
        text="",
        tables=[],
        page_count=40,
        is_scanned=True,
        images=[b"image"] * 40,
        error=None,
        timing=ParseTimingBreakdown(
            text_extract_ms=50,
            table_extract_ms=20,
            image_render_ms=4000,
        ),
    )


@pytest.fixture(autouse=True)
def reset_router_singleton():
    """Reset router singleton before each test."""
    reset_router()
    yield
    reset_router()


# ============================================================================
# RoutingDecision Tests
# ============================================================================


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_decision_creation(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            provider_name="groq",
            reason="Test reason",
            fallback_chain=["openai", "anthropic"],
            expected_latency_ms=1500,
            is_optimal=True,
        )

        assert decision.provider_name == "groq"
        assert decision.reason == "Test reason"
        assert decision.fallback_chain == ["openai", "anthropic"]
        assert decision.expected_latency_ms == 1500
        assert decision.is_optimal is True

    def test_decision_default_optimal(self):
        """Test is_optimal defaults to True."""
        decision = RoutingDecision(
            provider_name="openai",
            reason="Test",
            fallback_chain=[],
            expected_latency_ms=4000,
        )

        assert decision.is_optimal is True


# ============================================================================
# SmartRouter Initialization Tests
# ============================================================================


class TestSmartRouterInit:
    """Tests for SmartRouter initialization."""

    def test_router_init_with_settings(self, mock_settings_all_providers):
        """Test router initialization with settings."""
        router = SmartRouter(mock_settings_all_providers)

        assert router.settings == mock_settings_all_providers

    def test_router_init_uses_get_settings(self):
        """Test router uses get_settings when no settings provided."""
        with patch("src.extraction.router.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_get_settings.return_value = mock_settings

            router = SmartRouter()

            mock_get_settings.assert_called_once()
            assert router.settings == mock_settings


# ============================================================================
# Routing Disabled Tests
# ============================================================================


class TestRoutingDisabled:
    """Tests for routing when disabled."""

    def test_routing_disabled_uses_default_chain(
        self, mock_settings_routing_disabled, text_pdf_small
    ):
        """Test disabled routing uses default fallback chain."""
        router = SmartRouter(mock_settings_routing_disabled)

        decision = router.select_provider(text_pdf_small)

        assert decision.provider_name == "openai"
        assert decision.is_optimal is False
        assert "disabled" in decision.reason.lower()


# ============================================================================
# Speed Strategy Tests - Text PDFs
# ============================================================================


class TestSpeedStrategyTextPDF:
    """Tests for speed routing strategy with text PDFs."""

    @patch("src.extraction.router.is_provider_available")
    def test_small_text_pdf_routes_to_groq(
        self, mock_available, mock_settings_all_providers, text_pdf_small
    ):
        """Test small text PDF routes to Groq for speed."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_small, strategy="speed")

        assert decision.provider_name == "groq"
        assert decision.is_optimal is True
        assert decision.expected_latency_ms == 1500

    @patch("src.extraction.router.is_provider_available")
    def test_large_text_pdf_routes_to_gemini(
        self, mock_available, mock_settings_all_providers, text_pdf_large
    ):
        """Test large text PDF routes to Gemini for large context."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_large, strategy="speed")

        assert decision.provider_name == "gemini"
        assert decision.is_optimal is True

    @patch("src.extraction.router.is_provider_available")
    def test_small_text_pdf_falls_back_to_openai_without_groq(
        self, mock_available, mock_settings_openai_anthropic, text_pdf_small
    ):
        """Test small text PDF falls back to OpenAI when Groq unavailable."""
        mock_available.side_effect = lambda p, s: p in ["openai", "anthropic"]
        router = SmartRouter(mock_settings_openai_anthropic)

        decision = router.select_provider(text_pdf_small, strategy="speed")

        assert decision.provider_name == "openai"
        assert decision.is_optimal is False
        assert "Groq unavailable" in decision.reason

    @patch("src.extraction.router.is_provider_available")
    def test_large_text_pdf_uses_groq_without_gemini(
        self, mock_available, mock_settings_groq_only, text_pdf_large
    ):
        """Test large text PDF uses Groq when Gemini unavailable."""
        mock_available.side_effect = lambda p, s: p == "groq"
        router = SmartRouter(mock_settings_groq_only)

        decision = router.select_provider(text_pdf_large, strategy="speed")

        assert decision.provider_name == "groq"
        assert decision.is_optimal is False
        assert "Gemini unavailable" in decision.reason


# ============================================================================
# Speed Strategy Tests - Scanned PDFs
# ============================================================================


class TestSpeedStrategyScannedPDF:
    """Tests for speed routing strategy with scanned PDFs."""

    @patch("src.extraction.router.is_provider_available")
    def test_small_scanned_pdf_routes_to_groq(
        self, mock_available, mock_settings_all_providers, scanned_pdf_small
    ):
        """Test small scanned PDF (within 5 page limit) routes to Groq."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(scanned_pdf_small, strategy="speed")

        assert decision.provider_name == "groq"
        assert decision.is_optimal is True
        assert decision.expected_latency_ms == 2500

    @patch("src.extraction.router.is_provider_available")
    def test_medium_scanned_pdf_routes_to_fireworks(
        self, mock_available, mock_settings_all_providers, scanned_pdf_medium
    ):
        """Test medium scanned PDF (6+ pages) routes to Fireworks."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(scanned_pdf_medium, strategy="speed")

        assert decision.provider_name == "fireworks"
        assert decision.is_optimal is True
        assert "exceeds Groq" in decision.reason

    @patch("src.extraction.router.is_provider_available")
    def test_large_scanned_pdf_routes_to_openai(
        self, mock_available, mock_settings_all_providers, scanned_pdf_large
    ):
        """Test large scanned PDF (31+ pages) routes to OpenAI."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(scanned_pdf_large, strategy="speed")

        assert decision.provider_name == "openai"
        assert decision.is_optimal is True
        assert "most reliable" in decision.reason


# ============================================================================
# Accuracy Strategy Tests
# ============================================================================


class TestAccuracyStrategy:
    """Tests for accuracy routing strategy."""

    @patch("src.extraction.router.is_provider_available")
    def test_accuracy_prefers_openai(
        self, mock_available, mock_settings_all_providers, text_pdf_small
    ):
        """Test accuracy strategy prefers OpenAI GPT-4o."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_small, strategy="accuracy")

        assert decision.provider_name == "openai"
        assert decision.is_optimal is True
        assert "GPT-4o" in decision.reason or "accuracy" in decision.reason.lower()

    @patch("src.extraction.router.is_provider_available")
    def test_accuracy_falls_back_to_anthropic(
        self, mock_available, mock_settings_all_providers, text_pdf_small
    ):
        """Test accuracy strategy falls back to Anthropic without OpenAI."""
        mock_available.side_effect = lambda p, s: p in ["anthropic", "groq"]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_small, strategy="accuracy")

        assert decision.provider_name == "anthropic"
        assert decision.is_optimal is True
        assert "Claude" in decision.reason or "fallback" in decision.reason.lower()


# ============================================================================
# Cost Strategy Tests
# ============================================================================


class TestCostStrategy:
    """Tests for cost routing strategy."""

    @patch("src.extraction.router.is_provider_available")
    def test_cost_prefers_groq(
        self, mock_available, mock_settings_all_providers, text_pdf_small
    ):
        """Test cost strategy prefers Groq (cheapest)."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_small, strategy="cost")

        assert decision.provider_name == "groq"
        assert decision.is_optimal is True
        assert "cheapest" in decision.reason.lower()

    @patch("src.extraction.router.is_provider_available")
    def test_cost_skips_groq_for_large_scans(
        self, mock_available, mock_settings_all_providers, scanned_pdf_medium
    ):
        """Test cost strategy skips Groq for scanned PDFs exceeding limit."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "gemini",
            "fireworks",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(scanned_pdf_medium, strategy="cost")

        # Should skip Groq (5-image limit) and use next cheapest
        assert decision.provider_name != "groq"


# ============================================================================
# Fallback Chain Tests
# ============================================================================


class TestFallbackChain:
    """Tests for fallback chain generation."""

    @patch("src.extraction.router.is_provider_available")
    def test_fallback_chain_excludes_primary(
        self, mock_available, mock_settings_all_providers, text_pdf_small
    ):
        """Test fallback chain excludes the primary provider."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_small, strategy="speed")

        # Primary is groq, so groq should not be in fallback
        assert decision.provider_name == "groq"
        assert "groq" not in decision.fallback_chain

    @patch("src.extraction.router.is_provider_available")
    def test_fallback_chain_includes_available_providers(
        self, mock_available, mock_settings_all_providers, text_pdf_small
    ):
        """Test fallback chain includes configured providers."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_small, strategy="speed")

        # Fallback chain should include available providers
        has_openai = "openai" in decision.fallback_chain
        has_anthropic = "anthropic" in decision.fallback_chain
        assert has_openai or has_anthropic


# ============================================================================
# Provider Instance Generation Tests
# ============================================================================


class TestGetProvidersForDecision:
    """Tests for get_providers_for_decision method."""

    @patch("src.extraction.router.get_provider")
    def test_returns_providers_in_order(
        self, mock_get_provider, mock_settings_all_providers
    ):
        """Test providers are returned in correct order."""
        # Mock provider instances
        mock_primary = MagicMock()
        mock_primary.name = "groq"
        mock_fallback = MagicMock()
        mock_fallback.name = "openai"

        mock_get_provider.side_effect = lambda name, settings: (
            mock_primary if name == "groq" else mock_fallback
        )

        router = SmartRouter(mock_settings_all_providers)
        decision = RoutingDecision(
            provider_name="groq",
            reason="Test",
            fallback_chain=["openai"],
            expected_latency_ms=1500,
        )

        providers = router.get_providers_for_decision(decision)

        assert len(providers) == 2
        assert providers[0].name == "groq"
        assert providers[1].name == "openai"

    @patch("src.extraction.router.get_provider")
    def test_handles_provider_creation_failure(
        self, mock_get_provider, mock_settings_all_providers
    ):
        """Test graceful handling of provider creation failures."""
        # Mock provider that fails for fallback
        mock_primary = MagicMock()
        mock_primary.name = "groq"

        def get_provider_side_effect(name, settings):
            if name == "groq":
                return mock_primary
            raise ValueError(f"Unknown provider: {name}")

        mock_get_provider.side_effect = get_provider_side_effect

        router = SmartRouter(mock_settings_all_providers)
        decision = RoutingDecision(
            provider_name="groq",
            reason="Test",
            fallback_chain=["unknown_provider"],
            expected_latency_ms=1500,
        )

        providers = router.get_providers_for_decision(decision)

        # Should still return primary, skip failed fallback
        assert len(providers) == 1
        assert providers[0].name == "groq"


# ============================================================================
# Router Singleton Tests
# ============================================================================


class TestRouterSingleton:
    """Tests for router singleton management."""

    def test_get_router_creates_instance(self):
        """Test get_router creates singleton instance."""
        router1 = get_router()
        router2 = get_router()

        assert router1 is router2

    def test_reset_router_clears_singleton(self):
        """Test reset_router clears the singleton."""
        router1 = get_router()
        reset_router()
        router2 = get_router()

        assert router1 is not router2

    def test_get_router_with_settings(self):
        """Test get_router accepts settings parameter."""
        mock_settings = MagicMock()
        mock_settings.routing_enabled = False
        mock_settings.routing_strategy = "speed"
        mock_settings.llm_fallback_chain = ["openai"]

        router = get_router(mock_settings)

        assert router.settings == mock_settings


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("src.extraction.router.is_provider_available")
    def test_no_providers_available(
        self, mock_available, mock_settings_all_providers, text_pdf_small
    ):
        """Test handling when no providers are available."""
        mock_available.return_value = False
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_small, strategy="speed")

        # Should still return a decision (will fail later)
        assert decision.provider_name == "openai"
        assert decision.is_optimal is False

    def test_zero_page_document(self, mock_settings_all_providers):
        """Test handling of zero-page document."""
        zero_page_result = ParseResult(
            text="",
            tables=[],
            page_count=0,
            is_scanned=False,
            images=None,
            error=None,
        )

        router = SmartRouter(mock_settings_all_providers)
        decision = router.select_provider(zero_page_result, strategy="speed")

        # Should handle gracefully
        assert decision.provider_name is not None

    @patch("src.extraction.router.is_provider_available")
    def test_exactly_at_groq_limit(self, mock_available, mock_settings_all_providers):
        """Test scanned PDF with exactly 5 pages (at Groq limit)."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
        ]
        five_page_scanned = ParseResult(
            text="",
            tables=[],
            page_count=5,
            is_scanned=True,
            images=[b"img"] * 5,
            error=None,
        )

        router = SmartRouter(mock_settings_all_providers)
        decision = router.select_provider(five_page_scanned, strategy="speed")

        # Should use Groq (at limit, not over)
        assert decision.provider_name == "groq"

    @patch("src.extraction.router.is_provider_available")
    def test_one_over_groq_limit(self, mock_available, mock_settings_all_providers):
        """Test scanned PDF with 6 pages (just over Groq limit)."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
        ]
        six_page_scanned = ParseResult(
            text="",
            tables=[],
            page_count=6,
            is_scanned=True,
            images=[b"img"] * 6,
            error=None,
        )

        router = SmartRouter(mock_settings_all_providers)
        decision = router.select_provider(six_page_scanned, strategy="speed")

        # Should use Fireworks (over Groq limit)
        assert decision.provider_name == "fireworks"


# ============================================================================
# H6: Complexity-Based Routing Tests
# ============================================================================


class TestH6ComplexityRouting:
    """Tests for H6 context window management via complexity-based routing."""

    def test_calculate_complexity_with_no_tables(
        self, mock_settings_all_providers, text_pdf_small
    ):
        """Test complexity calculation with no tables returns 0."""
        router = SmartRouter(mock_settings_all_providers)

        complexity = router._calculate_complexity(text_pdf_small)

        assert complexity == 0

    def test_calculate_complexity_with_tables(
        self, mock_settings_all_providers, text_pdf_complex
    ):
        """Test complexity calculation with tables returns row count."""
        router = SmartRouter(mock_settings_all_providers)

        complexity = router._calculate_complexity(text_pdf_complex)

        # 51 rows (1 header + 50 data rows)
        assert complexity == 51

    def test_calculate_complexity_below_threshold(
        self, mock_settings_all_providers, text_pdf_simple_with_table
    ):
        """Test complexity calculation with small table."""
        router = SmartRouter(mock_settings_all_providers)

        complexity = router._calculate_complexity(text_pdf_simple_with_table)

        # 21 rows (1 header + 20 data rows)
        assert complexity == 21
        assert complexity <= mock_settings_all_providers.complexity_routing_threshold

    @patch("src.extraction.router.is_provider_available")
    def test_complex_document_routes_to_gemini(
        self, mock_available, mock_settings_all_providers, text_pdf_complex
    ):
        """
        H6: Complex documents fall back to OpenAI while Gemini routing is disabled.
        """
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_complex, strategy="speed")

        assert decision.provider_name == "openai"
        assert decision.is_optimal is False
        assert "Gemini unavailable" in decision.reason

    @patch("src.extraction.router.is_provider_available")
    def test_complex_document_falls_back_to_openai_without_gemini(
        self, mock_available, mock_settings_openai_anthropic, text_pdf_complex
    ):
        """H6: Complex document falls back to OpenAI when Gemini unavailable."""
        mock_available.side_effect = lambda p, s: p in ["openai", "anthropic"]
        router = SmartRouter(mock_settings_openai_anthropic)

        decision = router.select_provider(text_pdf_complex, strategy="speed")

        assert decision.provider_name == "openai"
        assert decision.is_optimal is False
        assert "Gemini unavailable" in decision.reason

    @patch("src.extraction.router.is_provider_available")
    def test_complex_document_warns_when_only_groq_available(
        self, mock_available, mock_settings_groq_only, text_pdf_complex
    ):
        """H6: Complex document warns when only Groq available (may truncate)."""
        mock_available.side_effect = lambda p, s: p == "groq"
        router = SmartRouter(mock_settings_groq_only)

        decision = router.select_provider(text_pdf_complex, strategy="speed")

        assert decision.provider_name == "groq"
        assert decision.is_optimal is False
        assert "WARNING" in decision.reason or "truncate" in decision.reason.lower()

    @patch("src.extraction.router.is_provider_available")
    def test_simple_document_with_table_routes_to_groq(
        self, mock_available, mock_settings_all_providers, text_pdf_simple_with_table
    ):
        """H6: Simple document (<=40 rows) routes to Groq for speed."""
        mock_available.side_effect = lambda p, s: p in [
            "openai",
            "anthropic",
            "groq",
            "gemini",
        ]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(text_pdf_simple_with_table, strategy="speed")

        # Should use Groq because complexity is below threshold
        assert decision.provider_name == "groq"
        assert decision.is_optimal is True

    @patch("src.extraction.router.is_provider_available")
    def test_complexity_threshold_boundary_at_40(
        self, mock_available, mock_settings_all_providers
    ):
        """H6: Document with exactly 40 rows should use Groq (at threshold)."""
        from src.extraction.pdf_parser import TableData

        # Create table with exactly 40 rows
        boundary_table = TableData(
            page=1,
            rows=[[f"Item {i}", "1", "100.00"] for i in range(40)],
            headers=["Item", "Qty", "Price"],
        )
        boundary_result = ParseResult(
            text="Test content",
            tables=[boundary_table],
            page_count=2,
            is_scanned=False,
            images=None,
            error=None,
        )

        mock_available.side_effect = lambda p, s: p in ["openai", "groq", "gemini"]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(boundary_result, strategy="speed")

        # At threshold (40), should NOT trigger complexity routing
        assert decision.provider_name == "groq"

    @patch("src.extraction.router.is_provider_available")
    def test_complexity_threshold_boundary_at_41(
        self, mock_available, mock_settings_all_providers
    ):
        """H6: Document with 41 rows should use OpenAI fallback (over threshold)."""
        from src.extraction.pdf_parser import TableData

        # Create table with 41 rows (one over threshold)
        over_threshold_table = TableData(
            page=1,
            rows=[[f"Item {i}", "1", "100.00"] for i in range(41)],
            headers=["Item", "Qty", "Price"],
        )
        over_threshold_result = ParseResult(
            text="Test content",
            tables=[over_threshold_table],
            page_count=2,
            is_scanned=False,
            images=None,
            error=None,
        )

        mock_available.side_effect = lambda p, s: p in ["openai", "groq", "gemini"]
        router = SmartRouter(mock_settings_all_providers)

        decision = router.select_provider(over_threshold_result, strategy="speed")

        # Over threshold (41 > 40), should use the large-context fallback
        assert decision.provider_name == "openai"
        assert "Gemini unavailable" in decision.reason
