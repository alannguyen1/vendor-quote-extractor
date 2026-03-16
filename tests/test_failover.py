"""
Tests for provider failover behavior in the LLM extractor.

Tests verify that the extractor correctly fails over from primary to fallback
provider when specific error conditions occur:
- HTTP 5xx server errors (APIStatusError with status >= 500)
- Timeout errors (APITimeoutError or standard timeout exceptions)
- Rate limiting (RateLimitError with 429 status)
- Connection errors (APIConnectionError)

These tests supplement test_llm_extractor.py which covers basic failover behavior.
This file focuses specifically on the error types specified in the PRD/specs:
- extraction-pipeline.md: Failover on HTTP 5xx, timeout >10s, rate limit (429)
"""

import asyncio
from datetime import date

import pytest
from httpx import Request, Response
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from src.extraction.llm_extractor import LLMExtractor
from src.extraction.providers import BaseLLMProvider
from src.schemas import (
    Amounts,
    Entity,
    ItemType,
    LineItem,
    VendorQuote,
)

# ============================================================================
# Mock Provider for Failover Tests
# ============================================================================


class MockFailoverProvider(BaseLLMProvider):
    """Mock provider that can simulate various error conditions."""

    def __init__(
        self,
        name: str,
        error: Exception | None = None,
        result: VendorQuote | None = None,
    ):
        """Initialize mock provider.

        Args:
            name: Provider name (e.g., "openai", "anthropic")
            error: Exception to raise when called
            result: VendorQuote to return if no error
        """
        self._name = name
        self._error = error
        self._result = result
        self._model = "mock-model"
        self._vision_model = "mock-vision-model"
        self._is_available = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def is_available(self) -> bool:
        return self._is_available

    async def extract_text(self, prompt: str) -> VendorQuote:
        if self._error:
            raise self._error
        if self._result:
            return self._result
        raise ValueError("MockFailoverProvider: No result or error configured")

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        if self._error:
            raise self._error
        if self._result:
            return self._result
        raise ValueError("MockFailoverProvider: No result or error configured")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_quote() -> VendorQuote:
    """Create a sample VendorQuote for successful extraction."""
    return VendorQuote(
        quote_id="FAIL-001",
        quote_date=date(2024, 1, 15),
        currency="USD",
        vendor=Entity(name="Failover Test Vendor"),
        line_items=[
            LineItem(
                line_number=1,
                description="Test Item",
                quantity=1,
                unit_price=100.00,
                extended_price=100.00,
                item_type=ItemType.hardware,
            )
        ],
        amounts=Amounts(
            subtotal=100.00,
            grand_total=100.00,
        ),
    )


def create_openai_request() -> Request:
    """Create a mock httpx Request for OpenAI exceptions."""
    return Request("POST", "https://api.openai.com/v1/chat/completions")


# ============================================================================
# HTTP 5xx Server Error Tests
# ============================================================================


class TestHTTP5xxFailover:
    """Tests for failover triggered by HTTP 500, 502, 503 errors."""

    @pytest.mark.asyncio
    async def test_500_internal_server_error_triggers_failover(self, sample_quote):
        """HTTP 500 Internal Server Error should trigger failover."""
        response = Response(500, request=create_openai_request())
        error = APIStatusError(
            message="Internal Server Error",
            response=response,
            body={"error": {"message": "Server error"}},
        )

        primary = MockFailoverProvider("openai", error=error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_502_bad_gateway_triggers_failover(self, sample_quote):
        """HTTP 502 Bad Gateway should trigger failover."""
        response = Response(502, request=create_openai_request())
        error = APIStatusError(
            message="Bad Gateway",
            response=response,
            body={"error": {"message": "Gateway error"}},
        )

        primary = MockFailoverProvider("openai", error=error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_503_service_unavailable_triggers_failover(self, sample_quote):
        """HTTP 503 Service Unavailable should trigger failover."""
        response = Response(503, request=create_openai_request())
        error = APIStatusError(
            message="Service Unavailable",
            response=response,
            body={"error": {"message": "Service unavailable"}},
        )

        primary = MockFailoverProvider("openai", error=error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True


# ============================================================================
# Rate Limiting (429) Tests
# ============================================================================


class TestRateLimitFailover:
    """Tests for failover triggered by HTTP 429 Rate Limit errors."""

    @pytest.mark.asyncio
    async def test_429_rate_limit_triggers_failover(self, sample_quote):
        """HTTP 429 Rate Limit should trigger fallback."""
        response = Response(429, request=create_openai_request())
        error = RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        primary = MockFailoverProvider("openai", error=error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_429_without_fallback_returns_error(self):
        """HTTP 429 without fallback provider should return error."""
        response = Response(429, request=create_openai_request())
        error = RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        primary = MockFailoverProvider("openai", error=error)

        extractor = LLMExtractor(providers=[primary])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is None
        assert result.error is not None
        assert "Rate limit" in result.error
        assert result.fallback_used is False


# ============================================================================
# Timeout Error Tests
# ============================================================================


class TestTimeoutFailover:
    """Tests for failover triggered by timeout errors."""

    @pytest.mark.asyncio
    async def test_api_timeout_triggers_failover(self, sample_quote):
        """OpenAI APITimeoutError should trigger fallback."""
        error = APITimeoutError(request=create_openai_request())

        primary = MockFailoverProvider("openai", error=error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_asyncio_timeout_triggers_failover(self, sample_quote):
        """asyncio.TimeoutError should trigger fallback."""
        error = asyncio.TimeoutError()

        primary = MockFailoverProvider("openai", error=error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True


# ============================================================================
# Connection Error Tests
# ============================================================================


class TestConnectionErrorFailover:
    """Tests for failover triggered by connection errors."""

    @pytest.mark.asyncio
    async def test_connection_error_triggers_failover(self, sample_quote):
        """OpenAI APIConnectionError should trigger fallback."""
        error = APIConnectionError(request=create_openai_request())

        primary = MockFailoverProvider("openai", error=error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True
        assert result.error is None


# ============================================================================
# Both Providers Fail Tests
# ============================================================================


class TestBothProvidersFail:
    """Tests for when both providers fail."""

    @pytest.mark.asyncio
    async def test_both_providers_5xx_returns_combined_error(self):
        """When both providers return 5xx, error contains both messages."""
        openai_response = Response(500, request=create_openai_request())
        openai_error = APIStatusError(
            message="OpenAI internal error",
            response=openai_response,
            body={"error": {"message": "OpenAI internal error"}},
        )
        anthropic_error = Exception("Anthropic overloaded")

        primary = MockFailoverProvider("openai", error=openai_error)
        fallback = MockFailoverProvider("anthropic", error=anthropic_error)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is None
        assert result.error is not None
        assert "Both providers failed" in result.error
        assert result.fallback_used is True
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_openai_timeout_anthropic_rate_limit(self):
        """OpenAI timeout + Anthropic rate limit should fail gracefully."""
        openai_error = APITimeoutError(request=create_openai_request())
        anthropic_error = Exception("Anthropic rate limited")

        primary = MockFailoverProvider("openai", error=openai_error)
        fallback = MockFailoverProvider("anthropic", error=anthropic_error)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is None
        assert "Both providers failed" in result.error


# ============================================================================
# Fallback Flag Verification Tests
# ============================================================================


class TestFallbackFlagAccuracy:
    """Tests to verify the fallback_used flag is correctly set."""

    @pytest.mark.asyncio
    async def test_successful_primary_sets_fallback_false(self, sample_quote):
        """Successful primary extraction should set fallback_used=False."""
        primary = MockFailoverProvider("openai", result=sample_quote)

        extractor = LLMExtractor(providers=[primary])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is False

    @pytest.mark.asyncio
    async def test_successful_fallback_sets_flag_true(self, sample_quote):
        """Successful fallback should set fallback_used=True."""
        openai_error = Exception("OpenAI failed")

        primary = MockFailoverProvider("openai", error=openai_error)
        fallback = MockFailoverProvider("anthropic", result=sample_quote)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is not None
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_both_fail_sets_fallback_true(self):
        """When both fail, fallback_used=True (indicates failover was attempted)."""
        openai_error = Exception("OpenAI error")
        anthropic_error = Exception("Anthropic error")

        primary = MockFailoverProvider("openai", error=openai_error)
        fallback = MockFailoverProvider("anthropic", error=anthropic_error)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is None
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_primary_fail_no_fallback_sets_flag_false(self):
        """When primary fails and no fallback, fallback_used=False."""
        openai_error = Exception("OpenAI error")

        primary = MockFailoverProvider("openai", error=openai_error)

        extractor = LLMExtractor(providers=[primary])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.quote is None
        assert result.fallback_used is False


# ============================================================================
# Error Message Content Tests
# ============================================================================


class TestErrorMessages:
    """Tests for error message content and formatting."""

    @pytest.mark.asyncio
    async def test_combined_error_has_both_messages(self):
        """Combined error message should include both provider errors."""
        openai_error = Exception("Primary: OpenAI unavailable")
        anthropic_error = Exception("Fallback: Anthropic quota exceeded")

        primary = MockFailoverProvider("openai", error=openai_error)
        fallback = MockFailoverProvider("anthropic", error=anthropic_error)

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.error is not None
        # Error should contain info about both failures
        assert "Primary:" in result.error or "OpenAI" in result.error
        assert "Fallback:" in result.error or "Anthropic" in result.error

    @pytest.mark.asyncio
    async def test_single_provider_error_message(self):
        """Single provider failure should have clear error message."""
        openai_error = Exception("API key invalid")

        primary = MockFailoverProvider("openai", error=openai_error)

        extractor = LLMExtractor(providers=[primary])
        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test pdf",
        )

        assert result.error is not None
        assert "API key invalid" in result.error
        # Should not say "Both providers failed"
        assert "Both providers failed" not in result.error
