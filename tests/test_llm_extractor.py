"""
Tests for the LLM extractor module.

Tests cover:
- ExtractionResult dataclass structure
- LLMExtractor initialization with provider injection
- Table formatting helper
- Confidence calculation algorithm
- Placeholder quote_id generation
- Primary provider extraction (mocked)
- Provider failover behavior (mocked)
- Vision extraction for scanned PDFs (mocked)

Uses mock fixtures from tests/mocks/ for realistic test data matching actual PDFs.
"""

import hashlib
import re
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.llm_extractor import (
    EXTRACTION_PROMPT,
    VISION_EXTRACTION_PROMPT,
    ExtractionResult,
    LLMExtractor,
)
from src.extraction.pdf_parser import TableData
from src.extraction.providers import ProviderConfig
from src.extraction.providers.base import BaseLLMProvider
from src.schemas import (
    Amounts,
    Entity,
    LineItem,
    LineItemBatch,
    VendorQuote,
    VendorQuoteHeader,
)
from tests.mocks import MINIMAL_QUOTE, MIXED_CATEGORY_QUOTE

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_quote() -> VendorQuote:
    """
    Return MIXED_CATEGORY_QUOTE fixture for mixed_category_quote.pdf.

    This provides $32,198.00 grand total with 3 line items.
    """
    return MIXED_CATEGORY_QUOTE


@pytest.fixture
def minimal_quote() -> VendorQuote:
    """Return MINIMAL_QUOTE fixture - minimal valid quote for edge case testing."""
    return MINIMAL_QUOTE


@pytest.fixture
def sample_tables() -> list[TableData]:
    """Create sample table data for testing."""
    return [
        TableData(
            page=1,
            rows=[
                ["Widget A", "2", "$50.00", "$100.00"],
                ["Widget B", "3", "$30.00", "$90.00"],
            ],
            headers=["Description", "Qty", "Unit Price", "Total"],
            bbox=(50, 100, 550, 300),
        ),
        TableData(
            page=2,
            rows=[
                ["Service Fee", "1", "$25.00", "$25.00"],
            ],
            headers=["Description", "Qty", "Unit Price", "Total"],
            bbox=(50, 100, 550, 200),
        ),
    ]


@pytest.fixture
def mock_settings():
    """Create mock settings with API keys configured."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-openai-key"
    settings.anthropic_api_key = "sk-ant-test-key"
    settings.llm_model = "gpt-4o-2024-08-06"
    settings.anthropic_model = "claude-3-5-sonnet-20241022"
    settings.llm_provider = "openai"
    settings.has_anthropic_key.return_value = True
    # Speed Phase 5 settings
    settings.max_text_chars = 30000
    settings.max_table_chars = 10000
    # H6: Per-provider context window limits
    settings.provider_max_text_chars = {
        "groq": 15000,
        "openai": 250000,
        "gemini": 500000,
        "anthropic": 200000,
        "fireworks": 60000,
    }
    settings.provider_max_table_chars = {
        "groq": 15000,
        "openai": 50000,
        "gemini": 100000,
        "anthropic": 50000,
        "fireworks": 20000,
    }
    # Configurable thresholds (v31.0)
    settings.math_tolerance = 0.01
    settings.line_item_low_confidence_threshold = 0.9
    # Chunking settings
    settings.chunking_enabled = True
    settings.chunk_rows_per_call = 15
    settings.chunking_min_rows = 30
    settings.large_doc_text_budget = 8000
    settings.complexity_routing_threshold = 40
    settings.enable_few_shot = False
    settings.chunk_parallel_concurrency = 3
    settings.provider_single_pass_max_rows = {
        "groq": 30,
        "openai": 40,
        "gemini": 150,
        "anthropic": 60,
        "fireworks": 30,
    }
    settings.quantity_auto_correction = True
    return settings


@pytest.fixture
def mock_settings_no_anthropic():
    """Create mock settings without Anthropic key."""
    settings = MagicMock()
    settings.openai_api_key = "sk-test-openai-key"
    settings.anthropic_api_key = None
    settings.llm_model = "gpt-4o-2024-08-06"
    settings.anthropic_model = None
    settings.llm_provider = "openai"
    settings.has_anthropic_key.return_value = False
    # Speed Phase 5 settings
    settings.max_text_chars = 30000
    settings.max_table_chars = 10000
    # H6: Per-provider context window limits
    settings.provider_max_text_chars = {
        "groq": 15000,
        "openai": 250000,
        "gemini": 500000,
        "anthropic": 200000,
        "fireworks": 60000,
    }
    settings.provider_max_table_chars = {
        "groq": 15000,
        "openai": 50000,
        "gemini": 100000,
        "anthropic": 50000,
        "fireworks": 20000,
    }
    # Configurable thresholds (v31.0)
    settings.math_tolerance = 0.01
    settings.line_item_low_confidence_threshold = 0.9
    # Chunking settings
    settings.chunking_enabled = True
    settings.chunk_rows_per_call = 15
    settings.chunking_min_rows = 30
    settings.large_doc_text_budget = 8000
    settings.complexity_routing_threshold = 40
    settings.enable_few_shot = False
    settings.chunk_parallel_concurrency = 3
    settings.provider_single_pass_max_rows = {
        "groq": 30,
        "openai": 40,
        "gemini": 150,
        "anthropic": 60,
        "fireworks": 30,
    }
    return settings


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        extract_text_result: VendorQuote | None = None,
        extract_text_error: Exception | None = None,
        extract_vision_result: VendorQuote | None = None,
        extract_vision_error: Exception | None = None,
        extract_text_handler=None,
    ):
        config = ProviderConfig(api_key="mock-key", model="mock-model")
        super().__init__(config)
        self._name = name
        self._client = MagicMock()  # Set client to make is_available work
        self._extract_text_result = extract_text_result
        self._extract_text_error = extract_text_error
        self._extract_vision_result = extract_vision_result
        self._extract_vision_error = extract_vision_error
        self._extract_text_handler = extract_text_handler

    @property
    def name(self) -> str:
        return self._name

    async def extract_text(self, prompt: str, response_model=None):
        if self._extract_text_error:
            raise self._extract_text_error
        if self._extract_text_handler:
            return self._extract_text_handler(prompt, response_model)
        if self._extract_text_result:
            return self._extract_text_result
        raise RuntimeError("No mock result configured")

    async def extract_vision(self, images: list[bytes], prompt: str) -> VendorQuote:
        if self._extract_vision_error:
            raise self._extract_vision_error
        if self._extract_vision_result:
            return self._extract_vision_result
        raise RuntimeError("No mock result configured")


class MockAsyncProvider(MockProvider):
    """Mock provider with async handler support for concurrency testing."""

    def __init__(self, name: str = "mock", extract_text_handler=None):
        super().__init__(name=name)
        self._async_handler = extract_text_handler

    async def extract_text(self, prompt: str, response_model=None):
        if self._async_handler:
            return await self._async_handler(prompt, response_model)
        return await super().extract_text(prompt, response_model)


# ============================================================================
# ExtractionResult Tests
# ============================================================================


class TestExtractionResult:
    """Tests for the ExtractionResult dataclass."""

    def test_successful_result(self, sample_quote: VendorQuote):
        """Test creating a successful extraction result."""
        result = ExtractionResult(
            quote=sample_quote,
            confidence=0.95,
            error=None,
            fallback_used=False,
        )
        assert result.quote is not None
        assert result.quote.quote_id == "MCQ-2024-0001"
        assert result.confidence == 0.95
        assert result.error is None
        assert result.fallback_used is False

    def test_failed_result(self):
        """Test creating a failed extraction result."""
        result = ExtractionResult(
            quote=None,
            confidence=0.0,
            error="API timeout",
            fallback_used=False,
        )
        assert result.quote is None
        assert result.confidence == 0.0
        assert result.error == "API timeout"
        assert result.fallback_used is False

    def test_fallback_result(self, sample_quote: VendorQuote):
        """Test creating a result that used fallback provider."""
        result = ExtractionResult(
            quote=sample_quote,
            confidence=0.85,
            error=None,
            fallback_used=True,
        )
        assert result.quote is not None
        assert result.fallback_used is True

    def test_result_with_provider_used(self, sample_quote: VendorQuote):
        """Test result includes provider_used field."""
        result = ExtractionResult(
            quote=sample_quote,
            confidence=0.95,
            error=None,
            fallback_used=False,
            provider_used="openai",
        )
        assert result.provider_used == "openai"


# ============================================================================
# LLMExtractor Initialization Tests
# ============================================================================


class TestLLMExtractorInit:
    """Tests for LLMExtractor initialization."""

    @patch("src.extraction.llm_extractor.get_settings")
    @patch("src.extraction.llm_extractor.get_providers")
    def test_init_with_providers_from_settings(
        self, mock_get_providers, mock_get_settings, mock_settings, sample_quote
    ):
        """Test initialization loads providers from settings."""
        mock_get_settings.return_value = mock_settings
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        mock_get_providers.return_value = [mock_provider]

        extractor = LLMExtractor()

        assert extractor.primary_provider.name == "openai"
        assert not extractor.has_fallback

    @patch("src.extraction.llm_extractor.get_settings")
    def test_init_with_injected_providers(
        self, mock_get_settings, mock_settings, sample_quote
    ):
        """Test initialization with injected providers."""
        mock_get_settings.return_value = mock_settings
        mock_provider = MockProvider(name="test", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])

        assert extractor.primary_provider.name == "test"

    @patch("src.extraction.llm_extractor.get_settings")
    @patch("src.extraction.llm_extractor.get_providers")
    def test_init_with_multiple_providers(
        self, mock_get_providers, mock_get_settings, mock_settings, sample_quote
    ):
        """Test initialization with multiple providers."""
        mock_get_settings.return_value = mock_settings
        provider1 = MockProvider(name="openai", extract_text_result=sample_quote)
        provider2 = MockProvider(name="anthropic", extract_text_result=sample_quote)
        mock_get_providers.return_value = [provider1, provider2]

        extractor = LLMExtractor()

        assert extractor.has_fallback

    @patch("src.extraction.llm_extractor.get_settings")
    @patch("src.extraction.llm_extractor.get_providers")
    def test_init_raises_with_no_providers(
        self, mock_get_providers, mock_get_settings, mock_settings
    ):
        """Test initialization raises when no providers available."""
        mock_get_settings.return_value = mock_settings
        mock_get_providers.return_value = []

        with pytest.raises(RuntimeError, match="No LLM providers available"):
            LLMExtractor()


# ============================================================================
# Table Formatting Tests
# ============================================================================


class TestFormatTables:
    """Tests for the _format_tables helper method."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_format_tables_with_headers(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test formatting tables with headers."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        tables = [
            TableData(
                page=1,
                rows=[["A", "1", "$10"], ["B", "2", "$20"]],
                headers=["Name", "Qty", "Price"],
                bbox=None,
            )
        ]

        result = extractor._format_tables(tables)

        assert "[Page 1]" in result
        assert "Name | Qty | Price" in result
        assert "[R01] A | 1 | $10" in result
        assert "[R02] B | 2 | $20" in result

    @patch("src.extraction.llm_extractor.get_settings")
    def test_format_tables_without_headers(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test formatting tables without headers."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"]],
                headers=[],
                bbox=None,
            )
        ]

        result = extractor._format_tables(tables)

        assert "[R01] A | 1" in result
        assert "[R02] B | 2" in result
        # No header = no page marker in current impl
        assert "[Page 1]" not in result

    @patch("src.extraction.llm_extractor.get_settings")
    def test_format_empty_tables(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test formatting empty table list."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        result = extractor._format_tables([])

        assert result == "No tables detected."

    @patch("src.extraction.llm_extractor.get_settings")
    def test_format_multiple_tables(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_tables,
        sample_quote,
    ):
        """Test formatting multiple tables from different pages."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        result = extractor._format_tables(sample_tables)

        assert "[Page 1]" in result
        assert "[Page 2]" in result
        assert "Widget A" in result
        assert "Service Fee" in result


# ============================================================================
# Confidence Calculation Tests
# ============================================================================


class TestConfidenceCalculation:
    """Tests for the _calculate_confidence method."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_full_confidence_all_fields(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_quote,
    ):
        """Test confidence calculation with all fields present."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        confidence = extractor._calculate_confidence(sample_quote)

        # Required: vendor.name, grand_total > 0, line_items = 3/3 = 1.0
        # Optional: quote_id, quote_date, customer, commercial_terms = 4/4 = 1.0
        # Total: 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert confidence == 1.0

    @patch("src.extraction.llm_extractor.get_settings")
    def test_confidence_minimal_quote(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        minimal_quote,
    ):
        """Test confidence calculation with only required fields."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=minimal_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        confidence = extractor._calculate_confidence(minimal_quote)

        # Required: vendor.name, grand_total > 0, line_items = 3/3 = 1.0
        # Optional: all None = 0/4 = 0.0
        # Total: 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert confidence == 0.7

    @patch("src.extraction.llm_extractor.get_settings")
    def test_confidence_with_no_line_items(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test confidence drops when line_items is empty."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])

        # Create a mock quote to test edge case
        quote = MagicMock()
        quote.vendor = Entity(name="Valid Vendor")
        quote.amounts = MagicMock()
        quote.amounts.grand_total = 100.00
        quote.line_items = []  # Empty - this should reduce confidence
        quote.quote_id = None
        quote.quote_date = None
        quote.customer = None
        quote.commercial_terms = None

        confidence = extractor._calculate_confidence(quote)

        # Required: vendor.name=True, grand_total>0=True, line_items(empty)=False
        # Optional: all None = 0/4
        # Total: 0.7 * (2/3) + 0.3 * 0 = ~0.47
        assert confidence == pytest.approx(0.47, abs=0.01)

    @patch("src.extraction.llm_extractor.get_settings")
    def test_confidence_zero_grand_total(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test confidence drops when grand_total is zero."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=0.00,
                    extended_price=0.00,
                )
            ],
            amounts=Amounts(subtotal=0.00, grand_total=0.00),
        )

        confidence = extractor._calculate_confidence(quote)

        # Required: vendor.name=True, grand_total>0=False, line_items=True = 2/3
        assert confidence == pytest.approx(0.47, abs=0.01)

    @patch("src.extraction.llm_extractor.get_settings")
    def test_confidence_partial_optional_fields(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test confidence with some optional fields present."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        quote = VendorQuote(
            quote_id="Q-123",  # Present
            quote_date=date(2024, 1, 1),  # Present
            currency="USD",
            vendor=Entity(name="Vendor"),
            customer=None,  # Missing
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.00,
                    extended_price=100.00,
                )
            ],
            amounts=Amounts(subtotal=100.00, grand_total=100.00),
            commercial_terms=None,  # Missing
        )

        confidence = extractor._calculate_confidence(quote)

        # Required: 3/3 = 1.0
        # Optional: quote_id + quote_date present, customer + terms missing = 2/4
        # Total: 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        assert confidence == 0.85


# ============================================================================
# LineItem Confidence Calculation Tests
# ============================================================================


class TestLineItemConfidenceCalculation:
    """Tests for per-LineItem confidence calculation (Gap 3.2 resolution).

    Confidence formula:
    - 40% weight: item_type classification present (critical for finance routing)
    - 25% weight: math consistency (quantity * unit_price ≈ extended_price)
    - 20% weight: SKU present (helps procurement matching)
    - 15% weight: manufacturer present (vendor identification)
    """

    @patch("src.extraction.llm_extractor.get_settings")
    def test_line_item_full_confidence(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test line item with all fields gets high confidence."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        # Item with all optional fields and consistent math
        from src.schemas import ItemType

        item = LineItem(
            line_number=1,
            description="Test Item",
            sku="SKU-12345",
            manufacturer="Test Mfg",
            quantity=2,
            unit_price=50.00,
            extended_price=100.00,  # 2 * 50 = 100 (consistent)
            item_type=ItemType.hardware,
        )

        confidence = extractor._calculate_line_item_confidence(item)

        # 40% classification + 25% math + 20% sku + 15% manufacturer = 100%
        assert confidence == 1.0

    @patch("src.extraction.llm_extractor.get_settings")
    def test_line_item_only_classification(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test line item with only classification present."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        from src.schemas import ItemType

        item = LineItem(
            line_number=1,
            description="Test Item",
            sku=None,  # Missing
            manufacturer=None,  # Missing
            quantity=2,
            unit_price=50.00,
            extended_price=100.00,  # Consistent
            item_type=ItemType.software,
        )

        confidence = extractor._calculate_line_item_confidence(item)

        # 40% classification + 25% math + 0% sku + 0% manufacturer = 65%
        assert confidence == 0.65

    @patch("src.extraction.llm_extractor.get_settings")
    def test_line_item_no_optional_fields(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test line item with no optional fields gets low confidence."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        item = LineItem(
            line_number=1,
            description="Test Item",
            sku=None,
            manufacturer=None,
            quantity=2,
            unit_price=50.00,
            extended_price=100.00,  # Consistent
            item_type=None,  # No classification
        )

        confidence = extractor._calculate_line_item_confidence(item)

        # 0% classification + 25% math + 0% sku + 0% manufacturer = 25%
        assert confidence == 0.25

    @patch("src.extraction.llm_extractor.get_settings")
    def test_line_item_math_inconsistency(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test line item with math inconsistency reduces confidence."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        from src.schemas import ItemType

        item = LineItem(
            line_number=1,
            description="Test Item",
            sku="SKU-123",
            manufacturer="Mfg",
            quantity=2,
            unit_price=50.00,
            extended_price=150.00,  # Should be 100, inconsistent!
            item_type=ItemType.hardware,
        )

        confidence = extractor._calculate_line_item_confidence(item)

        # 40% classification + 0% math (inconsistent) + 20% sku + 15% manufacturer = 75%
        assert confidence == 0.75

    @patch("src.extraction.llm_extractor.get_settings")
    def test_line_item_math_tolerance(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test line item with math within $0.01 tolerance passes."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        from src.schemas import ItemType

        item = LineItem(
            line_number=1,
            description="Test Item",
            sku="SKU-123",
            manufacturer="Mfg",
            quantity=3,
            unit_price=33.33,
            extended_price=99.99,  # 3 * 33.33 = 99.99 (exact match)
            item_type=ItemType.services,
        )

        confidence = extractor._calculate_line_item_confidence(item)

        # All fields present, math consistent
        assert confidence == 1.0

    @patch("src.extraction.llm_extractor.get_settings")
    def test_line_item_credit_negative_extended(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test credit line items with negative extended price."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        from src.schemas import ItemType

        item = LineItem(
            line_number=1,
            description="Credit",
            sku=None,
            manufacturer=None,
            quantity=1,
            unit_price=100.00,
            extended_price=-100.00,  # Negative for credit
            item_type=ItemType.credit,
        )

        confidence = extractor._calculate_line_item_confidence(item)

        # abs(-100) = 100 = 1 * 100, so math is consistent
        # 40% classification + 25% math + 0% sku + 0% manufacturer = 65%
        assert confidence == 0.65

    @patch("src.extraction.llm_extractor.get_settings")
    def test_line_item_empty_string_fields(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Test empty string fields are treated as missing."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        from src.schemas import ItemType

        item = LineItem(
            line_number=1,
            description="Test",
            sku="",  # Empty string - should count as missing
            manufacturer="   ",  # Whitespace only - should count as missing
            quantity=1,
            unit_price=100.00,
            extended_price=100.00,
            item_type=ItemType.hardware,
        )

        confidence = extractor._calculate_line_item_confidence(item)

        # 40% classification + 25% math + 0% sku + 0% manufacturer = 65%
        assert confidence == 0.65

    @patch("src.extraction.llm_extractor.get_settings")
    def test_enrich_line_items_updates_all_items(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Test enrichment updates confidence for all line items in quote."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=None)
        extractor = LLMExtractor(providers=[mock_provider])

        from src.schemas import ItemType

        # Create quote with line items having default confidence=1.0
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Full Item",
                    sku="SKU-001",
                    manufacturer="Mfg",
                    quantity=1,
                    unit_price=100.00,
                    extended_price=100.00,
                    item_type=ItemType.hardware,
                    confidence=1.0,  # Default
                ),
                LineItem(
                    line_number=2,
                    description="Minimal Item",
                    quantity=1,
                    unit_price=50.00,
                    extended_price=50.00,
                    item_type=None,  # Missing classification
                    confidence=1.0,  # Default
                ),
            ],
            amounts=Amounts(subtotal=150.00, grand_total=150.00),
        )

        enriched = extractor._enrich_line_items_confidence(quote)

        # First item: full fields → 1.0
        assert enriched.line_items[0].confidence == 1.0
        # Second item: no classification, no sku, no manufacturer → 0.25
        assert enriched.line_items[1].confidence == 0.25

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_enriches_line_item_confidence(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Test extract() method enriches line item confidence."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        from src.schemas import ItemType

        # Quote with items needing confidence calculation
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item with classification",
                    quantity=1,
                    unit_price=100.00,
                    extended_price=100.00,
                    item_type=ItemType.software,
                    confidence=1.0,  # Will be recalculated
                ),
            ],
            amounts=Amounts(subtotal=100.00, grand_total=100.00),
        )
        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        result = await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test",
        )

        assert result.quote is not None
        # Item has classification + consistent math but no sku/manufacturer
        # 40% + 25% + 0% + 0% = 65%
        assert result.quote.line_items[0].confidence == 0.65

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_vision_extract_enriches_line_item_confidence(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Test extract_from_images() also enriches line item confidence."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        # Quote with minimal fields
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Scanned item",
                    quantity=1,
                    unit_price=200.00,
                    extended_price=200.00,
                    item_type=None,  # No classification
                    confidence=1.0,
                ),
            ],
            amounts=Amounts(subtotal=200.00, grand_total=200.00),
        )
        mock_provider = MockProvider(name="openai", extract_vision_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        result = await extractor.extract_from_images(
            images=[b"fake image"],
            pdf_bytes=b"test",
        )

        assert result.quote is not None
        # No classification, no sku, no manufacturer, but math consistent
        # 0% + 25% + 0% + 0% = 25%
        assert result.quote.line_items[0].confidence == 0.25


# ============================================================================
# Placeholder Quote ID Tests
# ============================================================================


class TestPlaceholderQuoteId:
    """Tests for placeholder quote_id generation."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_placeholder_id_format(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        minimal_quote,
    ):
        """Test placeholder ID follows VQE-{hash[:8]} format."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=minimal_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        pdf_bytes = b"test pdf content"

        result = extractor._add_placeholder_quote_id(minimal_quote, pdf_bytes)

        expected_hash = hashlib.sha256(pdf_bytes).hexdigest()[:8].upper()
        assert result.quote_id == f"VQE-{expected_hash}"

    @patch("src.extraction.llm_extractor.get_settings")
    def test_placeholder_id_deterministic(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        minimal_quote,
    ):
        """Test same PDF bytes produces same placeholder ID."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=minimal_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        pdf_bytes = b"consistent content"

        result1 = extractor._add_placeholder_quote_id(minimal_quote, pdf_bytes)
        result2 = extractor._add_placeholder_quote_id(minimal_quote, pdf_bytes)

        assert result1.quote_id == result2.quote_id

    @patch("src.extraction.llm_extractor.get_settings")
    def test_placeholder_preserves_other_fields(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_quote,
    ):
        """Test adding placeholder ID doesn't modify other fields."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        # Remove quote_id to simulate missing ID
        quote_without_id = sample_quote.model_copy(update={"quote_id": None})
        pdf_bytes = b"test content"

        result = extractor._add_placeholder_quote_id(quote_without_id, pdf_bytes)

        assert result.quote_id is not None
        assert result.quote_id.startswith("VQE-")
        assert result.vendor.name == sample_quote.vendor.name
        assert result.amounts.grand_total == sample_quote.amounts.grand_total
        assert len(result.line_items) == len(sample_quote.line_items)


# ============================================================================
# Extract Method Tests (Mocked)
# ============================================================================


class TestExtractMethod:
    """Tests for the main extract() method with mocked providers."""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_success_primary(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_quote,
    ):
        """Test successful extraction with primary provider."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        result = await extractor.extract(
            text="Sample document text",
            tables=[],
            page_count=1,
            pdf_bytes=b"pdf content",
        )

        assert result.quote is not None
        assert result.error is None
        assert result.fallback_used is False
        assert result.confidence > 0
        assert result.provider_used == "openai"

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_failover_to_second_provider(
        self,
        mock_get_settings,
        mock_settings,
        sample_quote,
    ):
        """Test failover to second provider when primary fails."""
        mock_get_settings.return_value = mock_settings

        # Primary fails
        primary = MockProvider(
            name="openai",
            extract_text_error=Exception("OpenAI timeout"),
        )
        # Fallback succeeds
        fallback = MockProvider(
            name="anthropic",
            extract_text_result=sample_quote,
        )

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Sample document text",
            tables=[],
            page_count=1,
            pdf_bytes=b"pdf content",
        )

        assert result.quote is not None
        assert result.error is None
        assert result.fallback_used is True
        assert result.provider_used == "anthropic"

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_both_providers_fail(
        self,
        mock_get_settings,
        mock_settings,
    ):
        """Test error when both providers fail."""
        mock_get_settings.return_value = mock_settings

        primary = MockProvider(
            name="openai",
            extract_text_error=Exception("OpenAI error"),
        )
        fallback = MockProvider(
            name="anthropic",
            extract_text_error=Exception("Anthropic error"),
        )

        extractor = LLMExtractor(providers=[primary, fallback])
        result = await extractor.extract(
            text="Sample document text",
            tables=[],
            page_count=1,
            pdf_bytes=b"pdf content",
        )

        assert result.quote is None
        assert result.error is not None
        assert "Both providers failed" in result.error
        assert "openai: OpenAI error" in result.error
        assert "anthropic: Anthropic error" in result.error
        assert result.fallback_used is True
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_single_provider_fail(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
    ):
        """Test error when single provider fails (no fallback)."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        provider = MockProvider(
            name="openai",
            extract_text_error=Exception("OpenAI unavailable"),
        )

        extractor = LLMExtractor(providers=[provider])
        result = await extractor.extract(
            text="Sample document text",
            tables=[],
            page_count=1,
            pdf_bytes=b"pdf content",
        )

        assert result.quote is None
        assert result.error is not None
        assert "OpenAI unavailable" in result.error
        assert result.fallback_used is False

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_adds_placeholder_id_when_missing(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        minimal_quote,
    ):
        """Test placeholder quote_id is added when LLM doesn't extract one."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        # Quote without quote_id
        quote_no_id = minimal_quote.model_copy(update={"quote_id": None})
        mock_provider = MockProvider(name="openai", extract_text_result=quote_no_id)

        extractor = LLMExtractor(providers=[mock_provider])
        pdf_bytes = b"unique pdf content"
        result = await extractor.extract(
            text="Sample document text",
            tables=[],
            page_count=1,
            pdf_bytes=pdf_bytes,
        )

        assert result.quote is not None
        assert result.quote.quote_id is not None
        assert result.quote.quote_id.startswith("VQE-")

        # Verify hash matches
        expected_hash = hashlib.sha256(pdf_bytes).hexdigest()[:8].upper()
        assert result.quote.quote_id == f"VQE-{expected_hash}"

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_preserves_existing_quote_id(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_quote,
    ):
        """Test existing quote_id is not overwritten."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        result = await extractor.extract(
            text="Sample document text",
            tables=[],
            page_count=1,
            pdf_bytes=b"pdf content",
        )

        assert result.quote is not None
        assert result.quote.quote_id == "MCQ-2024-0001"  # Original ID preserved


# ============================================================================
# Vision Extraction Tests (Mocked)
# ============================================================================


class TestVisionExtraction:
    """Tests for extract_from_images() method with mocked vision API."""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_vision_extraction_success(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_quote,
    ):
        """Test successful vision extraction from page images."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_vision_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])

        # Simulate PNG image bytes
        fake_images = [b"PNG image page 1", b"PNG image page 2"]

        result = await extractor.extract_from_images(
            images=fake_images, pdf_bytes=b"original pdf"
        )

        assert result.quote is not None
        assert result.error is None
        assert result.fallback_used is False
        assert result.provider_used == "openai"

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_vision_extraction_failure(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
    ):
        """Test vision extraction error handling."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(
            name="openai",
            extract_vision_error=Exception("Vision API error"),
        )

        extractor = LLMExtractor(providers=[mock_provider])

        result = await extractor.extract_from_images(
            images=[b"fake image"], pdf_bytes=b"pdf"
        )

        assert result.quote is None
        assert result.error is not None
        assert "Vision extraction failed" in result.error

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_vision_extraction_failover(
        self,
        mock_get_settings,
        mock_settings,
        sample_quote,
    ):
        """Test vision extraction failover to second provider."""
        mock_get_settings.return_value = mock_settings

        primary = MockProvider(
            name="openai",
            extract_vision_error=Exception("OpenAI vision error"),
        )
        fallback = MockProvider(
            name="anthropic",
            extract_vision_result=sample_quote,
        )

        extractor = LLMExtractor(providers=[primary, fallback])

        result = await extractor.extract_from_images(
            images=[b"fake image"], pdf_bytes=b"pdf"
        )

        assert result.quote is not None
        assert result.fallback_used is True
        assert result.provider_used == "anthropic"

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_vision_adds_placeholder_id(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        minimal_quote,
    ):
        """Test vision extraction adds placeholder ID when missing."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote_no_id = minimal_quote.model_copy(update={"quote_id": None})
        mock_provider = MockProvider(name="openai", extract_vision_result=quote_no_id)

        extractor = LLMExtractor(providers=[mock_provider])
        pdf_bytes = b"scanned pdf content"

        result = await extractor.extract_from_images(
            images=[b"page image"], pdf_bytes=pdf_bytes
        )

        assert result.quote is not None
        assert result.quote.quote_id.startswith("VQE-")


# ============================================================================
# Extraction Prompt Tests
# ============================================================================


class TestExtractionPrompt:
    """Tests for the extraction prompt template."""

    def test_prompt_has_required_sections(self):
        """Test prompt contains required instruction sections."""
        assert "RULES:" in EXTRACTION_PROMPT
        assert "DOCUMENT TEXT:" in EXTRACTION_PROMPT
        assert "EXTRACTED TABLES:" in EXTRACTION_PROMPT

    def test_prompt_has_classification_rules(self):
        """Test prompt contains item type classification rules."""
        assert "hardware:" in EXTRACTION_PROMPT
        assert "software:" in EXTRACTION_PROMPT
        assert "services:" in EXTRACTION_PROMPT
        assert "discount/credit:" in EXTRACTION_PROMPT

    def test_prompt_has_amount_preference(self):
        """Test prompt has clear rules for amount extraction (H3 fix)."""
        # H3: Clarified grand_total extraction rules
        assert "AMOUNT EXTRACTION RULES:" in EXTRACTION_PROMPT
        assert "TCV (Total Contract Value)" in EXTRACTION_PROMPT
        assert "DISCOUNTED" in EXTRACTION_PROMPT
        assert "not MSRP" in EXTRACTION_PROMPT

    def test_prompt_placeholders(self):
        """Test prompt has correct placeholders for formatting."""
        assert "{text}" in EXTRACTION_PROMPT
        assert "{tables}" in EXTRACTION_PROMPT
        assert "{row_count}" in EXTRACTION_PROMPT

    def test_prompt_has_anti_dedup_instructions(self):
        """Test EXTRACTION_PROMPT contains anti-deduplication rules."""
        assert "LINE ITEM COMPLETENESS" in EXTRACTION_PROMPT
        assert "Do NOT merge or skip rows" in EXTRACTION_PROMPT
        assert "different locations or groups" in EXTRACTION_PROMPT
        assert "{row_count}" in EXTRACTION_PROMPT
        assert "Do not invent items" in EXTRACTION_PROMPT

    def test_vision_prompt_has_anti_dedup_instructions(self):
        """Test VISION_EXTRACTION_PROMPT contains anti-dedup rules (no row_count)."""
        assert "LINE ITEM COMPLETENESS" in VISION_EXTRACTION_PROMPT
        assert "do not merge or deduplicate" in VISION_EXTRACTION_PROMPT
        assert "different locations or groups" in VISION_EXTRACTION_PROMPT
        assert "Do not invent items" in VISION_EXTRACTION_PROMPT
        # Vision prompt should NOT have row_count placeholder
        assert "{row_count}" not in VISION_EXTRACTION_PROMPT


# ============================================================================
# Integration-style Tests (Still Mocked, but More Realistic)
# ============================================================================


class TestExtractorIntegration:
    """Integration-style tests with more realistic scenarios."""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_with_tables(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_quote,
        sample_tables,
    ):
        """Test extraction with document tables."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])
        result = await extractor.extract(
            text="Invoice #12345\nVendor: Test Corp",
            tables=sample_tables,
            page_count=2,
            pdf_bytes=b"pdf content",
        )

        assert result.quote is not None

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_truncates_long_text(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_quote,
    ):
        """Test text truncation for very long documents."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)

        extractor = LLMExtractor(providers=[mock_provider])

        # Create text longer than 30000 chars
        long_text = "A" * 50000

        result = await extractor.extract(
            text=long_text,
            tables=[],
            page_count=100,
            pdf_bytes=b"pdf content",
        )

        assert result.quote is not None


# ============================================================================
# Few-Shot Integration Tests (H4)
# ============================================================================


class TestFewShotIntegration:
    """Tests for H4: Few-shot learning integration into extraction prompts.

    These tests verify that:
    - _get_few_shot_context() retrieves examples from database
    - Few-shot context is properly appended to prompts
    - Empty examples don't break the flow
    - Database errors are handled gracefully
    """

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_session_factory")
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_get_few_shot_context_with_examples(
        self, mock_get_settings, mock_get_session_factory, mock_settings_no_anthropic
    ):
        """Test _get_few_shot_context returns formatted examples."""
        mock_settings_no_anthropic.enable_few_shot = True
        mock_get_settings.return_value = mock_settings_no_anthropic

        # Create mock CorrectionExample objects
        from unittest.mock import AsyncMock

        mock_example = MagicMock()
        mock_example.field_path = "amounts.grand_total"
        mock_example.original_value = "552012.00"
        mock_example.corrected_value = "597312.00"
        mock_example.document_context = "TCV vs net total confusion"

        # Mock the session and get_examples_for_field
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session
        mock_get_session_factory.return_value = mock_factory

        with patch(
            "src.extraction.llm_extractor.get_examples_for_field",
            new_callable=AsyncMock,
        ) as mock_get_examples:
            mock_get_examples.return_value = [mock_example]

            quote = VendorQuote(
                currency="USD",
                vendor=Entity(name="Test"),
                line_items=[
                    LineItem(
                        line_number=1,
                        description="Item",
                        quantity=1,
                        unit_price=100.0,
                        extended_price=100.0,
                    )
                ],
                amounts=Amounts(subtotal=100.0, grand_total=100.0),
            )
            mock_provider = MockProvider(name="openai", extract_text_result=quote)
            extractor = LLMExtractor(providers=[mock_provider])

            context = await extractor._get_few_shot_context()

            # Should have called get_examples_for_field for target fields
            assert mock_get_examples.call_count >= 1
            # Context should contain formatted examples with WRONG/CORRECT markers
            assert "EXTRACTION CORRECTIONS - Learn from these past mistakes:" in context
            assert 'WRONG (do not use): "552012.00"' in context
            assert 'CORRECT (use this): "597312.00"' in context

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_session_factory")
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_get_few_shot_context_empty_examples(
        self, mock_get_settings, mock_get_session_factory, mock_settings_no_anthropic
    ):
        """Test _get_few_shot_context returns empty string when no examples."""
        mock_settings_no_anthropic.enable_few_shot = True
        mock_get_settings.return_value = mock_settings_no_anthropic

        from unittest.mock import AsyncMock

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session
        mock_get_session_factory.return_value = mock_factory

        with patch(
            "src.extraction.llm_extractor.get_examples_for_field",
            new_callable=AsyncMock,
        ) as mock_get_examples:
            mock_get_examples.return_value = []  # No examples

            quote = VendorQuote(
                currency="USD",
                vendor=Entity(name="Test"),
                line_items=[
                    LineItem(
                        line_number=1,
                        description="Item",
                        quantity=1,
                        unit_price=100.0,
                        extended_price=100.0,
                    )
                ],
                amounts=Amounts(subtotal=100.0, grand_total=100.0),
            )
            mock_provider = MockProvider(name="openai", extract_text_result=quote)
            extractor = LLMExtractor(providers=[mock_provider])

            context = await extractor._get_few_shot_context()

            assert context == ""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_session_factory")
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_get_few_shot_context_handles_database_error(
        self, mock_get_settings, mock_get_session_factory, mock_settings_no_anthropic
    ):
        """Test _get_few_shot_context gracefully handles database errors."""
        mock_settings_no_anthropic.enable_few_shot = True
        mock_get_settings.return_value = mock_settings_no_anthropic

        # Make session factory raise an error
        mock_get_session_factory.side_effect = Exception("Database connection failed")

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        # Should not raise, should return empty string
        context = await extractor._get_few_shot_context()

        assert context == ""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_session_factory")
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_includes_few_shot_context(
        self, mock_get_settings, mock_get_session_factory, mock_settings_no_anthropic
    ):
        """Test extract() method includes few-shot context in prompt."""
        mock_settings_no_anthropic.enable_few_shot = True
        mock_get_settings.return_value = mock_settings_no_anthropic

        from unittest.mock import AsyncMock

        mock_example = MagicMock()
        mock_example.field_path = "line_items.item_type"
        mock_example.original_value = "hardware"
        mock_example.corrected_value = "services"
        mock_example.document_context = "Support contract"

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session
        mock_get_session_factory.return_value = mock_factory

        with patch(
            "src.extraction.llm_extractor.get_examples_for_field",
            new_callable=AsyncMock,
        ) as mock_get_examples:
            mock_get_examples.return_value = [mock_example]

            quote = VendorQuote(
                currency="USD",
                vendor=Entity(name="Test"),
                line_items=[
                    LineItem(
                        line_number=1,
                        description="Item",
                        quantity=1,
                        unit_price=100.0,
                        extended_price=100.0,
                    )
                ],
                amounts=Amounts(subtotal=100.0, grand_total=100.0),
            )
            mock_provider = MockProvider(name="openai", extract_text_result=quote)
            extractor = LLMExtractor(providers=[mock_provider])

            result = await extractor.extract(
                text="Test document",
                tables=[],
                page_count=1,
                pdf_bytes=b"test",
            )

            assert result.quote is not None
            # get_examples_for_field should have been called
            assert mock_get_examples.call_count >= 1

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_session_factory")
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_extract_from_images_includes_few_shot_context(
        self, mock_get_settings, mock_get_session_factory, mock_settings_no_anthropic
    ):
        """Test extract_from_images() includes few-shot context in prompt."""
        mock_settings_no_anthropic.enable_few_shot = True
        mock_get_settings.return_value = mock_settings_no_anthropic

        from unittest.mock import AsyncMock

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session
        mock_get_session_factory.return_value = mock_factory

        with patch(
            "src.extraction.llm_extractor.get_examples_for_field",
            new_callable=AsyncMock,
        ) as mock_get_examples:
            mock_get_examples.return_value = []  # No examples

            quote = VendorQuote(
                currency="USD",
                vendor=Entity(name="Test"),
                line_items=[
                    LineItem(
                        line_number=1,
                        description="Item",
                        quantity=1,
                        unit_price=100.0,
                        extended_price=100.0,
                    )
                ],
                amounts=Amounts(subtotal=100.0, grand_total=100.0),
            )
            mock_provider = MockProvider(name="openai", extract_vision_result=quote)
            extractor = LLMExtractor(providers=[mock_provider])

            result = await extractor.extract_from_images(
                images=[b"fake image"],
                pdf_bytes=b"test",
            )

            assert result.quote is not None
            # get_examples_for_field should have been called for vision extraction too
            assert mock_get_examples.call_count >= 1

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_session_factory")
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_few_shot_targets_correct_fields(
        self, mock_get_settings, mock_get_session_factory, mock_settings_no_anthropic
    ):
        """Test _get_few_shot_context queries the correct target fields."""
        mock_settings_no_anthropic.enable_few_shot = True
        mock_get_settings.return_value = mock_settings_no_anthropic

        from unittest.mock import AsyncMock

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session
        mock_get_session_factory.return_value = mock_factory

        with patch(
            "src.extraction.llm_extractor.get_examples_for_field",
            new_callable=AsyncMock,
        ) as mock_get_examples:
            mock_get_examples.return_value = []

            quote = VendorQuote(
                currency="USD",
                vendor=Entity(name="Test"),
                line_items=[
                    LineItem(
                        line_number=1,
                        description="Item",
                        quantity=1,
                        unit_price=100.0,
                        extended_price=100.0,
                    )
                ],
                amounts=Amounts(subtotal=100.0, grand_total=100.0),
            )
            mock_provider = MockProvider(name="openai", extract_text_result=quote)
            extractor = LLMExtractor(providers=[mock_provider])

            await extractor._get_few_shot_context()

            # Check that target fields were queried
            called_fields = [
                call[1]["field_path"] for call in mock_get_examples.call_args_list
            ]
            assert "amounts.grand_total" in called_fields
            assert "line_items.item_type" in called_fields
            assert "line_items.extended_price" in called_fields
            assert "line_items.description" in called_fields


# ============================================================================
# Row-Aware Table Truncation Tests
# ============================================================================


class TestRowAwareTruncation:
    """Tests for _truncate_tables_row_aware() method.

    Verifies that table truncation preserves complete rows and headers
    instead of cutting mid-row via hard string slicing.
    """

    @patch("src.extraction.llm_extractor.get_settings")
    def test_no_truncation_when_within_limit(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Tables within the char limit should be returned unchanged."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        table_text = (
            "[Page 1]\nName | Qty | Price\n"
            "----------\nItem A | 2 | $50\nItem B | 3 | $30"
        )
        result, kept, total = extractor._truncate_tables_row_aware(table_text, 10000)

        assert result == table_text
        assert kept == total
        assert total == 2

    @patch("src.extraction.llm_extractor.get_settings")
    def test_truncation_preserves_complete_rows(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Truncation should never cut a row in half."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        lines = ["[Page 1]", "Name | Qty | Price", "----------"]
        for i in range(20):
            lines.append(f"Item {i:02d} | {i + 1} | ${(i + 1) * 10:.2f}")
        table_text = "\n".join(lines)

        # Set limit to roughly half the content
        limit = len(table_text) // 2
        result, kept, total = extractor._truncate_tables_row_aware(table_text, limit)

        assert total == 20
        assert 0 < kept < 20
        # Every line in result should be a complete line from the original
        result_lines = result.split("\n")
        for line in result_lines:
            assert line in lines

    @patch("src.extraction.llm_extractor.get_settings")
    def test_truncation_preserves_table_headers_across_multiple_tables(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Headers from multiple tables should be preserved when budget allows."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        table1_lines = [
            "[Page 1]",
            "Name | Qty | Price",
            "----------",
            "Item A | 2 | $50.00",
            "Item B | 3 | $30.00",
            "",
        ]
        table2_lines = [
            "[Page 2]",
            "Name | Qty | Price",
            "----------",
            "Item C | 1 | $100.00",
            "Item D | 4 | $25.00",
        ]
        table_text = "\n".join(table1_lines + table2_lines)

        # Budget enough for table 1 + table 2 header + some rows
        t2_header = "[Page 2]\nName | Qty | Price\n----------\nItem C | 1 | $100.00"
        budget = len("\n".join(table1_lines)) + len(t2_header) + 10
        result, kept, total = extractor._truncate_tables_row_aware(table_text, budget)

        assert total == 4
        assert "[Page 1]" in result
        assert "[Page 2]" in result
        assert kept >= 3  # At least table 1 rows + some table 2 rows

    @patch("src.extraction.llm_extractor.get_settings")
    def test_truncation_reports_row_counts(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Row counts should accurately reflect kept vs total data rows."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        lines = ["[Page 1]", "H1 | H2", "------"]
        for i in range(10):
            lines.append(f"Row {i} | Val {i}")
        table_text = "\n".join(lines)

        # Very tight limit - only room for header + a few rows
        header_size = len("[Page 1]\nH1 | H2\n------\n")
        row_size = len("Row 0 | Val 0\n")
        limit = header_size + row_size * 3 + 5  # ~3 rows
        result, kept, total = extractor._truncate_tables_row_aware(table_text, limit)

        assert total == 10
        assert kept == 3

    @patch("src.extraction.llm_extractor.get_settings")
    def test_truncation_no_orphaned_page_markers(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """[Page N] markers should not appear without their header block."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        table1_lines = [
            "[Page 1]",
            "Name | Qty | Price",
            "----------",
            "Item A | 2 | $50.00",
            "Item B | 3 | $30.00",
            "",
        ]
        table2_lines = [
            "[Page 2]",
            "Name | Qty | Price",
            "----------",
            "Item C | 1 | $100.00",
        ]
        table_text = "\n".join(table1_lines + table2_lines)

        # Budget fits all of table 1 but NOT the [Page 2] header block
        budget = len("\n".join(table1_lines)) + len("[Page 2]") + 5
        result, kept, total = extractor._truncate_tables_row_aware(table_text, budget)

        # Should NOT contain orphaned [Page 2] without its header
        if "[Page 2]" in result:
            assert "Name | Qty | Price" in result.split("[Page 2]")[1]
        assert total == 3


# ============================================================================
# Column Pruning Tests
# ============================================================================


class TestColumnPruning:
    """Tests for _prune_non_financial_columns() method.

    Verifies that non-essential columns (notes, weight, warranty, etc.) are
    removed to reduce table size while preserving financial columns.
    """

    @patch("src.extraction.llm_extractor.get_settings")
    def test_prune_removes_non_essential_columns(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Non-essential columns should be removed from tables."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[
                    ["Widget A", "2", "$50.00", "In stock", "See manual"],
                    ["Widget B", "3", "$30.00", "Backordered", "N/A"],
                ],
                headers=["Description", "Qty", "Price", "Availability", "Notes"],
                bbox=None,
            )
        ]

        pruned = extractor._prune_non_financial_columns(tables)

        assert len(pruned) == 1
        assert pruned[0].headers == ["Description", "Qty", "Price"]
        assert pruned[0].rows[0] == ["Widget A", "2", "$50.00"]
        assert pruned[0].rows[1] == ["Widget B", "3", "$30.00"]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_prune_preserves_all_financial_columns(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Tables with only financial columns should not be modified."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["Widget A", "SKU-001", "2", "$50.00", "$100.00"]],
                headers=["Description", "SKU", "Qty", "Unit Price", "Total"],
                bbox=None,
            )
        ]

        pruned = extractor._prune_non_financial_columns(tables)

        assert pruned[0].headers == tables[0].headers
        assert pruned[0].rows == tables[0].rows

    @patch("src.extraction.llm_extractor.get_settings")
    def test_prune_skips_tables_without_headers(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Tables without headers should pass through unchanged."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "B", "C"]],
                headers=[],
                bbox=None,
            )
        ]

        pruned = extractor._prune_non_financial_columns(tables)

        assert pruned[0].rows == tables[0].rows

    @patch("src.extraction.llm_extractor.get_settings")
    def test_prune_guards_against_removing_all_columns(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """If all columns match non-essential, preserve the original table."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["Heavy", "Red", "See note"]],
                headers=["Weight", "Color", "Notes"],
                bbox=None,
            )
        ]

        pruned = extractor._prune_non_financial_columns(tables)

        # Should preserve original since pruning would remove everything
        assert pruned[0].headers == ["Weight", "Color", "Notes"]
        assert pruned[0].rows == [["Heavy", "Red", "See note"]]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_prune_case_insensitive_matching(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Column header matching should be case-insensitive."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["Widget A", "2", "$50.00", "2 lbs"]],
                headers=["Description", "Qty", "Price", "WEIGHT"],
                bbox=None,
            )
        ]

        pruned = extractor._prune_non_financial_columns(tables)

        assert "WEIGHT" not in pruned[0].headers
        assert len(pruned[0].headers) == 3


# ============================================================================
# Compact Formatting Tests
# ============================================================================


class TestCompactFormatting:
    """Tests for compact mode in _format_tables().

    Compact mode removes separator lines and blank lines between tables
    to reduce character count for large documents.
    """

    @patch("src.extraction.llm_extractor.get_settings")
    def test_compact_omits_separator_lines(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Compact mode should not include dash separator lines."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"]],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = extractor._format_tables(tables, compact=True)

        assert "---" not in result
        assert "[Page 1]" in result
        assert "Name | Qty" in result
        assert "A | 1" in result

    @patch("src.extraction.llm_extractor.get_settings")
    def test_compact_omits_blank_lines_between_tables(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Compact mode should not have blank lines between tables."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(page=1, rows=[["A", "1"]], headers=["Name", "Qty"], bbox=None),
            TableData(page=2, rows=[["B", "2"]], headers=["Name", "Qty"], bbox=None),
        ]

        result = extractor._format_tables(tables, compact=True)

        # Should not have consecutive newlines (blank line)
        assert "\n\n" not in result

    @patch("src.extraction.llm_extractor.get_settings")
    def test_compact_is_smaller_than_normal(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Compact output should use fewer characters than normal output."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"], ["C", "3"]],
                headers=["Name", "Qty"],
                bbox=None,
            ),
            TableData(
                page=2,
                rows=[["D", "4"], ["E", "5"]],
                headers=["Name", "Qty"],
                bbox=None,
            ),
        ]

        normal = extractor._format_tables(tables, compact=False)
        compact = extractor._format_tables(tables, compact=True)

        assert len(compact) < len(normal)

    @patch("src.extraction.llm_extractor.get_settings")
    def test_default_is_not_compact(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Default formatting should include separators and blank lines."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"]],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = extractor._format_tables(tables)

        assert "---" in result


# ============================================================================
# Separator Detection Tests
# ============================================================================


class TestSeparatorDetection:
    """Tests for stricter separator line detection in row-aware truncation.

    Ensures data rows starting with '-' (e.g., '- Discount') are NOT
    misclassified as separator lines.
    """

    @patch("src.extraction.llm_extractor.get_settings")
    def test_dash_data_row_not_treated_as_separator(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """A data row like '- Discount | 1 | -$50' should count as a data row."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        table_text = (
            "[Page 1]\n"
            "Name | Qty | Price\n"
            "-------------------\n"
            "Widget A | 2 | $50.00\n"
            "- Discount | 1 | -$10.00\n"
            "Widget B | 3 | $30.00"
        )

        result, kept, total = extractor._truncate_tables_row_aware(
            table_text, len(table_text) + 100
        )

        # All 3 data rows should be counted (including "- Discount")
        assert total == 3
        assert kept == 3

    @patch("src.extraction.llm_extractor.get_settings")
    def test_actual_separator_is_detected(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """A line of only dashes should be treated as separator, not data."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        table_text = "[Page 1]\nName | Qty\n----------\nA | 1\nB | 2"

        result, kept, total = extractor._truncate_tables_row_aware(
            table_text, len(table_text) + 100
        )

        # Only 2 data rows (separator is structural)
        assert total == 2
        assert kept == 2


# ============================================================================
# Anti-Deduplication Prompt Capture Tests
# ============================================================================


class TestAntiDeduplicationPrompt:
    """Tests verifying anti-dedup instructions appear in the formatted prompt.

    Uses a prompt-capturing mock provider to inspect the actual prompt
    text sent to the LLM, confirming row count and anti-dedup language.
    """

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_formatted_prompt_includes_row_count(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_tables,
    ):
        """Formatted prompt should include actual row count from tables."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        captured_prompts: list[str] = []

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )

        class CapturingProvider(MockProvider):
            async def extract_text(self, prompt: str, response_model=None):
                captured_prompts.append(prompt)
                return quote

        provider = CapturingProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[provider])

        await extractor.extract(
            text="Test document",
            tables=sample_tables,
            page_count=2,
            pdf_bytes=b"test",
        )

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        # sample_tables has 3 data rows (2 on page 1, 1 on page 2)
        assert "~3 data rows" in prompt
        assert "Do NOT merge or skip rows" in prompt

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_formatted_prompt_uses_unknown_when_no_tables(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
    ):
        """When no tables present, row_count should be 'unknown'."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        captured_prompts: list[str] = []

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )

        class CapturingProvider(MockProvider):
            async def extract_text(self, prompt: str, response_model=None):
                captured_prompts.append(prompt)
                return quote

        provider = CapturingProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[provider])

        await extractor.extract(
            text="Test document",
            tables=[],
            page_count=1,
            pdf_bytes=b"test",
        )

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "~unknown data rows" in prompt

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_truncated_prompt_uses_rows_kept_not_total(
        self,
        mock_get_settings,
        mock_settings_no_anthropic,
        sample_tables,
    ):
        """When tables are truncated, prompt should use rows_kept, not total_rows."""
        # Force truncation: sample_tables produce ~192 chars; limit to 120
        # so some data rows are kept but not all 3
        mock_settings_no_anthropic.max_table_chars = 120
        mock_settings_no_anthropic.provider_max_table_chars = {"openai": 120}
        mock_get_settings.return_value = mock_settings_no_anthropic

        captured_prompts: list[str] = []

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )

        class CapturingProvider(MockProvider):
            async def extract_text(self, prompt: str, response_model=None):
                captured_prompts.append(prompt)
                return quote

        provider = CapturingProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[provider])

        await extractor.extract(
            text="Test document",
            tables=sample_tables,
            page_count=2,
            pdf_bytes=b"test",
        )

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        # With 50-char limit, not all 3 rows fit; prompt should show rows_kept
        # which is less than total_rows (3). Must NOT show "~3 data rows".
        assert "data rows" in prompt
        # Extract the actual row count from the prompt
        import re

        match = re.search(r"~(\d+) data rows", prompt)
        assert match is not None, "Prompt should contain ~N data rows"
        rows_in_prompt = int(match.group(1))
        # rows_kept must be less than total (3) due to truncation
        assert rows_in_prompt < 3, (
            f"Expected rows_kept < 3 (total), got {rows_in_prompt}"
        )


# ============================================================================
# Chunk Table Rows Tests (Phase 3: H1)
# ============================================================================


class TestChunkTableRows:
    """Tests for _chunk_table_rows() method.

    Verifies that table rows are split into chunks with headers repeated
    and [RNN] tags preserved across chunk boundaries.
    """

    @patch("src.extraction.llm_extractor.get_settings")
    def test_single_chunk_when_under_limit(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Tables with fewer rows than chunk size produce one chunk."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1", "$10"], ["B", "2", "$20"]],
                headers=["Name", "Qty", "Price"],
                bbox=None,
            )
        ]

        chunks = extractor._chunk_table_rows(tables, rows_per_chunk=15)

        assert len(chunks) == 1
        assert "[R01]" in chunks[0]
        assert "[R02]" in chunks[0]
        assert "Name | Qty | Price" in chunks[0]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_multiple_chunks_split_correctly(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Rows exceeding chunk size are split into multiple chunks."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        rows = [[f"Item {i}", str(i), f"${i * 10}"] for i in range(1, 11)]
        tables = [
            TableData(page=1, rows=rows, headers=["Name", "Qty", "Price"], bbox=None)
        ]

        chunks = extractor._chunk_table_rows(tables, rows_per_chunk=3)

        # 10 rows / 3 per chunk = 4 chunks (3+3+3+1)
        assert len(chunks) == 4
        # First chunk has R01-R03
        assert "[R01]" in chunks[0]
        assert "[R03]" in chunks[0]
        assert "[R04]" not in chunks[0]
        # Last chunk has R10
        assert "[R10]" in chunks[3]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_headers_repeated_in_each_chunk(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Each chunk should include the table header for context."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        rows = [[f"Item {i}", str(i)] for i in range(1, 7)]
        tables = [TableData(page=1, rows=rows, headers=["Name", "Qty"], bbox=None)]

        chunks = extractor._chunk_table_rows(tables, rows_per_chunk=3)

        assert len(chunks) == 2
        for chunk in chunks:
            assert "Name | Qty" in chunk

    @patch("src.extraction.llm_extractor.get_settings")
    def test_rnn_tags_preserved_across_chunks(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """[RNN] tags should be sequential across chunks, not reset."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        rows = [[f"Item {i}", str(i)] for i in range(1, 7)]
        tables = [TableData(page=1, rows=rows, headers=["Name", "Qty"], bbox=None)]

        chunks = extractor._chunk_table_rows(tables, rows_per_chunk=3)

        # Second chunk should have R04, R05, R06 (not R01, R02, R03)
        assert "[R04]" in chunks[1]
        assert "[R05]" in chunks[1]
        assert "[R06]" in chunks[1]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_multi_table_chunking(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Rows from multiple tables are chunked together with correct tags."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"]],
                headers=["Name", "Qty"],
                bbox=None,
            ),
            TableData(
                page=2,
                rows=[["C", "3"], ["D", "4"]],
                headers=["Name", "Qty"],
                bbox=None,
            ),
        ]

        chunks = extractor._chunk_table_rows(tables, rows_per_chunk=3)

        # 4 rows total, chunk size 3 → 2 chunks
        assert len(chunks) == 2
        # First chunk: R01, R02 from page 1 + R03 from page 2
        assert "[R01]" in chunks[0]
        assert "[R02]" in chunks[0]
        assert "[R03]" in chunks[0]
        # Second chunk: R04 from page 2
        assert "[R04]" in chunks[1]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_empty_tables_produce_no_chunks(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Empty tables should produce no chunks."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        chunks = extractor._chunk_table_rows([], rows_per_chunk=15)

        assert len(chunks) == 0


# ============================================================================
# Merge Line Items Tests (Phase 3: H1)
# ============================================================================


class TestMergeLineItems:
    """Tests for _merge_line_items() method.

    Verifies deduplication by line_number and sort order.
    """

    @patch("src.extraction.llm_extractor.get_settings")
    def test_no_duplicates_returns_all(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Items with unique line_numbers are all preserved."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        items = [
            LineItem(
                line_number=1,
                description="A",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=2,
                description="B",
                quantity=1,
                unit_price=20,
                extended_price=20,
            ),
            LineItem(
                line_number=3,
                description="C",
                quantity=1,
                unit_price=30,
                extended_price=30,
            ),
        ]

        merged = extractor._merge_line_items(items)

        assert len(merged) == 3
        assert [i.line_number for i in merged] == [1, 2, 3]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_duplicates_keep_first(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Duplicate line_numbers keep the first occurrence."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        items = [
            LineItem(
                line_number=1,
                description="First",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=1,
                description="Duplicate",
                quantity=1,
                unit_price=20,
                extended_price=20,
            ),
            LineItem(
                line_number=2,
                description="Second",
                quantity=1,
                unit_price=30,
                extended_price=30,
            ),
        ]

        merged = extractor._merge_line_items(items)

        assert len(merged) == 2
        assert merged[0].description == "First"
        assert merged[1].line_number == 2

    @patch("src.extraction.llm_extractor.get_settings")
    def test_sorted_by_line_number(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Merged items are sorted by line_number."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        items = [
            LineItem(
                line_number=3,
                description="C",
                quantity=1,
                unit_price=30,
                extended_price=30,
            ),
            LineItem(
                line_number=1,
                description="A",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=2,
                description="B",
                quantity=1,
                unit_price=20,
                extended_price=20,
            ),
        ]

        merged = extractor._merge_line_items(items)

        assert [i.line_number for i in merged] == [1, 2, 3]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_empty_list(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Empty input produces empty output."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        merged = extractor._merge_line_items([])

        assert merged == []


# ============================================================================
# Build Table Summary Tests (Phase 3: H1)
# ============================================================================


class TestBuildTableSummary:
    """Tests for _build_table_summary() method."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_summary_with_headers(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Summary includes page number, row count, and column headers."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"]],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        summary = extractor._build_table_summary(tables, total_rows=2)

        assert "[Page 1]" in summary
        assert "2 rows" in summary
        assert "Name | Qty" in summary
        assert "Total data rows: 2" in summary

    @patch("src.extraction.llm_extractor.get_settings")
    def test_summary_without_headers(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Summary for headerless tables shows row count only."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [TableData(page=1, rows=[["A"], ["B"]], headers=[], bbox=None)]

        summary = extractor._build_table_summary(tables, total_rows=2)

        assert "[Page 1]" in summary
        assert "2 rows" in summary

    @patch("src.extraction.llm_extractor.get_settings")
    def test_summary_empty_tables(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Empty tables list produces default message."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        summary = extractor._build_table_summary([], total_rows=0)

        assert summary == "No tables detected."


# ============================================================================
# Extract Rows By Number Tests (Phase 4: H2)
# ============================================================================


class TestExtractRowsByNumber:
    """Tests for _extract_rows_by_number() method."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_extracts_specific_rows(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Extracts only the requested row numbers."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"], ["C", "3"]],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = extractor._extract_rows_by_number(tables, [2])

        assert "[R02]" in result
        assert "B | 2" in result
        assert "[R01]" not in result
        assert "[R03]" not in result

    @patch("src.extraction.llm_extractor.get_settings")
    def test_includes_header_context(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Extracted rows include their table header for context."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"]],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = extractor._extract_rows_by_number(tables, [1])

        assert "Name | Qty" in result
        assert "[Page 1]" in result

    @patch("src.extraction.llm_extractor.get_settings")
    def test_multi_table_row_extraction(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Rows from different tables are correctly numbered and extracted."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"]],
                headers=["Name", "Qty"],
                bbox=None,
            ),
            TableData(
                page=2,
                rows=[["C", "3"]],
                headers=["Name", "Qty"],
                bbox=None,
            ),
        ]

        # Row 3 is in the second table
        result = extractor._extract_rows_by_number(tables, [3])

        assert "[R03]" in result
        assert "C | 3" in result

    @patch("src.extraction.llm_extractor.get_settings")
    def test_no_matching_rows_returns_empty(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Requesting non-existent row numbers returns empty string."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(page=1, rows=[["A", "1"]], headers=["Name", "Qty"], bbox=None)
        ]

        result = extractor._extract_rows_by_number(tables, [99])

        assert result == ""


# ============================================================================
# Reconciliation Tests (Phase 4: H2)
# ============================================================================


class TestReconciliation:
    """Tests for _reconcile_missing_rows() method.

    Verifies that gap detection and re-extraction work correctly.
    """

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_no_reconciliation_when_all_rows_present(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """No reconciliation when extracted count >= 90% of expected."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=i,
                    description=f"Item {i}",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
                for i in range(1, 11)
            ],
            amounts=Amounts(subtotal=100, grand_total=100),
        )

        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1, rows=[["A"] for _ in range(10)], headers=["Name"], bbox=None
            )
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=10, provider=mock_provider
        )

        # No change — all rows present
        assert len(result.line_items) == 10

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_gap_detection_triggers_for_small_gaps(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Gap detection triggers reconciliation even for a single missing row."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        # 9 out of 10 — line_number 10 is missing
        existing_items = [
            LineItem(
                line_number=i,
                description=f"Item {i}",
                quantity=1,
                unit_price=10,
                extended_price=10,
            )
            for i in range(1, 10)  # 9 items, missing line 10
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=90, grand_total=90),
        )

        # Mock provider returns the missing item on reconciliation call
        missing_item = LineItem(
            line_number=10,
            description="Item 10",
            quantity=1,
            unit_price=10,
            extended_price=10,
        )
        recon_batch = LineItemBatch(line_items=[missing_item])

        def handler(prompt, response_model=None):
            return recon_batch

        mock_provider = MockProvider(
            name="openai",
            extract_text_handler=handler,
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1, rows=[["A"] for _ in range(10)], headers=["Name"], bbox=None
            )
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=10, provider=mock_provider
        )

        # Gap detection finds line 10 missing and re-extracts it
        assert len(result.line_items) == 10
        assert {i.line_number for i in result.line_items} == set(range(1, 11))

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_reconciliation_adds_missing_rows(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Missing rows are re-extracted and merged into the quote."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        # Only 7 out of 10 items — below 90% threshold
        existing_items = [
            LineItem(
                line_number=i,
                description=f"Item {i}",
                quantity=1,
                unit_price=10,
                extended_price=10,
            )
            for i in [1, 2, 3, 5, 6, 8, 10]
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=70, grand_total=70),
        )

        # Mock provider returns missing items on reconciliation call
        missing_items = [
            LineItem(
                line_number=4,
                description="Item 4",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=7,
                description="Item 7",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=9,
                description="Item 9",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
        ]
        recon_batch = LineItemBatch(line_items=missing_items)

        def handler(prompt, response_model=None):
            return recon_batch

        mock_provider = MockProvider(
            name="openai",
            extract_text_handler=handler,
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Row {i}"] for i in range(1, 11)],
                headers=["Name"],
                bbox=None,
            )
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=10, provider=mock_provider
        )

        assert len(result.line_items) == 10
        line_numbers = [i.line_number for i in result.line_items]
        assert line_numbers == list(range(1, 11))

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_reconciliation_handles_provider_error(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Reconciliation failure returns original quote unchanged."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        existing_items = [
            LineItem(
                line_number=i,
                description=f"Item {i}",
                quantity=1,
                unit_price=10,
                extended_price=10,
            )
            for i in [1, 2, 3]  # Only 3 of 10
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=30, grand_total=30),
        )

        # Provider raises on reconciliation
        mock_provider = MockProvider(
            name="openai",
            extract_text_error=Exception("Reconciliation failed"),
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Row {i}"] for i in range(1, 11)],
                headers=["Name"],
                bbox=None,
            )
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=10, provider=mock_provider
        )

        # Original quote returned unchanged
        assert len(result.line_items) == 3

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_no_reconciliation_when_zero_expected(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """No reconciliation when expected_rows is 0."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="A",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
            ],
            amounts=Amounts(subtotal=10, grand_total=10),
        )

        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        result = await extractor._reconcile_missing_rows(
            quote, [], expected_rows=0, provider=mock_provider
        )

        assert len(result.line_items) == 1


# ============================================================================
# Chunked Extraction Flow Tests (Phase 3: H1)
# ============================================================================


class TestChunkedExtraction:
    """Tests for the full chunked extraction flow via extract()."""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_chunking_activates_above_threshold(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Chunked extraction activates when total_rows > chunking_min_rows."""
        mock_settings_no_anthropic.chunking_min_rows = 5  # Low threshold for test
        mock_settings_no_anthropic.provider_single_pass_max_rows = {"openai": 5}
        mock_get_settings.return_value = mock_settings_no_anthropic

        header = VendorQuoteHeader(
            vendor=Entity(name="Test Vendor"),
            amounts=Amounts(subtotal=100, grand_total=100),
        )
        batch = LineItemBatch(
            line_items=[
                LineItem(
                    line_number=i,
                    description=f"Item {i}",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
                for i in range(1, 4)
            ]
        )

        call_count = {"header": 0, "chunk": 0}

        def handler(prompt, response_model=None):
            if response_model is VendorQuoteHeader:
                call_count["header"] += 1
                return header
            call_count["chunk"] += 1
            return batch

        mock_provider = MockProvider(name="openai", extract_text_handler=handler)
        extractor = LLMExtractor(providers=[mock_provider])

        # 6 rows > 5 threshold → chunked extraction
        tables = [
            TableData(
                page=1,
                rows=[[f"Row {i}", str(i)] for i in range(1, 7)],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is not None
        assert call_count["header"] == 1
        assert call_count["chunk"] >= 1

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_single_pass_below_threshold(
        self, mock_get_settings, mock_settings_no_anthropic, sample_quote
    ):
        """Single-pass extraction used when total_rows <= chunking_min_rows."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        mock_provider = MockProvider(name="openai", extract_text_result=sample_quote)
        extractor = LLMExtractor(providers=[mock_provider])

        # 3 rows << 30 threshold → single pass
        tables = [
            TableData(
                page=1,
                rows=[["A", "1"], ["B", "2"], ["C", "3"]],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is not None
        assert result.provider_used == "openai"

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_chunked_extraction_merges_line_items(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Chunked extraction correctly merges items from multiple chunks."""
        mock_settings_no_anthropic.chunking_min_rows = 3  # Low threshold
        mock_settings_no_anthropic.chunk_rows_per_call = 2
        mock_settings_no_anthropic.provider_single_pass_max_rows = {"openai": 3}
        mock_get_settings.return_value = mock_settings_no_anthropic

        header = VendorQuoteHeader(
            vendor=Entity(name="Test Vendor"),
            amounts=Amounts(subtotal=60, grand_total=60),
        )

        def handler(prompt, response_model=None):
            if response_model is VendorQuoteHeader:
                return header
            if response_model is LineItemBatch:
                # Parse [RNN] tags from table rows only (lines starting with [R)
                tags = re.findall(r"(?m)^\[R(\d+)\]", prompt)
                items = []
                for tag in tags:
                    n = int(tag)
                    items.append(
                        LineItem(
                            line_number=n,
                            description=f"Item-{chr(64 + n)}",
                            quantity=1,
                            unit_price=10,
                            extended_price=10,
                        )
                    )
                return LineItemBatch(line_items=items)
            raise RuntimeError("Unexpected call")

        mock_provider = MockProvider(name="openai", extract_text_handler=handler)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Item {i}", str(i)] for i in range(1, 7)],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is not None
        assert len(result.quote.line_items) == 6
        line_numbers = sorted([i.line_number for i in result.quote.line_items])
        assert line_numbers == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_chunked_extraction_handles_chunk_failure(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Partial chunk failure still produces results from successful chunks."""
        mock_settings_no_anthropic.chunking_min_rows = 3
        mock_settings_no_anthropic.chunk_rows_per_call = 2
        mock_settings_no_anthropic.provider_single_pass_max_rows = {"openai": 3}
        mock_get_settings.return_value = mock_settings_no_anthropic

        header = VendorQuoteHeader(
            vendor=Entity(name="Test Vendor"),
            amounts=Amounts(subtotal=40, grand_total=40),
        )

        def handler(prompt, response_model=None):
            if response_model is VendorQuoteHeader:
                return header
            if response_model is LineItemBatch:
                # Parse [RNN] tags from table rows only (lines starting with [R)
                tags = re.findall(r"(?m)^\[R(\d+)\]", prompt)
                # Fail for chunks containing R03 or R04 (second chunk)
                if any(int(t) > 2 for t in tags):
                    raise Exception("Chunk extraction failed")
                items = []
                for tag in tags:
                    n = int(tag)
                    items.append(
                        LineItem(
                            line_number=n,
                            description=f"Item-{chr(64 + n)}",
                            quantity=1,
                            unit_price=10,
                            extended_price=10,
                        )
                    )
                return LineItemBatch(line_items=items)
            raise RuntimeError("Unexpected call")

        mock_provider = MockProvider(name="openai", extract_text_handler=handler)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Item {i}", str(i)] for i in range(1, 5)],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        # Should still have items from the successful chunk
        assert result.quote is not None
        assert len(result.quote.line_items) >= 2

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_chunked_header_failure_returns_error(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Header extraction failure in chunked mode returns error result."""
        mock_settings_no_anthropic.chunking_min_rows = 3
        mock_settings_no_anthropic.provider_single_pass_max_rows = {"openai": 3}
        mock_get_settings.return_value = mock_settings_no_anthropic

        mock_provider = MockProvider(
            name="openai",
            extract_text_error=Exception("Header extraction failed"),
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Item {i}"] for i in range(1, 5)],
                headers=["Name"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is None
        assert result.error is not None
        assert "Header extraction failed" in result.error

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_parallel_chunk_execution_respects_semaphore(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Semaphore limits concurrent chunk extraction calls."""
        import asyncio

        mock_settings_no_anthropic.chunking_min_rows = 3
        mock_settings_no_anthropic.chunk_rows_per_call = 2
        mock_settings_no_anthropic.chunk_parallel_concurrency = 2
        mock_settings_no_anthropic.provider_single_pass_max_rows = {
            "groq": 30,
            "openai": 3,
            "gemini": 150,
            "anthropic": 100,
            "fireworks": 50,
        }
        mock_get_settings.return_value = mock_settings_no_anthropic

        header = VendorQuoteHeader(
            vendor=Entity(name="Test Vendor"),
            amounts=Amounts(subtotal=60, grand_total=60),
        )

        max_concurrent = {"val": 0}
        current_concurrent = {"val": 0}

        async def async_handler(prompt, response_model=None):
            if response_model is VendorQuoteHeader:
                return header
            if response_model is LineItemBatch:
                current_concurrent["val"] += 1
                max_concurrent["val"] = max(
                    max_concurrent["val"], current_concurrent["val"]
                )
                await asyncio.sleep(0.01)  # Simulate async work
                current_concurrent["val"] -= 1
                tags = re.findall(r"(?m)^\[R(\d+)\]", prompt)
                items = []
                for tag in tags:
                    n = int(tag)
                    items.append(
                        LineItem(
                            line_number=n,
                            description=f"Item-{n}",
                            quantity=1,
                            unit_price=10,
                            extended_price=10,
                        )
                    )
                return LineItemBatch(line_items=items)
            raise RuntimeError("Unexpected call")

        mock_provider = MockAsyncProvider(
            name="openai", extract_text_handler=async_handler
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Item {i}", str(i)] for i in range(1, 9)],  # 8 rows -> 4 chunks
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is not None
        # Semaphore=2 should limit concurrent chunks to at most 2
        assert max_concurrent["val"] <= 2

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_provider_aware_bypass_skips_chunking(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """High single-pass limits should bypass chunking for moderate row counts."""
        mock_settings_no_anthropic.chunking_min_rows = 5
        mock_settings_no_anthropic.provider_single_pass_max_rows = {
            "openai": 100,  # OpenAI can handle 100 rows single-pass
        }
        mock_get_settings.return_value = mock_settings_no_anthropic

        # Build a quote with 10 items matching the 10 rows to avoid reconciliation
        full_quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=i,
                    description=f"Item {i}",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
                for i in range(1, 11)
            ],
            amounts=Amounts(subtotal=100, grand_total=100),
        )

        call_count = {"total": 0}

        def handler(prompt, response_model=None):
            call_count["total"] += 1
            return full_quote  # Single-pass returns full quote

        mock_provider = MockProvider(name="openai", extract_text_handler=handler)
        extractor = LLMExtractor(providers=[mock_provider])

        # 10 rows > chunking_min_rows(5) but <= provider limit(100) -> single-pass
        tables = [
            TableData(
                page=1,
                rows=[[f"Item {i}", str(i)] for i in range(1, 11)],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is not None
        # Single-pass = exactly 1 LLM call (no header+chunks split)
        assert call_count["total"] == 1

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_provider_aware_bypass_still_chunks_small_context(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Provider with low single-pass limit still uses chunking for large docs."""
        mock_settings_no_anthropic.chunking_min_rows = 5
        mock_settings_no_anthropic.chunk_rows_per_call = 5
        mock_settings_no_anthropic.provider_single_pass_max_rows = {
            "groq": 30,  # Groq can only handle 30 rows single-pass
        }
        mock_get_settings.return_value = mock_settings_no_anthropic

        header = VendorQuoteHeader(
            vendor=Entity(name="Test Vendor"),
            amounts=Amounts(subtotal=100, grand_total=100),
        )

        call_count = {"header": 0, "chunk": 0}

        def handler(prompt, response_model=None):
            if response_model is VendorQuoteHeader:
                call_count["header"] += 1
                return header
            if response_model is LineItemBatch:
                call_count["chunk"] += 1
                tags = re.findall(r"(?m)^\[R(\d+)\]", prompt)
                items = []
                for tag in tags:
                    n = int(tag)
                    items.append(
                        LineItem(
                            line_number=n,
                            description=f"Item-{n}",
                            quantity=1,
                            unit_price=10,
                            extended_price=10,
                        )
                    )
                return LineItemBatch(line_items=items)
            raise RuntimeError("Unexpected call")

        mock_provider = MockProvider(name="groq", extract_text_handler=handler)
        extractor = LLMExtractor(providers=[mock_provider])

        # 50 rows > groq single-pass limit(30) -> chunking still used
        tables = [
            TableData(
                page=1,
                rows=[[f"Item {i}", str(i)] for i in range(1, 51)],
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is not None
        assert call_count["header"] == 1
        assert call_count["chunk"] >= 2  # Should have multiple chunks

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_parallel_chunk_failure_does_not_affect_others(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """One failed chunk should not block successful sibling chunks."""
        import asyncio

        mock_settings_no_anthropic.chunking_min_rows = 3
        mock_settings_no_anthropic.chunk_rows_per_call = 2
        mock_settings_no_anthropic.chunk_parallel_concurrency = 3
        mock_settings_no_anthropic.provider_single_pass_max_rows = {
            "openai": 3,
        }
        mock_get_settings.return_value = mock_settings_no_anthropic

        header = VendorQuoteHeader(
            vendor=Entity(name="Test Vendor"),
            amounts=Amounts(subtotal=60, grand_total=60),
        )

        async def async_handler(prompt, response_model=None):
            if response_model is VendorQuoteHeader:
                return header
            if response_model is LineItemBatch:
                tags = re.findall(r"(?m)^\[R(\d+)\]", prompt)
                # Fail chunks containing R03 or R04
                if any(int(t) in (3, 4) for t in tags):
                    await asyncio.sleep(0.01)
                    raise Exception("Simulated chunk failure")
                await asyncio.sleep(0.01)
                items = []
                for tag in tags:
                    n = int(tag)
                    items.append(
                        LineItem(
                            line_number=n,
                            description=f"Item-{n}",
                            quantity=1,
                            unit_price=10,
                            extended_price=10,
                        )
                    )
                return LineItemBatch(line_items=items)
            raise RuntimeError("Unexpected call")

        mock_provider = MockAsyncProvider(
            name="openai", extract_text_handler=async_handler
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Item {i}", str(i)] for i in range(1, 7)],  # 6 rows -> 3 chunks
                headers=["Name", "Qty"],
                bbox=None,
            )
        ]

        result = await extractor.extract(
            text="Test doc", tables=tables, page_count=1, pdf_bytes=b"test"
        )

        assert result.quote is not None
        # Chunks 1 (R01,R02) and 3 (R05,R06) succeed; chunk 2 (R03,R04) fails
        assert len(result.quote.line_items) >= 4
        descriptions = [i.description for i in result.quote.line_items]
        assert "Item-1" in descriptions
        assert "Item-2" in descriptions
        assert "Item-5" in descriptions
        assert "Item-6" in descriptions


# ============================================================================
# RNN Tag Rule in Prompt Tests
# ============================================================================


class TestRNNTagRule:
    """Tests verifying [RNN] tag alignment rule appears in prompts."""

    def test_extraction_prompt_has_rnn_rule(self):
        """EXTRACTION_PROMPT contains the line_number = [RNN] tag rule."""
        assert "line_number MUST equal the [RNN] tag number" in EXTRACTION_PROMPT
        assert "[R07]" in EXTRACTION_PROMPT
        assert "gap detection" in EXTRACTION_PROMPT

    def test_groq_system_prompt_has_rnn_rule(self):
        """Groq system prompt contains the [RNN] alignment rule."""
        from src.extraction.providers.groq_provider import GROQ_SYSTEM_PROMPT

        assert "line_number MUST equal the [RNN] tag number" in GROQ_SYSTEM_PROMPT
        assert "gap detection" in GROQ_SYSTEM_PROMPT


# ============================================================================
# Edge Case Tests (from RepoPrompt review)
# ============================================================================


class TestHeaderlessTableChunking:
    """Tests for headerless tables in _chunk_table_rows — ensures no header leakage."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_headerless_table_does_not_inherit_previous_header(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """A headerless table should NOT inherit the previous table's column headers."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="X",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
            ],
            amounts=Amounts(subtotal=10, grand_total=10),
        )
        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1, rows=[["A", "100"]], headers=["Name", "Price"], bbox=None
            ),
            TableData(page=2, rows=[["B", "200"]], headers=None, bbox=None),
        ]

        chunks = extractor._chunk_table_rows(tables, rows_per_chunk=50)
        assert len(chunks) == 1
        chunk = chunks[0]

        # The headerless table should get a page marker but NOT "Name | Price"
        lines = chunk.strip().split("\n")
        # First table: [Page 1] header, then [R01] row
        # Second table: [Page 2] (no column header), then [R02] row
        page2_idx = next(i for i, line in enumerate(lines) if "[Page 2]" in line)
        # The line at page2_idx should be just "[Page 2]".
        # It must not include the column header string.
        assert "Name | Price" not in lines[page2_idx]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_headerless_table_in_extract_rows_by_number(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """_extract_rows_by_number emits page marker for headerless tables."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="X",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
            ],
            amounts=Amounts(subtotal=10, grand_total=10),
        )
        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(page=1, rows=[["A"]], headers=["Col"], bbox=None),
            TableData(page=2, rows=[["B"]], headers=None, bbox=None),
        ]

        result = extractor._extract_rows_by_number(tables, [2])
        assert "[Page 2]" in result
        assert "Col" not in result  # Should NOT leak table 1's header


class TestExtractRowsHeaderOncePerTable:
    """Tests that _extract_rows_by_number emits headers only once per table."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_header_emitted_once_for_multiple_rows_same_table(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Multiple missing rows from same table should only emit header once."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="X",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
            ],
            amounts=Amounts(subtotal=10, grand_total=10),
        )
        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["A"], ["B"], ["C"], ["D"], ["E"]],
                headers=["Name"],
                bbox=None,
            ),
        ]

        result = extractor._extract_rows_by_number(tables, [1, 3, 5])
        # Header "Name" should appear exactly once
        assert result.count("Name") == 1
        # All three rows should be present
        assert "[R01]" in result
        assert "[R03]" in result
        assert "[R05]" in result


class TestChunkRowsPerCallGuard:
    """Tests for invalid chunk_rows_per_call values."""

    @patch("src.extraction.llm_extractor.get_settings")
    def test_zero_rows_per_chunk_raises(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """chunk_rows_per_call=0 should raise ValueError."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="X",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
            ],
            amounts=Amounts(subtotal=10, grand_total=10),
        )
        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(page=1, rows=[["A"]], headers=["Name"], bbox=None),
        ]

        with pytest.raises(ValueError, match="rows_per_chunk must be > 0"):
            extractor._chunk_table_rows(tables, rows_per_chunk=0)

    @patch("src.extraction.llm_extractor.get_settings")
    def test_negative_rows_per_chunk_raises(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """chunk_rows_per_call=-5 should raise ValueError."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="X",
                    quantity=1,
                    unit_price=10,
                    extended_price=10,
                )
            ],
            amounts=Amounts(subtotal=10, grand_total=10),
        )
        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(page=1, rows=[["A"]], headers=["Name"], bbox=None),
        ]

        with pytest.raises(ValueError, match="rows_per_chunk must be > 0"):
            extractor._chunk_table_rows(tables, rows_per_chunk=-5)


class TestReconciliationThresholdSmallCounts:
    """Tests for reconciliation threshold behavior with small expected_rows."""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_small_expected_triggers_reconciliation_with_ceil(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """With math.ceil, 1 out of 2 expected rows triggers reconciliation."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        # 1 out of 2 items — math.ceil(2 * 0.90) = 2, so 1 < 2 triggers reconciliation
        existing_items = [
            LineItem(
                line_number=1,
                description="Item 1",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=10, grand_total=10),
        )

        missing_item = LineItem(
            line_number=2,
            description="Item 2",
            quantity=1,
            unit_price=20,
            extended_price=20,
        )
        recon_batch = LineItemBatch(line_items=[missing_item])

        def handler(prompt, response_model=None):
            return recon_batch

        mock_provider = MockProvider(
            name="openai",
            extract_text_handler=handler,
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[["Row 1"], ["Row 2"]],
                headers=["Name"],
                bbox=None,
            )
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=2, provider=mock_provider
        )

        # Reconciliation should have added the missing row
        assert len(result.line_items) == 2
        assert {i.line_number for i in result.line_items} == {1, 2}

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_single_row_expected_no_reconciliation_needed(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """With 1 expected and 1 extracted, no reconciliation needed."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        existing_items = [
            LineItem(
                line_number=1,
                description="Item 1",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=10, grand_total=10),
        )

        mock_provider = MockProvider(name="openai", extract_text_result=quote)
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [TableData(page=1, rows=[["Row 1"]], headers=["Name"], bbox=None)]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=1, provider=mock_provider
        )

        assert len(result.line_items) == 1


# ============================================================================
# Summary Table Filtering Tests
# ============================================================================


class TestIsSummaryTable:
    """Test the _is_summary_table heuristic."""

    def _make_extractor(self, mock_settings):
        """Helper to create an LLMExtractor for testing."""
        with patch(
            "src.extraction.llm_extractor.get_settings", return_value=mock_settings
        ):
            mock_provider = MockProvider(name="openai", extract_text_result=None)
            return LLMExtractor(providers=[mock_provider])

    @patch("src.extraction.llm_extractor.get_settings")
    def test_detects_grand_total_table(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """A small table with 'grand total' in cells is a summary table."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[["QUOTE # PTCH247", "$314,432.95"]],
            headers=["Quote Number", "Grand Total"],
            bbox=None,
        )
        assert extractor._is_summary_table(table) is True

    @patch("src.extraction.llm_extractor.get_settings")
    def test_detects_subtotal_in_rows(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """A small table with 'subtotal' in row content is summary."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=8,
            rows=[
                ["Subtotal", "$280,000.00"],
                ["Tax", "$34,432.95"],
                ["Grand Total", "$314,432.95"],
            ],
            headers=None,
            bbox=None,
        )
        assert extractor._is_summary_table(table) is True

    @patch("src.extraction.llm_extractor.get_settings")
    def test_does_not_filter_large_table(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Tables with >3 rows are never classified as summary."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        # A 10-row table even with a "total" header should pass through
        rows = [[f"Item {i}", "100.00"] for i in range(10)]
        table = TableData(
            page=1,
            rows=rows,
            headers=["Description", "Grand Total"],
            bbox=None,
        )
        assert extractor._is_summary_table(table) is False

    @patch("src.extraction.llm_extractor.get_settings")
    def test_does_not_filter_small_line_item_table(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """A small table without summary keywords is NOT summary."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[
                ["Cisco Switch 9300", "5", "$4,500.00", "$22,500.00"],
                ["SFP Module", "10", "$350.00", "$3,500.00"],
            ],
            headers=["Description", "Qty", "Unit Price", "Extended Price"],
            bbox=None,
        )
        assert extractor._is_summary_table(table) is False

    @patch("src.extraction.llm_extractor.get_settings")
    def test_detects_quote_number_table(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """A small table with 'quote #' is summary."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[["PTCH247", "10/03/2024", "Active"]],
            headers=["Quote #", "Date", "Status"],
            bbox=None,
        )
        assert extractor._is_summary_table(table) is True

    @patch("src.extraction.llm_extractor.get_settings")
    def test_detects_payment_terms_table(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """A small table with 'payment terms' is summary."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[["Net 30", "Wire Transfer"]],
            headers=["Payment Terms", "Method"],
            bbox=None,
        )
        assert extractor._is_summary_table(table) is True

    @patch("src.extraction.llm_extractor.get_settings")
    def test_detects_bill_to_table(self, mock_get_settings, mock_settings_no_anthropic):
        """A small table with 'bill to' is summary."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[["Acme Corp", "123 Main St"]],
            headers=["Bill To", "Address"],
            bbox=None,
        )
        assert extractor._is_summary_table(table) is True


class TestFilterLineItemTables:
    """Test _filter_line_item_tables partitioning."""

    def _make_extractor(self, mock_settings):
        with patch(
            "src.extraction.llm_extractor.get_settings", return_value=mock_settings
        ):
            mock_provider = MockProvider(name="openai", extract_text_result=None)
            return LLMExtractor(providers=[mock_provider])

    @patch("src.extraction.llm_extractor.get_settings")
    def test_separates_summary_from_line_items(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Summary and line-item tables are correctly partitioned."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        summary_table = TableData(
            page=1,
            rows=[["PTCH247", "$314,432.95"]],
            headers=["Quote #", "Grand Total"],
            bbox=None,
        )
        line_item_table = TableData(
            page=2,
            rows=[[f"Item {i}", "1", "100.00", "100.00"] for i in range(20)],
            headers=["Description", "Qty", "Unit Price", "Extended"],
            bbox=None,
        )

        li_tables, sum_tables = extractor._filter_line_item_tables(
            [summary_table, line_item_table]
        )

        assert len(li_tables) == 1
        assert len(sum_tables) == 1
        assert li_tables[0].page == 2
        assert sum_tables[0].page == 1

    @patch("src.extraction.llm_extractor.get_settings")
    def test_no_summary_tables(self, mock_get_settings, mock_settings_no_anthropic):
        """When all tables are line items, nothing is filtered."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[[f"Product {i}", "1", "50.00"] for i in range(15)],
            headers=["Description", "Qty", "Price"],
            bbox=None,
        )

        li_tables, sum_tables = extractor._filter_line_item_tables([table])

        assert len(li_tables) == 1
        assert len(sum_tables) == 0

    @patch("src.extraction.llm_extractor.get_settings")
    def test_empty_table_list(self, mock_get_settings, mock_settings_no_anthropic):
        """Empty input returns empty outputs."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        li_tables, sum_tables = extractor._filter_line_item_tables([])

        assert len(li_tables) == 0
        assert len(sum_tables) == 0

    @patch("src.extraction.llm_extractor.get_settings")
    def test_multiple_summary_tables_filtered(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Multiple summary tables are all filtered."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        tables = [
            TableData(
                page=1,
                rows=[["PTCH247", "Active"]],
                headers=["Quote #", "Status"],
                bbox=None,
            ),
            TableData(
                page=1,
                rows=[["Acme Corp"]],
                headers=["Bill To"],
                bbox=None,
            ),
            TableData(
                page=2,
                rows=[[f"Item {i}", "100.00"] for i in range(25)],
                headers=["Description", "Price"],
                bbox=None,
            ),
            TableData(
                page=8,
                rows=[["$314,432.95"]],
                headers=["Grand Total"],
                bbox=None,
            ),
        ]

        li_tables, sum_tables = extractor._filter_line_item_tables(tables)

        assert len(li_tables) == 1
        assert li_tables[0].page == 2
        assert len(sum_tables) == 3


# ============================================================================
# Table Preprocessing Tests
# ============================================================================


class TestIsRepeatedHeaderRow:
    """Test detection of repeated column headers as data rows."""

    def test_exact_match(self):
        """Row matching headers exactly is detected."""
        assert LLMExtractor._is_repeated_header_row(
            ["ITEM", "QTY", "SKU#", "UNIT PRICE", "EXT. PRICE"],
            ["ITEM", "QTY", "SKU#", "UNIT PRICE", "EXT. PRICE"],
        )

    def test_case_insensitive(self):
        """Match is case-insensitive."""
        assert LLMExtractor._is_repeated_header_row(
            ["item", "qty", "sku#", "unit price", "ext. price"],
            ["ITEM", "QTY", "SKU#", "UNIT PRICE", "EXT. PRICE"],
        )

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace is ignored."""
        assert LLMExtractor._is_repeated_header_row(
            ["  ITEM ", " QTY"],
            ["ITEM", "QTY"],
        )

    def test_different_content_not_matched(self):
        """Real data row is not matched as header."""
        assert not LLMExtractor._is_repeated_header_row(
            ["Cisco Switch 9300", "3", "C9300-48P-E", "$6,271.22", "$18,813.66"],
            ["ITEM", "QTY", "SKU#", "UNIT PRICE", "EXT. PRICE"],
        )

    def test_different_length_not_matched(self):
        """Row with different column count is not matched."""
        assert not LLMExtractor._is_repeated_header_row(
            ["ITEM", "QTY"],
            ["ITEM", "QTY", "SKU#"],
        )


class TestStripTaxLines:
    """Test stripping embedded per-item tax lines from cell text."""

    def _make_extractor(self, mock_settings):
        with patch(
            "src.extraction.llm_extractor.get_settings", return_value=mock_settings
        ):
            mock_provider = MockProvider(name="openai", extract_text_result=None)
            return LLMExtractor(providers=[mock_provider])

    @patch("src.extraction.llm_extractor.get_settings")
    def test_strips_tax_line_from_multiline_cell(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Embedded TAX: line is removed from multi-line description."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        cell = (
            "Cisco - power supply - redundant - 650 Watt\n"
            "Mfg. Part#: C9K-PWR-650WAC-R/2\n"
            "UNSPSC: 39121004\n"
            "TAX: SANTA MONICA, CA TAX: 10.2500% $341.23\n"
            "Contract: Standard Pricing"
        )
        cleaned, count = extractor._strip_tax_lines(cell)
        assert count == 1
        assert "341.23" not in cleaned
        assert "Cisco - power supply" in cleaned
        assert "Contract: Standard Pricing" in cleaned

    @patch("src.extraction.llm_extractor.get_settings")
    def test_strips_zero_tax_line(self, mock_get_settings, mock_settings_no_anthropic):
        """TAX lines with 0.0000% and $.00 are also stripped."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        cell = (
            "Cisco Smart Net Total Care\n"
            "TAX: SANTA MONICA, CA .0000% $.00\n"
            "Contract: Standard Pricing"
        )
        cleaned, count = extractor._strip_tax_lines(cell)
        assert count == 1
        assert "TAX:" not in cleaned

    @patch("src.extraction.llm_extractor.get_settings")
    def test_no_tax_lines_unchanged(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Cell without TAX: lines is returned unchanged."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        cell = "Cisco Catalyst 9300 - switch - 48 ports"
        cleaned, count = extractor._strip_tax_lines(cell)
        assert count == 0
        assert cleaned == cell

    @patch("src.extraction.llm_extractor.get_settings")
    def test_single_line_tax_cell(self, mock_get_settings, mock_settings_no_anthropic):
        """A cell that is entirely a TAX line becomes empty."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        cell = "TAX: SANTA MONICA, CA TAX: 10.2500% $341.23"
        cleaned, count = extractor._strip_tax_lines(cell)
        assert count == 1
        assert cleaned == ""


class TestPreprocessTables:
    """Test full table preprocessing pipeline."""

    def _make_extractor(self, mock_settings):
        with patch(
            "src.extraction.llm_extractor.get_settings", return_value=mock_settings
        ):
            mock_provider = MockProvider(name="openai", extract_text_result=None)
            return LLMExtractor(providers=[mock_provider])

    @patch("src.extraction.llm_extractor.get_settings")
    def test_removes_repeated_header_rows(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Repeated column headers at page breaks are removed."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[
                ["Cisco Switch", "2", "ABC123", "$1,000.00", "$2,000.00"],
                ["ITEM", "QTY", "SKU#", "UNIT PRICE", "EXT. PRICE"],  # repeated header
                ["Cisco Router", "1", "DEF456", "$5,000.00", "$5,000.00"],
            ],
            headers=["ITEM", "QTY", "SKU#", "UNIT PRICE", "EXT. PRICE"],
            bbox=None,
        )

        result = extractor._preprocess_tables([table])
        assert len(result) == 1
        assert len(result[0].rows) == 2  # header row removed
        assert result[0].rows[0][0] == "Cisco Switch"
        assert result[0].rows[1][0] == "Cisco Router"

    @patch("src.extraction.llm_extractor.get_settings")
    def test_strips_embedded_tax_from_cells(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Embedded TAX: lines are stripped from cell text."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=2,
            rows=[
                [
                    "Cisco power supply\n"
                    "Mfg. Part#: C9K-PWR\n"
                    "TAX: SANTA MONICA, CA TAX: 10.2500% $341.23\n"
                    "Contract: Standard Pricing",
                    "2",
                    "5071935",
                    "$1,664.54",
                    "$3,329.08",
                ],
            ],
            headers=["ITEM", "QTY", "SKU#", "UNIT PRICE", "EXT. PRICE"],
            bbox=None,
        )

        result = extractor._preprocess_tables([table])
        assert len(result[0].rows) == 1
        assert "341.23" not in result[0].rows[0][0]
        assert "Cisco power supply" in result[0].rows[0][0]

    @patch("src.extraction.llm_extractor.get_settings")
    def test_both_cleanups_applied(self, mock_get_settings, mock_settings_no_anthropic):
        """Both header removal and tax stripping work together."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        table = TableData(
            page=1,
            rows=[
                ["ITEM", "QTY", "PRICE"],  # repeated header
                [
                    "Product A\nTAX: CA TAX: 10% $50.00",
                    "1",
                    "$500.00",
                ],
                ["Product B", "2", "$300.00"],
            ],
            headers=["ITEM", "QTY", "PRICE"],
            bbox=None,
        )

        result = extractor._preprocess_tables([table])
        assert len(result[0].rows) == 2  # header row removed
        assert "TAX:" not in result[0].rows[0][0]  # tax stripped
        assert "Product A" in result[0].rows[0][0]
        assert result[0].rows[1][0] == "Product B"  # unchanged


# ============================================================================
# Gap-Based Reconciliation Tests
# ============================================================================


class TestGapBasedReconciliation:
    """Tests for always-on gap detection in _reconcile_missing_rows()."""

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_reconciliation_triggers_for_any_gap(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """55/58 items triggers reconciliation for the 3 missing rows."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        # 55 items present, missing line_numbers 33, 34, 35
        present_numbers = [i for i in range(1, 59) if i not in (33, 34, 35)]
        existing_items = [
            LineItem(
                line_number=i,
                description=f"Item {i}",
                quantity=1,
                unit_price=100,
                extended_price=100,
            )
            for i in present_numbers
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=5500, grand_total=5500),
        )

        # Mock provider returns the 3 missing items
        missing_items = [
            LineItem(
                line_number=n,
                description=f"Item {n}",
                quantity=1,
                unit_price=100,
                extended_price=100,
            )
            for n in [33, 34, 35]
        ]
        recon_batch = LineItemBatch(line_items=missing_items)

        def handler(prompt, response_model=None):
            return recon_batch

        mock_provider = MockProvider(
            name="openai",
            extract_text_handler=handler,
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Row {i}"] for i in range(1, 59)],
                headers=["Name"],
                bbox=None,
            )
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=58, provider=mock_provider
        )

        assert len(result.line_items) == 58
        assert {i.line_number for i in result.line_items} == set(range(1, 59))

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_no_reconciliation_when_all_rows_present_full(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """58/58 items — no gaps, no LLM call."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        existing_items = [
            LineItem(
                line_number=i,
                description=f"Item {i}",
                quantity=1,
                unit_price=100,
                extended_price=100,
            )
            for i in range(1, 59)
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=5800, grand_total=5800),
        )

        # Provider should NOT be called — error if it is
        mock_provider = MockProvider(
            name="openai",
            extract_text_error=Exception("Should not be called"),
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(
                page=1,
                rows=[[f"Row {i}"] for i in range(1, 59)],
                headers=["Name"],
                bbox=None,
            )
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=58, provider=mock_provider
        )

        assert len(result.line_items) == 58

    @pytest.mark.asyncio
    @patch("src.extraction.llm_extractor.get_settings")
    async def test_reconciliation_filters_invalid_line_numbers(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """Items with line_number > expected_rows don't create false gaps."""
        mock_get_settings.return_value = mock_settings_no_anthropic

        # 3 valid items + 1 with line_number=99 (> expected_rows=3)
        existing_items = [
            LineItem(
                line_number=1,
                description="Item 1",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=2,
                description="Item 2",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=3,
                description="Item 3",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
            LineItem(
                line_number=99,
                description="Mislabeled",
                quantity=1,
                unit_price=10,
                extended_price=10,
            ),
        ]
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test"),
            line_items=existing_items,
            amounts=Amounts(subtotal=40, grand_total=40),
        )

        # Provider should NOT be called — no real gaps
        mock_provider = MockProvider(
            name="openai",
            extract_text_error=Exception("Should not be called"),
        )
        extractor = LLMExtractor(providers=[mock_provider])

        tables = [
            TableData(page=1, rows=[["A"], ["B"], ["C"]], headers=["Name"], bbox=None)
        ]

        result = await extractor._reconcile_missing_rows(
            quote, tables, expected_rows=3, provider=mock_provider
        )

        # All 3 valid line_numbers present — no reconciliation triggered
        assert len(result.line_items) == 4  # Original items unchanged


# ============================================================================
# Quantity Auto-Correction Tests
# ============================================================================


class TestQuantityAutoCorrection:
    """Tests for _auto_correct_quantities() method."""

    def _make_extractor(self, settings):
        mock_provider = MockProvider(name="openai", extract_text_result=None)
        return LLMExtractor(providers=[mock_provider])

    @patch("src.extraction.llm_extractor.get_settings")
    def test_clean_integer_correction(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """qty=1, unit=100, extended=300 -> qty corrected to 3."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        items = [
            LineItem(
                line_number=1,
                description="Widget",
                quantity=1,
                unit_price=100.00,
                extended_price=300.00,
            ),
        ]

        result = extractor._auto_correct_quantities(items)

        assert result[0].quantity == 3

    @patch("src.extraction.llm_extractor.get_settings")
    def test_skips_non_integer_ratio(
        self, mock_get_settings, mock_settings_no_anthropic
    ):
        """qty=1, unit=100, extended=250 -> no correction (2.5 is not integer)."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        items = [
            LineItem(
                line_number=1,
                description="Bundle",
                quantity=1,
                unit_price=100.00,
                extended_price=250.00,
            ),
        ]

        result = extractor._auto_correct_quantities(items)

        assert result[0].quantity == 1  # Unchanged

    @patch("src.extraction.llm_extractor.get_settings")
    def test_skips_credits(self, mock_get_settings, mock_settings_no_anthropic):
        """Negative extended_price (credit) -> no correction."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        items = [
            LineItem(
                line_number=1,
                description="Credit",
                quantity=1,
                unit_price=500.00,
                extended_price=-500.00,
            ),
        ]

        result = extractor._auto_correct_quantities(items)

        assert result[0].quantity == 1  # Unchanged
        assert result[0].extended_price == -500.00

    @patch("src.extraction.llm_extractor.get_settings")
    def test_skips_zero_unit_price(self, mock_get_settings, mock_settings_no_anthropic):
        """unit_price=0 -> no division by zero, no correction."""
        mock_get_settings.return_value = mock_settings_no_anthropic
        extractor = self._make_extractor(mock_settings_no_anthropic)

        items = [
            LineItem(
                line_number=1,
                description="Free item",
                quantity=1,
                unit_price=0.00,
                extended_price=0.00,
            ),
        ]

        result = extractor._auto_correct_quantities(items)

        assert result[0].quantity == 1  # Unchanged
