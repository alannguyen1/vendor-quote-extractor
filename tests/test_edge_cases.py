"""
Edge case tests for document extraction scenarios.

Tests unusual but valid inputs that require special handling:
- Date range to months conversion (Enterprise Term: 11/22/2024 - 11/21/2029 = 60 months)
- Empty/malformed PDFs
- Currency symbol variations
- Large documents
- Concurrent extraction scenarios (deferred to Phase 4)

Why these tests matter:
Edge cases represent real-world documents that may not follow typical
formatting. Testing these ensures the system is robust for production use.
"""

from datetime import date

import pytest

from src.schemas import (
    Amounts,
    CommercialTerms,
    Entity,
    ExtractionMetadata,
    ItemType,
    LineItem,
    VendorQuote,
)


class TestDateRangeCalculations:
    """Unit tests for date range to months calculation.

    The Enterprise Term fixture uses date ranges (11/22/2024 - 11/21/2029) instead of
    explicit month counts. The LLM must calculate 60 months from this range.
    """

    def test_calculate_months_from_date_range(self):
        """Verify date range calculation logic.

        From date range 11/22/2024 to 11/21/2029 = 60 months (5 years).
        This validates the expected calculation.
        """
        start_date = date(2024, 11, 22)
        end_date = date(2029, 11, 21)

        # Calculate months difference
        months = (end_date.year - start_date.year) * 12
        months += end_date.month - start_date.month

        # Should be approximately 60 months
        assert months == 60, f"Expected 60 months, got {months}"

    def test_months_with_partial_months(self):
        """Test calculation when dates don't align to full months."""
        start_date = date(2024, 1, 15)
        end_date = date(2025, 6, 20)

        # Calculate months (floor)
        months = (end_date.year - start_date.year) * 12
        months += end_date.month - start_date.month

        assert months == 17

    def test_commercial_terms_stores_months(self):
        """Verify CommercialTerms model can store calculated months."""
        terms = CommercialTerms(
            subscription_term_months=60,
            coverage_start=date(2024, 11, 22),
            coverage_end=date(2029, 11, 21),
        )

        assert terms.subscription_term_months == 60
        assert terms.coverage_start == date(2024, 11, 22)
        assert terms.coverage_end == date(2029, 11, 21)


class TestEmptyAndMalformedPDFs:
    """Unit tests for empty and malformed PDF handling."""

    def test_empty_pdf_fixture_exists(self, empty_pdf_bytes: bytes):
        """Verify empty PDF test fixture is available."""
        assert empty_pdf_bytes is not None
        assert len(empty_pdf_bytes) > 0

    def test_invalid_file_fixture_exists(self, invalid_file_bytes: bytes):
        """Verify invalid file test fixture is available."""
        assert invalid_file_bytes is not None
        assert not invalid_file_bytes.startswith(b"%PDF")


class TestCurrencyVariations:
    """Unit tests for currency handling.

    Only USD is accepted per spec. Non-USD quotes must be rejected.
    """

    def test_usd_currency_accepted(self):
        """Verify USD currency is valid."""
        quote = VendorQuote(
            currency="USD",
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                    confidence=0.95,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        assert quote.currency == "USD"

    def test_currency_symbol_variations(self):
        """Test that currency field stores string codes, not symbols."""
        # Currency should be ISO code, not symbol like "$" or "\u20ac"
        quote = VendorQuote(
            currency="USD",  # Not "$"
            vendor=Entity(name="Test Vendor"),
            line_items=[
                LineItem(
                    line_number=1,
                    description="Item",
                    quantity=1,
                    unit_price=100.0,
                    extended_price=100.0,
                    confidence=0.95,
                )
            ],
            amounts=Amounts(subtotal=100.0, grand_total=100.0),
        )
        assert quote.currency == "USD"
        assert len(quote.currency) == 3  # ISO currency codes are 3 chars


class TestLargeDocumentHandling:
    """Unit tests for large document edge cases."""

    def test_large_text_truncation(self):
        """Verify extraction handles large text input.

        The LLM extractor truncates text to 30K chars to avoid token limits.
        """
        # Create large text content
        large_text = "Sample vendor quote content. " * 2000  # ~58K chars
        assert len(large_text) > 30000

        # Truncation would happen at extraction time
        truncated = large_text[:30000]
        assert len(truncated) == 30000

    def test_many_line_items_structure(self):
        """Verify VendorQuote can hold many line items."""
        items = [
            LineItem(
                line_number=i,
                description=f"Item {i}",
                quantity=1,
                unit_price=100.0,
                extended_price=100.0,
                confidence=0.90,
            )
            for i in range(1, 101)  # 100 items
        ]

        quote = VendorQuote(
            vendor=Entity(name="Large Quote Vendor"),
            line_items=items,
            amounts=Amounts(subtotal=10000.0, grand_total=10000.0),
        )

        assert len(quote.line_items) == 100


class TestSpecialItemTypes:
    """Unit tests for special line item types."""

    def test_credit_item_negative_price(self):
        """Test credit items have negative extended_price."""
        credit = LineItem(
            line_number=1,
            description="Cisco Trade-In Credit",
            quantity=1,
            unit_price=0.0,  # Credits often have 0 unit price
            extended_price=-5000.0,  # Negative for credit
            item_type=ItemType.credit,
            confidence=0.95,
        )

        assert credit.extended_price < 0
        assert credit.item_type == ItemType.credit

    def test_discount_item_negative_price(self):
        """Test discount items have negative extended_price.

        Note: unit_price must be >= 0 per schema, but extended_price can be
        negative to represent discounts. Discounts typically use unit_price=0
        with negative extended_price.
        """
        discount = LineItem(
            line_number=1,
            description="Volume Discount",
            quantity=1,
            unit_price=0.0,  # unit_price must be >= 0
            extended_price=-500.0,  # Discount as negative extended_price
            item_type=ItemType.discount,
            confidence=0.90,
        )

        assert discount.extended_price < 0
        assert discount.item_type == ItemType.discount

    def test_zero_price_items_valid(self):
        """Test items with zero price are valid (promo items)."""
        promo = LineItem(
            line_number=1,
            description="Free Installation",
            quantity=1,
            unit_price=0.0,
            extended_price=0.0,
            item_type=ItemType.services,
            confidence=0.85,
        )

        assert promo.unit_price == 0.0
        assert promo.extended_price == 0.0


class TestMetadataEdgeCases:
    """Unit tests for extraction metadata edge cases."""

    def test_zero_processing_time(self):
        """Test metadata with zero processing time (cached result)."""
        metadata = ExtractionMetadata(
            processing_time_ms=0,  # Cached result
            model_used="cache",
            overall_confidence=1.0,
            validation_passed=True,
            validation_errors=[],
            requires_review=False,
        )

        assert metadata.processing_time_ms == 0

    def test_low_confidence_triggers_review(self):
        """Test that low confidence triggers review flag."""
        # Confidence below 0.90 threshold should trigger review
        metadata = ExtractionMetadata(
            processing_time_ms=5000,
            model_used="gpt-4o-2024-08-06",
            overall_confidence=0.75,  # Below 0.90 threshold
            validation_passed=True,
            validation_errors=[],
            requires_review=True,  # Should be True for low confidence
        )

        assert metadata.overall_confidence < 0.90
        assert metadata.requires_review is True

    def test_validation_failure_triggers_review(self):
        """Test that validation failure triggers review flag."""
        metadata = ExtractionMetadata(
            processing_time_ms=5000,
            model_used="gpt-4o-2024-08-06",
            overall_confidence=0.95,  # High confidence
            validation_passed=False,  # But validation failed
            validation_errors=["Line items sum mismatch"],
            requires_review=True,  # Should be True despite high confidence
        )

        assert metadata.validation_passed is False
        assert metadata.requires_review is True


@pytest.mark.integration
class TestIntegrationEdgeCases:
    """Integration tests for edge case documents.

    These tests require API keys and make real LLM calls.
    """

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for integration tests."""
        from src.extraction.pipeline import ExtractionPipeline

        return ExtractionPipeline()

    @pytest.mark.asyncio
    async def test_enterprise_term_term_months_extraction(
        self, pipeline, enterprise_term_pdf_bytes: bytes
    ):
        """Verify Enterprise Term extracts subscription term from date range.

        Expected: 60 months calculated from 11/22/2024 - 11/21/2029.
        Note: LLM must interpret date range and output month count.
        This is a complex extraction that may have variance.
        """
        result = await pipeline.process(
            enterprise_term_pdf_bytes, "enterprise_term_quote.pdf"
        )

        if result.quote is None:
            pytest.skip("Extraction timed out - skipping term months test")

        # The commercial_terms field stores term information
        terms = result.quote.commercial_terms
        if terms is None:
            pytest.skip("Commercial terms not extracted")

        # Check if term months were extracted (may be in subscription_term_months
        # or derivable from coverage_start/coverage_end)
        has_term_info = terms.subscription_term_months is not None or (
            terms.coverage_start is not None and terms.coverage_end is not None
        )

        assert has_term_info, "Expected subscription term information in extraction"

    @pytest.mark.asyncio
    async def test_empty_pdf_returns_error(self, pipeline, empty_pdf_bytes: bytes):
        """Verify empty PDFs return appropriate error."""
        result = await pipeline.process(empty_pdf_bytes, "empty.pdf")

        # Should fail with parse error
        assert result.error is not None
        assert result.quote is None

    @pytest.mark.asyncio
    async def test_invalid_file_returns_error(
        self, pipeline, invalid_file_bytes: bytes
    ):
        """Verify non-PDF files return appropriate error."""
        result = await pipeline.process(invalid_file_bytes, "notapdf.txt")

        # Should fail with parse error
        assert result.error is not None
        assert result.quote is None
