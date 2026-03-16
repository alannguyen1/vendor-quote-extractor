"""
Mock LLM response fixtures for unit testing.

These fixtures represent the expected extraction results from the 5 test PDFs:
- mixed_category_quote.pdf: $32,198.00 grand total
- discounted_hardware_quote.pdf: $26,763.19 grand total (includes tax)
- enterprise_term_quote.pdf: $143,391.60, 60-month term
- service_order_with_credits.pdf: TCV $597,312, NRC $45,300
- multi_page_quote.pdf: $314,432.95, 50+ items

Usage:
    from tests.mocks.llm_responses import MIXED_CATEGORY_QUOTE, get_mock_extractor

    # Use fixtures in assertions
    assert MIXED_CATEGORY_QUOTE.amounts.grand_total == 32198.00

    # Get a mocked extractor that returns a specific quote
    extractor = get_mock_extractor(MIXED_CATEGORY_QUOTE)

Why Mock LLM Responses:
- Unit tests should be deterministic and fast (no API latency)
- Avoid LLM API costs in CI/CD pipelines
- Test edge cases without crafting actual PDFs
- Validate post-extraction logic (validation, metadata) independently
"""

from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from src.extraction.llm_extractor import ExtractionResult
from src.schemas import (
    Amounts,
    CommercialTerms,
    Entity,
    ItemType,
    LineItem,
    TermType,
    VendorQuote,
)

if TYPE_CHECKING:
    from src.extraction.llm_extractor import LLMExtractor


# ============================================================================
# Mixed Category Quote Fixture
# Expected: $32,198.00 grand total, tax $1,398.00
# ============================================================================

MIXED_CATEGORY_QUOTE = VendorQuote(
    quote_id="MCQ-2024-0001",
    quote_date=date(2024, 1, 15),
    valid_until=date(2024, 2, 15),
    currency="USD",
    vendor=Entity(
        name="Northwind Systems",
        address="123 Tech Drive, Silicon Valley, CA 94025",
        phone="(555) 123-4567",
        email="sales@northwindsystems.example",
    ),
    customer=Entity(
        name="Acme Retail Group",
        address="456 Business Blvd, San Francisco, CA 94105",
    ),
    line_items=[
        LineItem(
            line_number=1,
            description="Cisco Catalyst 9300-48P-HW",
            sku="C9300-48P-HW",
            manufacturer="Cisco",
            quantity=4,
            unit_price=5500.00,
            extended_price=22000.00,
            item_type=ItemType.hardware,
        ),
        LineItem(
            line_number=2,
            description="LIC-DNA-Essentials 3-Year",
            sku="LIC-DNA-E-3Y",
            manufacturer="Cisco",
            quantity=4,
            unit_price=1200.00,
            extended_price=4800.00,
            item_type=ItemType.software,
            term_type=TermType.multi_year,
            term_length_months=36,
        ),
        LineItem(
            line_number=3,
            description="Implementation Services",
            sku="SVC-INSTALL",
            manufacturer=None,
            quantity=1,
            unit_price=4000.00,
            extended_price=4000.00,
            item_type=ItemType.services,
        ),
    ],
    amounts=Amounts(
        subtotal=30800.00,
        discounts=0.0,
        shipping=0.0,
        tax=1398.00,
        grand_total=32198.00,
    ),
    commercial_terms=CommercialTerms(
        payment_terms="Net 30",
        subscription_term_months=36,
    ),
)


# ============================================================================
# Discounted Hardware Quote Fixture
# Expected: $26,763.19 grand total (subtotal $24,858.02 + tax $1,905.17)
# All hardware items
# ============================================================================

DISCOUNTED_HARDWARE_QUOTE = VendorQuote(
    quote_id="DHQ-2024-0142",
    quote_date=date(2024, 2, 1),
    valid_until=date(2024, 3, 1),
    currency="USD",
    vendor=Entity(
        name="Summit Equipment Supply",
        address="789 Industrial Way, Austin, TX 78701",
        phone="(512) 555-7890",
        email="quotes@summitequipment.example",
    ),
    customer=Entity(
        name="Atlas Manufacturing Group",
        address="321 Factory Lane, Dallas, TX 75201",
    ),
    line_items=[
        LineItem(
            line_number=1,
            description="Heavy Duty Server Rack 42U",
            sku="RACK-42U-HW",
            quantity=2,
            unit_price=2500.00,
            extended_price=5000.00,
            item_type=ItemType.hardware,
        ),
        LineItem(
            line_number=2,
            description="Dell PowerEdge R750-HW",
            sku="PE-R750-HW",
            manufacturer="Dell",
            quantity=4,
            unit_price=4200.00,
            extended_price=16800.00,
            item_type=ItemType.hardware,
        ),
        LineItem(
            line_number=3,
            description="Structured Cabling Kit",
            sku="CABLE-KIT-HW",
            quantity=1,
            unit_price=1558.02,
            extended_price=1558.02,
            item_type=ItemType.hardware,
        ),
        LineItem(
            line_number=4,
            description="Installation Labor",
            sku="SVC-LABOR",
            quantity=8,
            unit_price=187.50,
            extended_price=1500.00,
            item_type=ItemType.services,
        ),
    ],
    amounts=Amounts(
        subtotal=24858.02,
        discounts=0.0,
        shipping=0.0,
        tax=1905.17,
        grand_total=26763.19,
    ),
    commercial_terms=CommercialTerms(payment_terms="Net 30"),
)


# ============================================================================
# Enterprise Term Quote Fixture
# Expected: $143,391.60, 60-month term (date range: 11/22/2024 - 11/21/2029)
# ============================================================================

ENTERPRISE_TERM_QUOTE = VendorQuote(
    quote_id="ETQ-2024-1001",
    quote_date=date(2024, 11, 1),
    valid_until=date(2024, 12, 31),
    currency="USD",
    vendor=Entity(
        name="Bluewave Software",
        address="100 Innovation Park, Orlando, FL 32801",
        email="sales@bluewave.example",
    ),
    customer=Entity(
        name="Global Retail Group",
    ),
    line_items=[
        LineItem(
            line_number=1,
            description="LIC-Enterprise Platform Annual",
            sku="LIC-ENT-ANNUAL",
            quantity=5,
            unit_price=19878.32,
            extended_price=99391.60,
            item_type=ItemType.software,
            term_type=TermType.annual,
            term_length_months=60,
        ),
        LineItem(
            line_number=2,
            description="Premium Support Package",
            sku="SVC-PREMIUM",
            quantity=5,
            unit_price=4000.00,
            extended_price=20000.00,
            item_type=ItemType.services,
            term_type=TermType.annual,
            term_length_months=60,
        ),
        LineItem(
            line_number=3,
            description="Implementation Services",
            sku="SVC-IMPL",
            quantity=1,
            unit_price=24000.00,
            extended_price=24000.00,
            item_type=ItemType.services,
            term_type=TermType.one_time,
        ),
    ],
    amounts=Amounts(
        subtotal=143391.60,
        discounts=0.0,
        shipping=0.0,
        tax=0.0,
        grand_total=143391.60,
    ),
    commercial_terms=CommercialTerms(
        payment_terms="Net 45",
        subscription_term_months=60,
        auto_renew=True,
        coverage_start=date(2024, 11, 22),
        coverage_end=date(2029, 11, 21),
    ),
)


# ============================================================================
# Service Order With Credits Fixture
# Expected: TCV $597,312, NRC $45,300, 36-month term
# Contains negative credit line item
# ============================================================================

SERVICE_ORDER_WITH_CREDITS_QUOTE = VendorQuote(
    quote_id="SOC-2024-0789",
    quote_date=date(2024, 3, 15),
    valid_until=date(2024, 4, 15),
    currency="USD",
    vendor=Entity(
        name="Cedar Managed Services",
        address="500 Enterprise Blvd, Chicago, IL 60601",
        phone="(312) 555-9000",
        email="orders@cedarms.example",
    ),
    customer=Entity(
        name="Beacon Financial Group",
        address="200 Wall Street, New York, NY 10005",
    ),
    line_items=[
        LineItem(
            line_number=1,
            description="Managed Infrastructure Services - 36 Month",
            sku="MNS-36M",
            quantity=36,
            unit_price=12000.00,
            extended_price=432000.00,
            item_type=ItemType.services,
            term_type=TermType.monthly,
            term_length_months=36,
        ),
        LineItem(
            line_number=2,
            description="Cisco UCS Server Infrastructure",
            sku="UCS-INFRA-HW",
            manufacturer="Cisco",
            quantity=1,
            unit_price=165312.00,
            extended_price=165312.00,
            item_type=ItemType.hardware,
        ),
        LineItem(
            line_number=3,
            description="Initial Setup & Migration NRC",
            sku="NRC-SETUP",
            quantity=1,
            unit_price=45300.00,
            extended_price=45300.00,
            item_type=ItemType.services,
            term_type=TermType.one_time,
        ),
        LineItem(
            line_number=4,
            description="Cisco Trade-In Credit",
            sku="CREDIT-TRADE",
            manufacturer="Cisco",
            quantity=1,
            unit_price=45300.00,  # Unit price is positive
            extended_price=-45300.00,  # Extended price is negative for credits
            item_type=ItemType.credit,
        ),
    ],
    amounts=Amounts(
        subtotal=597312.00,
        discounts=0.0,
        shipping=0.0,
        tax=0.0,
        grand_total=597312.00,
        net_price_total=552012.00,  # After credit applied
    ),
    commercial_terms=CommercialTerms(
        payment_terms="Net 60",
        subscription_term_months=36,
    ),
)


# ============================================================================
# Multi-Page Quote Fixture
# Expected: $314,432.95, 50+ line items (multi-page)
# ============================================================================

# Create 50+ line items for the multi-page quote fixture
_multi_page_line_items = []
for i in range(1, 52):
    item_type = ItemType.hardware if i % 3 != 0 else ItemType.software
    unit_price = 5000.00 + (i * 100)
    quantity = 1 + (i % 3)
    _multi_page_line_items.append(
        LineItem(
            line_number=i,
            description=f"Automation Component Part #{i:03d}",
            sku=f"AUTO-{i:04d}",
            manufacturer="Various",
            quantity=quantity,
            unit_price=unit_price,
            extended_price=round(unit_price * quantity, 2),
            item_type=item_type,
        )
    )

# Calculate totals for the multi-page quote fixture
# Note: The mock line items sum to a different value than the actual PDF
# We use discounts to make the math consistent with the expected grand_total
_multi_page_subtotal = sum(item.extended_price for item in _multi_page_line_items)
_multi_page_tax = round(_multi_page_subtotal * 0.0625, 2)  # 6.25% tax
_multi_page_shipping = 1500.00
_multi_page_grand_total = 314432.95  # Expected value from fixture data
# Calculate discount to balance: subtotal + tax + shipping - discounts = grand_total
_multi_page_discounts = round(
    _multi_page_subtotal
    + _multi_page_tax
    + _multi_page_shipping
    - _multi_page_grand_total,
    2,
)

MULTI_PAGE_QUOTE = VendorQuote(
    quote_id="MPQ-2024-0500",
    quote_date=date(2024, 4, 1),
    valid_until=date(2024, 5, 1),
    currency="USD",
    vendor=Entity(
        name="Riverstone Distribution",
        address="750 Robot Way, Detroit, MI 48201",
        email="sales@riverstone.example",
    ),
    customer=Entity(
        name="Large Manufacturing Co.",
    ),
    line_items=_multi_page_line_items,
    amounts=Amounts(
        subtotal=round(_multi_page_subtotal, 2),
        discounts=_multi_page_discounts,
        shipping=_multi_page_shipping,
        tax=_multi_page_tax,
        grand_total=_multi_page_grand_total,
        list_price_total=round(_multi_page_subtotal, 2),
        net_price_total=_multi_page_grand_total,
    ),
    commercial_terms=CommercialTerms(payment_terms="Net 30"),
)


# ============================================================================
# Minimal Quote Fixture (for edge case testing)
# ============================================================================

MINIMAL_QUOTE = VendorQuote(
    currency="USD",
    vendor=Entity(name="Minimal Vendor"),
    line_items=[
        LineItem(
            line_number=1,
            description="Basic Item",
            quantity=1,
            unit_price=100.00,
            extended_price=100.00,
        )
    ],
    amounts=Amounts(
        subtotal=100.00,
        grand_total=100.00,
    ),
)


# ============================================================================
# Mock Extractor Factory
# ============================================================================


def get_mock_extractor(quote: VendorQuote, confidence: float = 0.95) -> "LLMExtractor":
    """
    Create a mocked LLMExtractor that returns the specified quote.

    This factory creates an extractor with mocked OpenAI/Anthropic clients
    that immediately return the given quote without making API calls.

    Args:
        quote: The VendorQuote to return from extract() calls
        confidence: The confidence score to return (default 0.95)

    Returns:
        A mocked LLMExtractor instance

    Usage:
        extractor = get_mock_extractor(MIXED_CATEGORY_QUOTE)
        result = await extractor.extract(text="...", tables=[], page_count=1)
        assert result.quote.amounts.grand_total == 32198.00
    """
    from src.extraction.llm_extractor import LLMExtractor

    with (
        patch("src.extraction.llm_extractor.get_settings") as mock_settings,
        patch("src.extraction.llm_extractor.instructor.from_openai") as mock_openai,
    ):
        # Configure mock settings
        settings = MagicMock()
        settings.openai_api_key = "sk-mock-key"
        settings.anthropic_api_key = None
        settings.llm_model = "gpt-4o-mock"
        settings.anthropic_model = None
        settings.has_anthropic_key.return_value = False
        mock_settings.return_value = settings

        # Configure mock OpenAI client to return the quote
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=quote)
        mock_openai.return_value = mock_client

        return LLMExtractor()


def get_mock_extraction_result(
    quote: VendorQuote,
    confidence: float = 0.95,
    fallback_used: bool = False,
    error: str | None = None,
) -> ExtractionResult:
    """
    Create a mock ExtractionResult with the specified quote.

    This is useful for testing code that consumes ExtractionResult objects
    without needing to run the full extractor.

    Args:
        quote: The VendorQuote to include in the result
        confidence: Confidence score (default 0.95)
        fallback_used: Whether fallback provider was used (default False)
        error: Optional error message

    Returns:
        ExtractionResult with the specified values
    """
    return ExtractionResult(
        quote=quote,
        confidence=confidence,
        error=error,
        fallback_used=fallback_used,
    )


# ============================================================================
# Quote Lookup by Filename
# ============================================================================

FIXTURE_QUOTES = {
    "mixed_category_quote.pdf": MIXED_CATEGORY_QUOTE,
    "discounted_hardware_quote.pdf": DISCOUNTED_HARDWARE_QUOTE,
    "enterprise_term_quote.pdf": ENTERPRISE_TERM_QUOTE,
    "service_order_with_credits.pdf": SERVICE_ORDER_WITH_CREDITS_QUOTE,
    "multi_page_quote.pdf": MULTI_PAGE_QUOTE,
}


def get_quote_for_fixture(filename: str) -> VendorQuote | None:
    """
    Get the expected mock quote for a given fixture filename.

    Args:
        filename: The PDF fixture filename

    Returns:
        The corresponding VendorQuote mock, or None if not found
    """
    return FIXTURE_QUOTES.get(filename)
